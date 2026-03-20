# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import base64
import random
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import defaultdict
from typing import Optional, List, Dict, Any
import multiprocessing

import torch

from verl import DataProto


def extract_last_turn_response_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """
    从响应掩码中提取最后一轮回答的掩码
    
    Args:
        response_mask: 响应掩码张量，形状为 (bs, response_length)，
                      其中1表示对应位置是响应内容，0表示非响应内容（轮次分隔）
    
    Returns:
        last_turn_response_mask: 最后一轮响应的掩码张量，形状与输入相同，
                                仅最后一轮响应位置为1，其余为0
    """
    batch_size, response_length = response_mask.shape
    
    # 初始化最后一轮掩码
    last_turn_response_mask = torch.zeros_like(response_mask)
    
    # 找到每个样本的所有非零位置（响应内容）
    nonzero_positions = response_mask.nonzero()  # 形状为 (num_nonzero, 2)，每行是 (batch_idx, position)
    
    if nonzero_positions.numel() == 0:
        return last_turn_response_mask  # 没有任何响应内容
    
    # 按批次分组，找到每个样本的最后一个响应位置
    batch_indices = nonzero_positions[:, 0]
    positions = nonzero_positions[:, 1]
    
    # 找到每个样本的最后一个非零位置（最后一轮的结束位置）
    last_nonzero_idx = []
    for i in range(batch_size):
        # 找到当前样本的所有非零位置
        sample_positions = positions[batch_indices == i]
        if sample_positions.numel() > 0:
            last_nonzero_idx.append(sample_positions.max().item())
        else:
            last_nonzero_idx.append(-1)  # 没有响应内容
    
    # 找到每个样本的最后一个零位置（最后一轮的开始位置）
    # 先创建反向掩码（0和1互换）
    inverse_mask = 1 - response_mask
    zero_positions = inverse_mask.nonzero()  # 形状为 (num_zero, 2)，每行是 (batch_idx, position)
    
    last_zero_idx = [-1] * batch_size
    if zero_positions.numel() > 0:
        zero_batch_indices = zero_positions[:, 0]
        zero_positions_vals = zero_positions[:, 1]
        
        for i in range(batch_size):
            # 找到当前样本的所有零位置
            sample_zero_positions = zero_positions_vals[zero_batch_indices == i]
            if sample_zero_positions.numel() > 0:
                # 找到小于最后一个非零位置的最大零位置
                valid_zeros = sample_zero_positions[sample_zero_positions < last_nonzero_idx[i]]
                if valid_zeros.numel() > 0:
                    last_zero_idx[i] = valid_zeros.max().item()
    
    # 填充最后一轮掩码
    for i in range(batch_size):
        if last_nonzero_idx[i] == -1:
            continue  # 没有响应内容
        
        # 最后一轮的起始位置：最后一个零位置 + 1（如果没有零位置则从0开始）
        start = last_zero_idx[i] + 1
        # 最后一轮的结束位置：最后一个非零位置 + 1（切片是左闭右开）
        end = last_nonzero_idx[i] + 1
        
        # 确保起始位置有效
        start = max(0, start)
        # 确保结束位置不超过响应长度
        end = min(response_length, end)
        
        if start < end:  # 确保有效区间
            last_turn_response_mask[i, start:end] = 1.0
    
    return last_turn_response_mask


class SvgLayoutRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.RaRTK_threshold = 0.5  # Threshold for RaRTK

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra_advantage_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        response_mask = data.batch["response_mask"]
        last_turn_response_mask = extract_last_turn_response_mask(response_mask)
        

        if 'env_reward' in data.batch.keys():
            extra_advantage_tensor += data.batch['env_reward']
            print(f' [DEBUG env_reward] mean={extra_advantage_tensor.mean().item()}, min={extra_advantage_tensor.min().item()}, max={extra_advantage_tensor.max().item()}')

        indexed_data = defaultdict(lambda: defaultdict(list))
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            # bg_image = data.non_tensor_batch["origin_multi_modal_data"][i]["image"][0]
            # tool_result_images = data.non_tensor_batch["tool_result_list"][i].get('origin_multi_modal_data', {}).get('image', [])
            tool_result_images_scores = data.non_tensor_batch["tool_result_list"][i].get("tool_result_images_scores", [])

            indexed_data[extra_info["index"]]["tool_result_images_scores"].append(1 if len(tool_result_images_scores) > 2 else 0)
            indexed_data[extra_info["index"]]["answer_scores"].append(tool_result_images_scores[-1])

        # RaRTK = {}
        # for key, value in indexed_data.items():
        #     if len(value) > 0:
        #         RaRTK[key] = sum(value) / len(value)
        #     else:
        #         RaRTK[key] = 0.0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            tool_result_images_scores = data.non_tensor_batch["tool_result_list"][i].get('tool_result_images_scores', [])
            # RaRTK_reward = max(0.0, self.RaRTK_threshold - RaRTK.get(data_item.non_tensor_batch["extra_info"]["index"], 0.0)) if len(tool_result_images_scores) > 2 else 0.0
            
            reward = tool_result_images_scores[0]   # -1是最后的，0为第一轮的
            
            reward_tensor[i, valid_response_length - 1] = reward
            # extra_advantage_tensor[i] += RaRTK_reward
            # answer_punishment = tool_result_images_scores[-1] - max(indexed_data[extra_info["index"]]["answer_scores"]) if len(tool_result_images_scores) < 4 else 0.0  # 最多是5，小于5是不到最大都惩罚，小于4是最后两次有灵活
            answer_punishment = (tool_result_images_scores[-1] - max(indexed_data[extra_info["index"]]["answer_scores"])) * max(0.0, 1.0 - len(tool_result_images_scores) / 4.0) if len(tool_result_images_scores) > 1 else 0.0  # 最多是5，小于5是不到最大都惩罚，小于4是最后两次有灵活
            extra_advantage_tensor[i] += answer_punishment * last_turn_response_mask[i]
            
            reward_extra_info["index"].append(extra_info["index"])
            reward_extra_info["reward_parts"].append({
                "answer_score": tool_result_images_scores[-1],
                # "RaRTK_reward": RaRTK_reward,
                "first": tool_result_images_scores[0],
                "tool_score_mean": sum(tool_result_images_scores[:-1]) / (len(tool_result_images_scores) - 1) if len(tool_result_images_scores) > 1 else 0,
                "second-first": tool_result_images_scores[1] - tool_result_images_scores[0] if len(tool_result_images_scores) > 1 else 0,
                "final-first": tool_result_images_scores[-1] - tool_result_images_scores[0],
                "valid-second-first": tool_result_images_scores[1] - tool_result_images_scores[0] if len(tool_result_images_scores) > 2 else -100,
                "valid-final-first": tool_result_images_scores[-1] - tool_result_images_scores[0] if len(tool_result_images_scores) > 2 else -100,
                "answer_punishment": answer_punishment,
                "tool_call_cnt": len(tool_result_images_scores),
                "tool_result_images_scores": tool_result_images_scores,
            })


        reward_extra_info["extra_advantage_tensor"] = extra_advantage_tensor
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
