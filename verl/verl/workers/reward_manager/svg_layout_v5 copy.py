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
from verl.utils.svg_utils import export_svg_to_img


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

        action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        if 'env_reward' in data.batch.keys():
            extra_advantage_tensor += data.batch['env_reward']
            print(f' [DEBUG env_reward] mean={extra_advantage_tensor.mean().item()}, min={extra_advantage_tensor.min().item()}, max={extra_advantage_tensor.max().item()}')

        # indexed_data = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            # bg_image = data.non_tensor_batch["origin_multi_modal_data"][i]["image"][0]
            # tool_result_images = data.non_tensor_batch["tool_result_list"][i].get('origin_multi_modal_data', {}).get('image', [])
            tool_result_images_scores = data.non_tensor_batch["tool_result_list"][i].get('tool_result_images_scores', [])

            # indexed_data[extra_info["index"]].append(1 if len(tool_result_images_scores) > 2 else 0)

        # RaRTK = {}
        # for key, value in indexed_data.items():
        #     if len(value) > 0:
        #         RaRTK[key] = sum(value) / len(value)
        #     else:
        #         RaRTK[key] = 0.0

        # for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            tool_result_images_scores = data.non_tensor_batch["tool_result_list"][i].get('tool_result_images_scores', [])
            # RaRTH_reward = 2 * max(0.0, self.RaRTK_threshold - RaRTK.get(data_item.non_tensor_batch["extra_info"]["index"], 0.0)) if len(tool_result_images_scores) > 2 else 0.0
            
            reward = tool_result_images_scores[0]   # -1是最后的，0为第一轮的
            
            reward_tensor[i, valid_response_length - 1] = reward
            # extra_advantage_tensor[i] += RaRTH_reward
            
            reward_extra_info["index"].append(extra_info["index"])
            reward_extra_info["reward_parts"].append({
                "answer_score": tool_result_images_scores[-1],
                # "RaRTH_reward": RaRTH_reward,
                "first": tool_result_images_scores[0],
                "tool_score_mean": sum(tool_result_images_scores[:-1]) / (len(tool_result_images_scores) - 1) if len(tool_result_images_scores) > 1 else 0,
                "second-first": tool_result_images_scores[1] - tool_result_images_scores[0] if len(tool_result_images_scores) > 1 else 0,
                "final-first": tool_result_images_scores[-1] - tool_result_images_scores[0],
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
