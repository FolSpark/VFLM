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
        
        self.servers = "http://10.0.2.226:8999/compute_score"
        self.api_num = 64

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        # if 'env_reward' in data.batch.keys():
            # reward_tensor += data.batch['env_reward']
            # print(f' [DEBUG env_reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        already_print_data_sources = {}

        params = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            bg_image = data.non_tensor_batch["origin_multi_modal_data"][i]["image"][0]
            
            tool_result_images = data.non_tensor_batch["tool_result_list"][i].get('origin_multi_modal_data', {}).get('image', [])

            params.append({
                "index": i,
                "response_str": response_str,
                "text_content": extra_info["text_content"],
                "bg_image": bg_image,
                "tool_result_images": tool_result_images
            })

        # 并行请求
        with multiprocessing.Pool(processes=min(self.api_num, len(data))) as pool:
            for score in tqdm(pool.imap(self.compute_score, params), total=len(params), desc="Computing reward scores"):
                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[score["index"], valid_response_length - 1] = reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    # print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def get_image_b64_str(self, image: Image.Image) -> str:
        """
        转换为base64编码字符串
        :param image: PIL.Image对象
        :return: base64编码的字符串
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # 返回data URI格式的字符串
        return f"data:image;base64,{encoded_image_text}"

    def extract_answer(self, action_string: str) -> Optional[str]:
        answer = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer.group(1) if answer else None

    def compute_image_score(self, text_content, bg_img_b64_str, rendered_img_b64_str) -> float:
        for cnt in range(3):  # Retry up to 3 times
            try:
                # url = self.api_pool.get()
                url = self.servers
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "prompt": text_content,
                    "image1": bg_img_b64_str,
                    "image2": rendered_img_b64_str,
                }
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=600)
                response = response.json()
                # self.api_pool.put(url)  # Put the URL back into the pool for reuse
                return response["score"]
            except Exception as e:
                print(f"Error count {cnt} during API call: {e}")
                # self.api_pool.put(url)  # Put the URL back into the pool for reuse
                continue
        
    def compute_score(self, param) -> Dict[str, Any]:
        """
        Compute the score based on the response string and the background image.
        
        Args:
            response_str: The response string from the model.
            text_content: The text content to be used for scoring.
            bg_image: The background image as a PIL Image object.
        
        Returns:
            A dictionary containing the index and the computed score.
        """
        response_str = param["response_str"]
        text_content = param["text_content"]
        bg_image = param["bg_image"]
        tool_result_images = param["tool_result_images"]
        
        is_format_error = False
        answer_str = self.extract_answer(response_str)
        if not answer_str:
            is_format_error = True
        else:
            if answer_str.startswith("```svg\n"):
                answer_str = answer_str[7:]
            if answer_str.endswith("```"):
                answer_str = answer_str[:-3]
            if not (answer_str.startswith("<svg") and answer_str.strip().endswith("</svg>")):
                is_format_error = True
        
        if is_format_error:
            format_reward = -1.0
            black_image = Image.new("RGB", bg_image.size, (0, 0, 0))
            answer_score = self.compute_image_score(
                text_content=text_content,
                bg_img_b64_str=self.get_image_b64_str(bg_image),
                rendered_img_b64_str=self.get_image_b64_str(black_image)
            )
        else:
            format_reward = 0.0
            try:
                rendered_img = export_svg_to_img(answer_str, bg_image)
            except Exception as e:
                print(f"Error during SVG rendering: {e}")
                rendered_img = Image.new("RGB", bg_image.size, (0, 0, 0))
            answer_score = self.compute_image_score(
                text_content=text_content,
                bg_img_b64_str=self.get_image_b64_str(bg_image),
                rendered_img_b64_str=self.get_image_b64_str(rendered_img)
            )
        
        tool_reward = 0.0
        if len(tool_result_images) > 0:
            tool_images_scores = [
                self.compute_image_score(
                    text_content=text_content,
                    bg_img_b64_str=self.get_image_b64_str(bg_image),
                    rendered_img_b64_str=self.get_image_b64_str(tool_image)
                ) for tool_image in tool_result_images
            ]
            mean_tool_score = sum(tool_images_scores) / len(tool_images_scores)
            if (answer_score - 0.1) > mean_tool_score:
                tool_reward = 1.0
            # if all(answer_score - 0.1 > x for x in tool_images_scores):
            #     tool_reward = 1.0
            # # elif len(tool_images_scores) > 1 and abs(answer_score - tool_images_scores[-1]) < 0.05 and all( answer_score - 0.1 > x for x in tool_images_scores[:-1]):
            # elif len(tool_images_scores) > 1 and abs(answer_score - tool_images_scores[-1]) < 0.05 and (answer_score - 0.1) > mean_tool_score:
            #     tool_reward = 1.0
            else:
                tool_reward = 0.0
        
        return {
            "index": param["index"],
            "score": format_reward + answer_score + tool_reward,
            "reward_parts": {
                "format_reward": format_reward,
                "answer_score": answer_score,
                "tool_reward": tool_reward,
                "tool_images_scores": tool_images_scores if len(tool_result_images) > 0 else [],
            }
        }
