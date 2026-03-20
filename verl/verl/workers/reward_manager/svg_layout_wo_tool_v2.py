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


class ToolCallStep0SvgLayoutRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        self.servers = "http://10.0.2.95:8999/compute_score"
        self.api_num = 96

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

            params.append({
                "index": i,
                "response_str": response_str,
                "text_content": extra_info["tools_kwargs"]["text_content"],
                "bg_image": bg_image
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

    def extract_tool_call(self, action_string: str) -> Optional[str]:
        tool_call = re.search(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call.group(1) if tool_call else None

    def extract_tool_and_svg(self, text: str):
        # 提取工具名
        tool_start = text.find("TOOL:")
        if tool_start == -1:
            return None, None
        
        tool_name_start = tool_start + len("TOOL:")
        tool_name_end = text.find("\n", tool_name_start)
        if tool_name_end == -1:
            tool_name_end = len(text)
        
        tool_name = text[tool_name_start:tool_name_end].strip()
        
        params_start = text.find("PARAMS:", tool_name_end)
        if params_start == -1:
            return tool_name, None
        params_start += len("PARAMS:")
        params_string = text[params_start:].strip()
        if params_string.startswith("svg_code:"):
            params_string = params_string[len("svg_code:"):].strip()
        return tool_name, params_string

    def extract_last_svg_code(self, text):
        pattern = r'```svg\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1] if matches else None
    
    def format_reward(self, response_str: str) -> float:
        pattern = re.compile(r'<think>.*?</think>\n<tool_call>.*?</tool_call>', re.DOTALL)
        match_result = re.fullmatch(pattern, response_str)
        return 0.1 if match_result else -0.1

    def compute_image_score(self, svg_code, text_content, bg_img_b64_str, rendered_img_b64_str) -> float:
        for cnt in range(3):  # Retry up to 3 times
            try:
                # url = self.api_pool.get()
                url = self.servers
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "prompt": text_content,
                    "svg_code": svg_code,
                    "image1": bg_img_b64_str,
                    "image2": rendered_img_b64_str,
                }
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=600)
                response = response.json()
                # self.api_pool.put(url)  # Put the URL back into the pool for reuse
                # return response
                image_score = response["score"]
                ocr_accuracy = response["ocr_accuracy"]
                svg_code_accuracy = response["svg_code_accuracy"]

                # return image_score + ocr_accuracy + svg_code_accuracy
                return {"score": image_score, "ocr_accuracy": ocr_accuracy, "svg_code_accuracy": svg_code_accuracy}
                # return response["score"]
            except Exception as e:
                print(f"Error count {cnt} during API call: {e}")
                # self.api_pool.put(url)  # Put the URL back into the pool for reuse
                continue
        print("Failed to compute image score after multiple attempts")
        raise RuntimeError("Failed to compute image score after multiple attempts")

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
        
        try:
            format_reward = self.format_reward(response_str)
            tool_call_str = self.extract_tool_call(response_str)

            if tool_call_str:
                tool_name, svg_code = self.extract_tool_and_svg(tool_call_str)
                
                if tool_name != "svg_to_image_tool":
                    print(f'[DEBUG] Unknown tool name\n{response_str=}')
                    format_reward = -0.1

                if not svg_code:
                    format_reward = -0.1
                    svg_code = self.extract_last_svg_code(response_str)
            else:
                print(f'[DEBUG] No tool call found in response\n{response_str=}')
                format_reward = -0.1
                svg_code = self.extract_last_svg_code(response_str)


            if svg_code:
                if svg_code.startswith("```svg\n"):
                    svg_code = svg_code[7:]
                if svg_code.endswith("```"):
                    svg_code = svg_code[:-3]
                try:
                    rendered_img = export_svg_to_img(svg_code, bg_image)
                except Exception as e:
                    format_reward = -0.1
                    print(f"Error during SVG rendering: {e}")
                    rendered_img = Image.new("RGB", bg_image.size, (0, 0, 0))
            else:
                rendered_img = Image.new("RGB", bg_image.size, (0, 0, 0))
        
        except Exception as e:
            print(f"Error computing score for index {param['index']}: {e}")
            format_reward = -0.1
            svg_code = ""
            rendered_img = Image.new("RGB", bg_image.size, (0, 0, 0))
            
        answer_score = self.compute_image_score(
            svg_code=svg_code if svg_code else "",
            text_content=text_content,
            bg_img_b64_str=self.get_image_b64_str(bg_image),
            rendered_img_b64_str=self.get_image_b64_str(rendered_img)
        )
        
        image_score = answer_score["score"]
        ocr_accuracy = answer_score["ocr_accuracy"]
        svg_code_accuracy = answer_score["svg_code_accuracy"]
        
        return {
            "index": param["index"],
            "score": format_reward + image_score + 0.25 * ocr_accuracy + 0.25 * svg_code_accuracy,
            # "score": format_reward + image_score,
            "reward_parts": {
                "format_reward": format_reward,
                # "answer_score": answer_score,
                "image_score": image_score,
                "ocr_accuracy": ocr_accuracy,
                "svg_code_accuracy": svg_code_accuracy,
            }
        }
        # return {
        #     "index": param["index"],
        #     "score": format_reward + image_score,
        #     "reward_parts": {
        #         "format_reward": format_reward,
        #         # "answer_score": answer_score,
        #         "image_score": image_score,
        #         "ocr_accuracy": ocr_accuracy,
        #         "svg_code_accuracy": svg_code_accuracy,
        #     }
        # }
