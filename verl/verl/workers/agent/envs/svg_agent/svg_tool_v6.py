import re
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any

from verl.utils.constants import IMAGE_PLACEHOLDER
from verl.utils.svg_utils import export_svg_to_img, export_svg_with_bg
from verl.workers.agent.tool_envs import ToolBase


class SvgToolEnv(ToolBase):
    name = "svg_to_image_tool"
    user_prompt = ""

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed
        self.text_content = ""
        self.servers = "http://10.0.2.95:8999/compute_score"
    
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
        print("Failed to compute image score after multiple attempts")
        raise RuntimeError("Failed to compute image score after multiple attempts")

    def extract_answer(self, action_string: str) -> tuple:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        if answer:
            return (True, answer[-1].strip())  # Return the last answer found
        # 应对<answer></think>等情况，查看是否有<answer>开头标签，不管结尾
        answer = re.findall(r'<answer>(.*?)<', action_string, re.DOTALL)
        if answer:
            return (False, answer[-1].strip())
        return (False, None)

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        # 这也类似answer的提取方式，可能会有其他情况
        if tool_call_match:
            return tool_call_match.group(1).strip()
        # 应对<tool_call></think>等情况，查看是否有<tool_call>开头标签，不管结尾
        tool_call_match = re.search(r'<tool_call>(.*?)<', action_string, re.DOTALL)
        if tool_call_match:
            return tool_call_match.group(1).strip()
        return ""   # 不是None，否则后面find会报错

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

    def execute(self, action_string: str, tool_infos, step: int, is_last_step: bool, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        try:
            format_reward = 0.0
            is_valid, answer = self.extract_answer(action_string)
            if answer:
                format_reward = 0.0 if is_valid else -0.2
                if answer.startswith("```svg\n"):
                    answer = answer[7:]
                if answer.endswith("```"):
                    answer = answer[:-3]
                try:
                    rendered_img = export_svg_to_img(answer, self.multi_modal_data['image'][0])
                    # rendered_img = export_svg_with_bg(answer, self.multi_modal_data['image'][0])
                    if rendered_img is None:
                        rendered_img = Image.new('RGB', (224, 224), color='black')
                except Exception as e:
                    print(f'[DEBUG] Error in exporting SVG: {e}')
                    rendered_img = Image.new('RGB', (224, 224), color='black')
                image_score = self.compute_image_score(
                    self.text_content,
                    self.get_image_b64_str(self.multi_modal_data['image'][0]),
                    self.get_image_b64_str(rendered_img)
                )
                if step == 0:
                    reward = 0.0
                else:
                    if step > 0:
                        # reward = image_score - max(tool_infos["tool_result_images_scores"])
                        if image_score >= max(tool_infos["tool_result_images_scores"]):
                            reward = 0.5
                        else:
                            reward = image_score - max(tool_infos["tool_result_images_scores"])
                    else:
                        reward = 0.0
                return "", reward, True, {"image_score": image_score, "origin_multi_modal_data": {"image": [rendered_img]}}

            tool_call = self.extract_action(action_string)
            tool_name, svg_code = self.extract_tool_and_svg(tool_call)

            if tool_name != "svg_to_image_tool":
                print(f'[DEBUG] Unknown tool name\n{action_string=}')
                format_reward = -0.2

            if svg_code is None:
                svg_code = self.extract_last_svg_code(action_string)

            # Zoom in by cropping the image
            if svg_code:
                if svg_code.startswith("```svg\n"):
                    svg_code = svg_code[7:]
                if svg_code.endswith("```"):
                    svg_code = svg_code[:-3]
                try:
                    rendered_img = export_svg_to_img(svg_code, self.multi_modal_data['image'][0])
                    # rendered_img = export_svg_with_bg(svg_code, self.multi_modal_data['image'][0])
                    if rendered_img is None:
                        rendered_img = Image.new('RGB', (224, 224), color='black')
                except Exception as e:
                    format_reward = -0.2
                    print(f'[DEBUG] Error in exporting SVG: {e}')
                    rendered_img = Image.new('RGB', (224, 224), color='black')
            else:
                format_reward = -0.2
                print(f'[DEBUG] No SVG code found in action string: {action_string}')
                rendered_img = Image.new('RGB', (224, 224), color='black')
            image_score = self.compute_image_score(
                self.text_content,
                self.get_image_b64_str(self.multi_modal_data['image'][0]),
                self.get_image_b64_str(rendered_img)
            )

            if step == 0:
                reward = 0.0
            else:
                prev_max = max(tool_infos["tool_result_images_scores"])
                # reward = image_score - prev_max
                if image_score >= prev_max - 0.05:
                    reward = image_score - prev_max
                else:
                    reward = image_score - prev_max - 0.05  # Slightly penalize if the score is significantly lower than the previous max
            reward += format_reward
            if is_last_step:
                reward -= 0.1  # Slightly penalize the last step to encourage earlier answer
                return "", reward, True, {"image_score": image_score, "origin_multi_modal_data": {"image": [rendered_img]}}
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n<tool_response>" + IMAGE_PLACEHOLDER + "</tool_response><|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [rendered_img]}
            }
            done = False
            info = {"status": "success", "tool_used": tool_name, "image_score": image_score, "origin_multi_modal_data": {"image": [rendered_img]}}
            # print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info

        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)}\n{action_string=}')
            # a black image as error response
            rendered_img = Image.new('RGB', (224, 224), color='black')
            image_score = self.compute_image_score(
                                self.text_content,
                                self.get_image_b64_str(self.multi_modal_data['image'][0]),
                                self.get_image_b64_str(rendered_img)
                            )
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            if step == 0:
                reward = image_score
            else:
                reward = image_score - max(tool_infos["tool_result_images_scores"])
            done = False
            info = {"error": str(e), "status": "failed", "image_score": image_score, "origin_multi_modal_data": {"image": [rendered_img]}}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        self.text_content = kwargs.get("text_content", "")
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

