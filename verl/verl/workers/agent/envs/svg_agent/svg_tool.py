import re
import json
from PIL import Image
from typing import Optional, List, Dict, Any

from verl.utils.constants import IMAGE_PLACEHOLDER
from verl.utils.svg_utils import export_svg_to_img
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


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer
        
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
        if not tool_call_match:
            raise ValueError("No tool call found in the action string.")
        
        tool_call_json = tool_call_match.group(1).strip()
        try:
            tool_call = json.loads(tool_call_json)
            return tool_call
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {e}")

    def execute(self, action_string: str, **kwargs) -> tuple:
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
            answer = self.extract_answer(action_string)
            if answer:
                # print(f'[DEBUG] SUCCESS ANSWER {action_string=}')
                return "", 0.0, True, {}
            tool_call = self.extract_action(action_string)
            tool_name = tool_call["name"]
            svg_code = tool_call["arguments"]["svg_code"]
            
            if tool_name == "svg_to_image_tool":
                # Zoom in by cropping the image
                img = self.multi_modal_data['image'][0]
                if svg_code.startswith("```svg\n") and svg_code.endswith("```"):
                    svg_code = svg_code[7:-3].strip()
                rendered_img = export_svg_to_img(svg_code, img)
                
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            
            # Prepare the observation
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n<tool_response>" + IMAGE_PLACEHOLDER + "</tool_response><|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [rendered_img]},
            }
            reward = 0.5  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            # print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info
            
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)}')
            print(f'[DEBUG] {action_string=}')
            obs = {
                "prompt": f"<|im_start|>user\n<tool_response>Error: {str(e)}</tool_response><|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": []},
            }
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

