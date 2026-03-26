import os
import re
import json
import random
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests

from svg_preprocess.utils import export_svg_to_img


parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://127.0.0.1:30091/v1', help='API URL')
parser.add_argument('--save_path', type=str, default="outputs/VFLM", help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

clients = [
    OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
]

# import ipdb; ipdb.set_trace()
eval_model_name = clients[0].models.list().model_dump()['data'][0]['id']

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function ",
    "function": {
        "name": "svg_to_image_tool",
        "description": Convert SVG code to an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "svg_code": {
                    "type": "string",
                    "description": "The SVG code to convert to an image."
                }
            },
            "required":[
                "svg_code"
            ]
        }
    }
}
</tools>

# How to call this tool
Wrap the SVG code with specific markers (``` and ```) within <tool_call></tool_call> XML tags.

**Example**: 
<tool_call>
TOOL: svg_to_image_tool
PARAMS: 
svg_code:
```svg
...
```
</tool_call>"""

system_prompt = (
    "You are an experienced visual layout designer and SVG engineer, skilled at elegantly typesetting specified text on background images provided by users.\n"
    "You know how to apply unique aesthetic principles to design professional and appealing layouts, using SVG code to create beautiful layouts. Please design a final layout plan based on the background image and text content provided by the user.\n"
    "In the SVG code, use the image tag to reference the background image: href=\"background-image.png\", and other elements only need to design content related to the text.\n"
    "Please design an SVG code layout plan based on the following background image and text content provided by the user.\n"
    "You should first view the background image, think about how to typeset the text on the background image, design a version of SVG code, correctly reference the background image in the SVG code, then call the svg_to_image tool, and you will get the picture of your SVG. Then, based on the picture, judge whether the typesetting of your picture meets the expectations, whether the background image is correctly referenced, and whether the text is beautiful. If the typesetting effect is not good enough, modify the SVG code, and repeatedly reflect after tool calls until the typesetting effect is better. Finally, output the final SVG code.\n"
    "Format: <think>...</think>\n<tool_call>...</tool_call>(if tools needed) <answer>...</answer>"
)


start_token = "<tool_call>"
end_token = "</tool_call>"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def extract_answer(text):
    # 使用非贪婪模式匹配<answer>和</answer>之间的所有内容
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL允许匹配换行符
    return matches


def extract_tool_and_svg(text: str):
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


def process(text, bg_image_path, save_name):
    if os.path.exists(f"{save_path}/{save_name}-answer.png"):
        print(f"File {save_name}-answer.png already exists, skipping.")
        return
    # img, test_path = img_arg
    client = random.choice(clients)
    image = Image.open(bg_image_path)
    image_width, image_height = image.size


    base64_image = encode_image_to_base64(bg_image_path)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system + '\n\n' + system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"background-image.png: "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "max_pixels": 1024 * 28 * 28},
                {"type": "text", "text": f"image size: {image_width}*{image_height}\ntexts: \n{text}"},
            ],
        }
    ]

    response_message = ""

    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 4:
                break

            params = {
                "model": eval_model_name,
                "messages": messages,
                "temperature": 0.7,
                # "max_tokens": 32768,
                # "stop": ["<|im_end|>", "</tool_call>"],
                # "stop_token_ids": [151658, 151645],
                # "include_stop_str_in_output": True
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content

            # import ipdb; ipdb.set_trace()
            # response_message = response[0].outputs[0].text
            # print(f"Turn {turn_idx}: {response_message}")
            open(f"{save_path}/{save_name}-{turn_idx}.txt", "w").write(response_message)
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                # import ipdb; ipdb.set_trace()
                # action_list = eval(action_list)

                # svg_code = action_list['arguments']['svg_code']
                # if svg_code.startswith("```svg\n"):
                #     svg_code = svg_code[7:]
                # if svg_code.endswith("```"):
                #     svg_code = svg_code[:-3]
                # print(f"SVG Code: {svg_code}")
                tool_name, svg_code = extract_tool_and_svg(action_list)
                if svg_code.startswith("```svg\n") and svg_code.endswith("```"):
                    svg_code = svg_code[7:-3].strip()
                assert tool_name == "svg_to_image_tool", f"Unexpected tool name: {tool_name}"
                open(f"{save_path}/{save_name}-{turn_idx}.svg", "w").write(svg_code)    
                
                rendered_image = export_svg_to_img(svg_code, image)
                rendered_image.save(f"{save_path}/{save_name}-{turn_idx}.png")
                rendered_image_base64 = encode_pil_image_to_base64(rendered_image)
                rendered_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rendered_image_base64}"}, "max_pixels": MAX_PIXELS}

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                content_f.append(rendered_image_content)
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                messages.extend(_message)
                turn_idx += 1
                if turn_idx > 4:
                    print(f"{save_name}: Too many turns, breaking after {turn_idx} turns. retrying...")
                    raise ValueError("Too many turns, something went wrong.")


            try_count += 1

        # if '</answer>' in response_message and '<answer>' in response_message:
        #     output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
        # else:
        output_text = response_message
        
        output_svg = extract_answer(output_text)[0]
        if output_svg.startswith("```svg\n"):
            output_svg = output_svg[7:]
        if output_svg.endswith("```"):
            output_svg = output_svg[:-3]
        # import ipdb; ipdb.set_trace()
        open(f"{save_path}/{save_name}-answer.svg", "w").write(output_svg)
        output_image = export_svg_to_img(output_svg, image)
        output_image.save(f"{save_path}/{save_name}-answer.png")

        save_info = {}
        save_info['pred_ans'] = output_text
        # save_info['pred_output'] = print_messages
        save_info['status'] = status
        return save_info
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'
        return None


def call_api(data):
    id = os.path.basename(data['svg_file']).replace('.svg', '')
    text = data['text_content']
    # bg_image_path = os.path.join("datasets/svg-data/data/eval_dataset_1000", data['bg_image_file'])
    bg_image_path = os.path.join("datasets/svg-data/data", data['bg_image_file'])
    save_name = id
    process(text, bg_image_path, save_name)
    
    

if __name__ == "__main__":

    all_data = json.load(open("datasets/svg-data/data/eval_dataset_1000.json", "r"))
    
    with multiprocessing.Pool(args.num_workers) as p:
            list(tqdm(p.imap(call_api, all_data), total=len(all_data)))
