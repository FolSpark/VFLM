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

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
import json
from PIL import Image
import base64


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image;base64,{encoded_string}"


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

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**: 
<tool_call> 
{"name": "svg_to_image_tool", "arguments": {"svg_code": "```svg\n...\n```"}}
</tool_call>"""

system_prompt = (
    "You are an experienced visual layout designer and SVG engineer, skilled at elegantly typesetting specified text on background images provided by users.\n"
    "You know how to apply unique aesthetic principles to design professional and appealing layouts, using SVG code to create beautiful layouts. Please design a final layout plan based on the background image and text content provided by the user.\n"
    "In the SVG code, use the image tag to reference the background image: href=\"background-image.png\", and other elements only need to design content related to the text.\n"
    "Please design an SVG code layout plan based on the following background image and text content provided by the user.\n"
    "You should first view the background image, think about how to typeset the text on the background image, design a version of SVG code, correctly reference the background image in the SVG code, then call the svg_to_image tool, and you will get the picture of your SVG. Then, based on the picture, judge whether the typesetting of your picture meets the expectations, whether the background image is correctly referenced, and whether the text is beautiful. If the typesetting effect is not good enough, modify the SVG code, and repeatedly reflect after tool calls until the typesetting effect is better. Finally, output the final SVG code.\n"
    "Format: <think>...</think> <tool_call>...</tool_call>(if tools needed) <answer>...</answer>"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='datasets/svg-data/data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='datasets/svg-data/data/high_quality_data_all_filter_16k.json')
    parser.add_argument('--train_size', type=float, default=0.999)
    parser.add_argument('--test_size', type=float, default=0.001)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSON dataset
    def gen_from_json(path, k=0):
        with open(path) as f:
            data = json.load(f)
            for item in data:
                yield item
    
    raw_dataset = Dataset.from_generator(gen_from_json, gen_kwargs={'path': args.data_path})
    print(f"len raw_dataset: {len(raw_dataset)}")
    
    # Shuffle the dataset
    raw_dataset = raw_dataset.shuffle(seed=42)

    # TRAIN_SIZE = int(len(raw_dataset) * args.train_size)
    # TEST_SIZE = len(raw_dataset) - TRAIN_SIZE
    TRAIN_SIZE = 79872  # 78848
    TEST_SIZE = len(raw_dataset) - TRAIN_SIZE
    print(f"TRAIN_SIZE: {TRAIN_SIZE}, TEST_SIZE: {TEST_SIZE}")

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            images = example.pop('images')
            # images = [convert_image_to_base64(os.path.join(args.local_dir, image_path)) for image_path in images]
            messages = example.pop('messages')

            data = {
                "data_source": "layout_svg_w_tool",
                "prompt": [
                    {
                        "role": "system",
                        "content": instruction_prompt_system + '\n\n' + system_prompt,
                    },
                    {
                        "role": "user",
                        "content": messages[1]['content'],
                    }
                ],
                "images": [images[0]],
                "env_name": "svg_to_image_tool",
                "ability": "design",
                "reward_model": {
                    "style": "model",
                    "ground_truth": open(os.path.join("datasets/svg-data/data/process_data_0424", os.path.basename(images[0])).replace('-bg.png', '.svg'), 'r').read(),
                },
                "relative_path": True,  # 不是相对路径就不要写
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "text_content": messages[1]['content'].split('\ntexts: \n')[-1],
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    import ipdb; ipdb.set_trace()

    train_dataset.to_parquet(os.path.join(local_dir, 'train_rl_w_tool.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_rl_w_tool.parquet'))
