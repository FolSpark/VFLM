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
"""
Preprocess the Congliu/Chinese-DeepSeek-R1-Distill-data-110k dataset to parquet format
"""

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='datasets/md2svg')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='datasets/md2svg/data_2100.json')
    parser.add_argument('--train_size', type=float, default=0.95)
    parser.add_argument('--test_size', type=float, default=0.05)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'md2svg'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSON dataset
    def gen_from_json(path, k=0):
        with open(path) as f:
            data = json.load(f)
            for item in data:
                yield item
    
    raw_dataset = Dataset.from_generator(gen_from_json, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    TRAIN_SIZE = int(len(raw_dataset) * args.train_size)
    TEST_SIZE = len(raw_dataset) - TRAIN_SIZE

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    

    def make_map_fn(split):
        def process_fn(example, idx):
            # question = make_prefix(example, template_type=args.template_type)
            system_prompt = "You are a helpful assistant."
            prompt = f"""# 角色设定
你是一个专业的信息架构师兼SVG工程师，擅长将复杂文本转化为信息可视化图形。具备平面设计原则、SVG语法规范、视觉认知心理学三重领域知识。

# 输出要求
- 只输出SVG代码，不需要输出其他内容

# 开始工作
<输入文本>：

{example['markdown_text']}

<输出SVG>："""
            response = example['svg_text']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "code",
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example['id'],
                    'system': system_prompt,
                    "prompt": prompt,
                    'response': response,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)