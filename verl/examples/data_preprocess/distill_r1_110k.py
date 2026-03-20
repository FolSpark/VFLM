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
    # parser.add_argument('--local_dir', default='/mnt/jfs/zhangshengzhuo/rl/verl/examples/data/Chinese-DeepSeek-R1-Distill-data-110k')
    parser.add_argument('--local_dir', default='/mnt/yscfs/zhangshengzhuo/rl/dataset/Chinese-DeepSeek-R1-Distill-data-110k')
    parser.add_argument('--hdfs_dir', default=None)
    # parser.add_argument('--data_path', default='/mnt/jfs/zhangshengzhuo/rl/data/Congliu/Chinese-DeepSeek-R1-Distill-data-110k/distill_r1_110k.jsonl')
    parser.add_argument('--data_path', default='/mnt/yscfs/zhangshengzhuo/rl/data/distill_r1_110k.jsonl')
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--test_size', type=float, default=0.9)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'Chinese-DeepSeek-R1-Distill-data-110k'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path, k=0):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
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
            question = example['input']
            think = example['reasoning_content']
            answer = example['content']
            solution = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'system': system_prompt,
                    'answer': solution,
                    "question": question,
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