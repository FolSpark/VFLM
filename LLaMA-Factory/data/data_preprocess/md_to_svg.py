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
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/datasets/md2svg')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='data/datasets/md2svg/data_mixed_all.json')
    parser.add_argument('--train_size', type=float, default=0.999)
    parser.add_argument('--test_size', type=float, default=0.001)
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
    print(f"Total dataset size: {len(raw_dataset)}")
    
    # Shuffle the dataset
    raw_dataset = raw_dataset.shuffle(seed=42)
    
    TRAIN_SIZE = int(len(raw_dataset) * args.train_size)
    TEST_SIZE = len(raw_dataset) - TRAIN_SIZE

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            # question = make_prefix(example, template_type=args.template_type)
            # system_prompt = "You are a helpful assistant."
            # prompt = example['instruction']
            # response = example['output']
            # data = {
            #     "data_source": data_source,
            #     "instruction": max_steps,
                
            # }
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
