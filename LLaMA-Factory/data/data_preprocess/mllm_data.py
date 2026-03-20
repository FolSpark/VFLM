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
使用于把mllm 训练的 json格式的数据集转换为parquet格式，并划分train/test集
"""

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the input JSON dataset')
    parser.add_argument('--output_name', required=True, help='Base name for the output files')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--train_size', type=int, help='优先指定train size，不然就指定train ratio')
    parser.add_argument('--train_ratio', type=float, default=0.99)

    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_path)

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
    
    # train size和train ratio二选一
    if args.train_size:
        TRAIN_SIZE = args.train_size
        TEST_SIZE = len(raw_dataset) - TRAIN_SIZE
    elif args.train_ratio:
        TRAIN_SIZE = int(len(raw_dataset) * args.train_ratio)
        TEST_SIZE = len(raw_dataset) - TRAIN_SIZE
    else:
        raise ValueError("Either train_size or train_ratio must be specified.")

    print(f"TRAIN_SIZE: {TRAIN_SIZE}, TEST_SIZE: {TEST_SIZE}")

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    

    def make_map_fn(split):
        def process_fn(example, idx):
            example.update({
                "extra_info": {
                    'split': split,
                    'index': idx,
                }})
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    train_dataset.to_parquet(os.path.join(args.output_dir, f'{args.output_name}_train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output_dir, f'{args.output_name}_test.parquet'))