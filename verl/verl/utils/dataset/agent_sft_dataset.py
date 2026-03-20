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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd

import torch
import json
from omegaconf import OmegaConf
from jinja2 import Template
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def load_jsonl(filename):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


class AgentSFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        
        system_key = config.get('system_key', 'system')
        system_dict_keys = config.get('system_dict_keys', None)
        prompt_key = config.get('prompt_key', 'prompt')
        prompt_dict_keys = config.get('prompt_dict_keys', None)
        response_key = config.get('response_key', 'response')
        response_dict_keys = config.get('response_dict_keys', None)
        max_length = config.get('max_length', 1024)
        truncation = config.get('truncation', 'error')
        
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_length = max_length

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):

        self.raw_data = []
        for file_path in self.parquet_files:
            # read parquet files and cache
            self.raw_data += load_jsonl(file_path)
        if not self.raw_data:
            raise Exception("JSONL Data Loading Error")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        data_dict = self.raw_data[item]
        messages = data_dict['message']
        for _t in messages:
            assert _t['content'].strip(), "Data contains empty turns!"
        
        # iteratively tokenize and mask
        start_token = "<|im_start|>"
        end_token = "<|im_end|>"

        all_input_ids, all_attention_masks, all_loss_masks = [], [], []
        for message in messages:
            role = message['role']
            content = message['content']
            mask = message['mask']
            
            prefix_str = f"{start_token}{role}\n"
            prefix_tokenized = tokenizer(prefix_str, return_tensors='pt', add_special_tokens=False)

            turn_str = f"{content}"
            tokenized = tokenizer(turn_str, return_tensors='pt', add_special_tokens=False)

            suffix_str = f"{end_token}\n"
            suffix_tokenized = tokenizer(suffix_str, return_tensors='pt', add_special_tokens=False)

            _input_ids = torch.cat((prefix_tokenized['input_ids'][0], tokenized['input_ids'][0], suffix_tokenized['input_ids'][0]))
            _attention_mask = torch.cat((prefix_tokenized['attention_mask'][0], tokenized['attention_mask'][0], suffix_tokenized['attention_mask'][0]))

            if mask == 0:
                # _loss_mask = torch.zeros(size=_input_ids.size(), dtype=_input_ids.dtype)
                _loss_mask = torch.cat((
                    torch.zeros(size=prefix_tokenized['input_ids'][0].size(), dtype=_input_ids.dtype),
                    torch.zeros(size=tokenized['input_ids'][0].size(), dtype=_input_ids.dtype),
                    torch.ones(size=suffix_tokenized['input_ids'][0].size(), dtype=_input_ids.dtype)
                ))
            else:
                _loss_mask = torch.cat((
                    torch.zeros(size=prefix_tokenized['input_ids'][0].size(), dtype=_input_ids.dtype),
                    torch.ones(size=tokenized['input_ids'][0].size(), dtype=_input_ids.dtype),
                    torch.ones(size=suffix_tokenized['input_ids'][0].size(), dtype=_input_ids.dtype)
                ))
            _loss_mask[-1] = 0

            all_input_ids.append(_input_ids)
            all_attention_masks.append(_attention_mask)
            all_loss_masks.append(_loss_mask)

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        loss_mask = torch.cat(all_loss_masks, dim=0)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            # loss mask pad same as attention mask
            loss_mask = torch.cat((loss_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }


if __name__ == "__main__":
    dataset = AgentSFTDataset(
        parquet_files='/mnt/yscfs/xubenfeng/ys_agent_dev/thinkingagent/data/version/thinkingagent_v01_0221.jsonl',
        tokenizer='/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-72B/',
        config=OmegaConf.create({
            'max_length': 128*1024,
            'truncation': 'error',
        })
    )

    results = dataset.__getitem__(0)
    tokenizer = dataset.tokenizer

    breakpoint()
    print(tokenizer.decode(results['input_ids'][:100], skip_special_tokens=False))
    
    masked = torch.where(results['loss_mask'] == 1, results['input_ids'], 151643)
    print(tokenizer.decode(masked[:100], skip_special_tokens=False))