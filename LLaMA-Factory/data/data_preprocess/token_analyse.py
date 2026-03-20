import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-tokenizer")

# 分析token的长度
# with open("./short_cards_with_svg_2k_700_1.jsonl", 'r') as f:
#     data1 = [json.loads(line) for line in f.readlines()]

# with open("./short_cards_with_svg_2k_700_2.jsonl", 'r') as f:
#     data2 = [json.loads(line) for line in f.readlines()]

# with open("./short_cards_with_svg_2k_700_3.jsonl", 'r') as f:
#     data3 = [json.loads(line) for line in f.readlines()]

# data = data1 + data2 + data3

data = json.load(open("data/datasets/svg-data/data/rethink_multi_dataset_v3_filter.json", 'r'))

prompt_lens = []
response_lens = []
total_lens = []
total_lens_3k = []

filter_data = []

for item in tqdm(data):
    # prompt = item['markdown_text']
    # response = item['svg_text']
    # prompt_tokens = tokenizer(prompt)['input_ids']
    # response_tokens = tokenizer(response)['input_ids']
    # prompt_lens.append(len(prompt_tokens))
    # response_lens.append(len(response_tokens))
    # total_lens.append(len(prompt_tokens) + len(response_tokens))
    
    message = item['messages']
    prompt = tokenizer.apply_chat_template(message, tokenize = True, add_generation_prompt = False)
    
    total_lens.append(len(prompt))
    if len(prompt) > 32 * 1024 - 1024 * len(item['images']):
        continue
    # item['images'] = ["data/datasets/penpot-data/" + image for image in item['images']]
    filter_data.append(item)
    
    # import ipdb; ipdb.set_trace()
    total_lens_3k.append(len(prompt))

total_lens.sort(reverse=True)
total_lens_3k.sort(reverse=True)

# 绘制直方图
# plt.hist(total_lens, edgecolor='black')
# plt.title('total token')
# plt.xlabel('token len')
# plt.ylabel('frequency')
# plt.savefig("hist.png")

import ipdb; ipdb.set_trace()

json.dump(filter_data, open("data/datasets/svg-data/data/rethink_multi_dataset_v3_filter_len32k.json", 'w'), indent=4, ensure_ascii=False)