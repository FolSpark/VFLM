#!/bin/bash
set -e
set -x

# hf_model_path=models/Qwen2.5-VL-3B-Instruct      # 原始模型路径，需要从这里读config
# local_dir=work_dirs/debug-critic_8/global_step_100/actor

# 看自己的模型是否是tie word embedding，是的话加上（store true），否则保存后会变大  Qwen2.5-3B及以下是tie word embedding，更大的不是
# PYTHONPATH=. python scripts/model_merger.py merge --backend fsdp --tie-word-embedding --local_dir $local_dir  --hf_model_path $hf_model_path --target_dir ${local_dir}_huggingface


# PYTHONPATH=. python scripts/model_merger.py test --backend fsdp --tie-word-embedding --local_dir $local_dir --hf_model_path $hf_model_path  --test_hf_dir ${local_dir}_huggingface

local_dir=work_dirs/tool_rl_grpo_svg_v3_3_7b_from_sft-2025-07-17/global_step_90

PYTHONPATH=. python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir work_dirs/tool_rl_grpo_svg_v3_3_7b_from_sft-2025-07-17/global_step_90/actor \
    --target_dir work_dirs/tool_rl_grpo_svg_v3_3_7b_from_sft-2025-07-17/global_step_90/actor_huggingface 
