export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export PYTHONUNBUFFERED=1


export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>
wandb online
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export IMAGE_PLACEHOLDER="<|image|>"

ray start --head --dashboard-host=0.0.0.0