set -x

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=f4eeb2f2bcbec56195df62e39b6fee6e8f39e108


DATA_PATH=$HOME/data/gsm8k

# change max_position_embeddings/sliding_window from 32768 to 131072
# tokenizer.json add 4 token: <think></think> <answer></answer>
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-0.5B"
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-0.5B"
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-7B"
MODEL_PATH=/mnt/yscfs/zhangshengzhuo/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

BASE_NAME="Qwen25-15B"
LR=1e-6
train_batch_size=1024
ppo_mini_batch_size=256
val_batch_size=1024
max_prompt_length=512
max_response_length=1024

ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length)* 2 ))

echo ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu

rollout_n=5
kl_coef=0.001
EXP_NAME="$BASE_NAME-LR$LR-ROLLOUT$rollout_n"
echo $EXP_NAME

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$BASE_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@