set -x

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=f4eeb2f2bcbec56195df62e39b6fee6e8f39e108
export PYTHONPATH=/mnt/jfs/zhangshengzhuo/rl/verl:$PYTHONPATH


# DATA_DIR=$HOME/data/gsm8k
# DATA_DIR=/mnt/jfs/zhangshengzhuo/rl/dataset/kk/base
# MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-7B"

# change max_position_embeddings/sliding_window from 32768 to 131072
# tokenizer.json add 4 token: <think></think> <answer></answer>
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-0.5B"
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-0.5B"
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-7B"
# MODEL_PATH=/mnt/yscfs/zhangshengzhuo/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

DATA_DIR=/mnt/jfs/zhangshengzhuo/rl/dataset/kk/instruct
MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-7B-Instruct-1M"

export N_NODES=2
export N_GPUS=8
export HYDRA_FULL_ERROR=1
export ROLLOUT_TP_SIZE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen-7B' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.default_local_dir=logic_rl \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log

    # trainer.logger=['console','wandb'] \
