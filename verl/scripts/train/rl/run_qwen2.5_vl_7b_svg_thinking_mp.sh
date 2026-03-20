set -x

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export PYTHONUNBUFFERED=1


export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>
wandb online
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export IMAGE_PLACEHOLDER="<|image|>"

train_files=datasets/svg-data/data/train_rl_wo_tool.parquet
test_files=datasets/svg-data/data/test_rl_wo_tool.parquet

# MODEL_PATH=LF_models/qwen2.5_vl_7b_svg-think_66k_0713_
MODEL_PATH=LF_models/qwen2.5_vl_7b_svg-think_66k_0713_/checkpoint-256

PROJECT_NAME=design_rl
RUN_NAME="tool_rl_grpo_svg_thinking_7b_from_sft_epoch1-2025-08-07"


max_prompt_length=4096
max_response_length=12288
ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length)*2 ))
train_batch_size=64
ppo_mini_batch_size=64
ulysses_sequence_parallel_size=1


DIR=`pwd`

mkdir -p ./work_dirs/${RUN_NAME}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DIR}/${train_files} \
    data.val_files=${DIR}/${test_files} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.image_key=images \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=right \
    actor_rollout_ref.model.path=${DIR}/${MODEL_PATH} \
    +actor_rollout_ref.processor_kwargs.max_pixels=802816 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=svg_layout_think \
    reward_model.launch_reward_fn_async=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=${DIR}/work_dirs/${RUN_NAME} \
    trainer.rollout_data_dir=${DIR}/work_dirs/${RUN_NAME}/rollout \
    trainer.validation_data_dir=${DIR}/work_dirs/${RUN_NAME}/val \
    trainer.val_before_train=False \
    trainer.log_val_generations=1 \
    +trainer.rollout_freq=50 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 $@ 2>&1 | tee "${DIR}/work_dirs/${RUN_NAME}/verl-training.log"

