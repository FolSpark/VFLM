# Discliamer: the model used in the script is only for academic purpose.
set -x

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export PYTHONUNBUFFERED=1

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>
wandb online
# export CUDA_VISIBLE_DEVICES=4,5,6,7

svg_train_path=datasets/svg-data/data/train_rl_1.parquet
svg_test_path=datasets/svg-data/data/val_rl_1.parquet

train_files="['$svg_train_path']"
test_files="['$svg_test_path']"

MODEL_PATH=models/qwen2.5_vl_7b_svg-v3_2025-05-03
RM_MODEL_PATH=models/rm-qwen2.5_vl_3b_2025-06-04/checkpoint-2100

PROJECT_NAME=design_rl
RUN_NAME="rl_grpo_svg_7b_from_sft-2025-06-11"

mkdir -p ./work_dirs/${RUN_NAME}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.image_key=images \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation=right \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    +actor_rollout_ref.processor_kwargs.max_pixels=401408 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +reward_model.task=LayoutSVG \
    reward_model.strategy=fsdp \
    reward_model.enable=True \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=False \
    +reward_model.processor_kwargs.max_pixels=401408 \
    reward_model.max_length=4096 \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=False \
    reward_model.micro_batch_size_per_gpu=32 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=work_dirs/${RUN_NAME} \
    trainer.rollout_data_dir=work_dirs/${RUN_NAME}/rollout \
    trainer.validation_data_dir=work_dirs/${RUN_NAME}/val \
    trainer.val_before_train=False \
    trainer.log_val_generations=1 \
    +val_gen_before_train=True \
    +trainer.rollout_freq=50 \
    +trainer.val_gen_before_train=True \
    +trainer.val_gen_freq=10 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 $@ 2>&1 | tee "work_dirs/${RUN_NAME}/verl-training.log"

