set -e
set -x

# export NCCL_P2P_DISABLE="1"
# export IBN_P2P_DISABLE="1"
export CUDA_VISIBLE_DEVICES=0,1

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>
export WANDB_PROJECT=design_sft

export SWANLAB_API_KEY=<your_api_key>
# export SWANLAB_MODE=local       # disabled, cloud, local, offline 就用 cloud 或者 local 就行
SWANLAB_MODE=local


# 802816 = 28*28*1024 相当于896*896大小图片
# export IMAGE_PLACEHOLDER="<|image|>"

wandb offline


GPU_NUM=2
TOTAL_BATCH_SIZE=32
PER_GPU_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$(( TOTAL_BATCH_SIZE / ( GPU_NUM * PER_GPU_BATCH_SIZE ) ))

PROJECT_NAME=design_sft
RUN_NAME="llava1.5_7b_OpenCOLE_TextLayout_lora_2026-01-28"


mkdir -p work_dirs/${RUN_NAME}

accelerate launch --num_machines 1 --num_processes ${GPU_NUM} --machine_rank 0 --main_process_port 29502 --mixed_precision bf16 \
  src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path models/llava-hf/llava-1.5-7b-hf \
    --preprocessing_num_workers 64 \
    --finetuning_type lora \
    --lora_rank 128 \
    --lora_target all \
    --template llava \
    --bf16 True \
    --flash_attn fa2 \
    --enable_liger_kernel True \
    --packing False \
    --dataset_dir data \
    --dataset OpenCOLE_TextLayout_train \
    --eval_dataset OpenCOLE_TextLayout_test \
    --tokenized_path data/datasets/svg-data/tokenized_OpenCOLE_TextLayout \
    --media_dir data/datasets/svg-data/data \
    --overwrite_cache \
    --cutoff_len 16384 \
    --learning_rate 2e-04 \
    --num_train_epochs 6.0 \
    --warmup_ratio 0.03 \
    --max_samples 1000000 \
    --per_device_train_batch_size ${PER_GPU_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --eval_strategy epoch \
    --eval_steps 400 \
    --eval_on_start \
    --per_device_eval_batch_size 4 \
    --save_strategy epoch \
    --save_steps 300 \
    --report_to wandb \
    --use_swanlab True \
    --swanlab_mode ${SWANLAB_MODE} \
    --swanlab_project ${PROJECT_NAME} \
    --swanlab_run_name ${RUN_NAME} \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 21600000000 \
    --optim adamw_torch \
    --run_name ${RUN_NAME} \
    --output_dir work_dirs/${RUN_NAME}  2>&1 | tee "work_dirs/${RUN_NAME}/training.log"

