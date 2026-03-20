set -e
set -x

# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"
# export CUDA_VISIBLE_DEVICES=4,5,6,7


RUN_NAME="rm-qwen2.5_vl_3b_2025-05-09"


export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>

# 802816 = 28*28*1024 相当于896*896大小图片

wandb online


mkdir -p work_dirs/${RUN_NAME}

accelerate launch src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage rm_image \
    --do_train True \
    --model_name_or_path models/Qwen2.5-VL-3B-Instruct \
    --preprocessing_num_workers 64 \
    --finetuning_type freeze \
    --freeze_trainable_layers 36 \
    --template qwen2_vl \
    --flash_attn fa2 \
    --enable_liger_kernel True \
    --dataset_dir data \
    --dataset layout_rm_train \
    --eval_dataset layout_rm_test \
    --media_dir data/datasets/svg-data/data \
    --tokenized_path data/datasets/svg-data/tokenized_rm_0424 \
    --cutoff_len 8192 \
    --image_max_pixels 401408 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_steps 50 \
    --save_safetensors False \
    --warmup_ratio 0.03 \
    --packing False \
    --report_to wandb \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 1800000000 \
    --optim adamw_torch \
    --eval_strategy steps \
    --eval_steps 390 \
    --eval_on_start \
    --per_device_eval_batch_size 1 \
    --run_name ${RUN_NAME} \
    --output_dir work_dirs/${RUN_NAME}  2>&1 | tee "work_dirs/${RUN_NAME}/training.log"

