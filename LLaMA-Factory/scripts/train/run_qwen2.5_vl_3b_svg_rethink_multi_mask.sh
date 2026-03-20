set -e
set -x

# export NCCL_P2P_DISABLE="1"
# export IBN_P2P_DISABLE="1"
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=<your_api_key>
export WANDB_PROJECT=design_sft

# 802816 = 28*28*1024 相当于896*896大小图片
export IMAGE_PLACEHOLDER="<|image|>"

wandb online


GPU_NUM=8
TOTAL_BATCH_SIZE=64
PER_GPU_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$(( TOTAL_BATCH_SIZE / ( GPU_NUM * PER_GPU_BATCH_SIZE ) ))

RUN_NAME="qwen2.5_vl_3b_svg"


mkdir -p work_dirs/${RUN_NAME}

accelerate launch --num_machines 1 --num_processes ${GPU_NUM} --main_process_port 29502 --mixed_precision bf16 \
  src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path models/Qwen2.5-VL-3B-Instruct \
    --preprocessing_num_workers 64 \
    --finetuning_type full \
    --template qwen2_vl \
    --flash_attn fa2 \
    --enable_liger_kernel True \
    --dataset_dir data \
    --dataset layout_svg_multi_mix_no_mask_v3_train \
    --eval_dataset layout_svg_multi_mix_no_mask_v3_test \
    --tokenized_path data/datasets/svg-data/tokenized_rethink_mix_v3_tofix \
    --media_dir data/datasets/svg-data/data \
    --overwrite_cache \
    --mask_first_turn_only True \
    --cutoff_len 32768 \
    --image_max_pixels 802816 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size ${PER_GPU_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_only_model \
    --save_strategy epoch \
    --save_steps 300 \
    --warmup_ratio 0.03 \
    --packing False \
    --report_to wandb \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 21600000000 \
    --optim adamw_torch \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size 1 \
    --run_name ${RUN_NAME} \
    --output_dir work_dirs/${RUN_NAME}  2>&1 | tee "work_dirs/${RUN_NAME}/training.log"

