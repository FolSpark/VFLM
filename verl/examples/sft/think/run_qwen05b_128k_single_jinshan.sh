set -x

export VERL_SFT_LOGGING_LEVEL='ERROR'
# DATA_PATH=/mnt/jfs/zhangshengzhuo/rl/verl/examples/data/math
DATA_PATH=/mnt/jfs/zhangshengzhuo/rl/verl/examples/data/Chinese-DeepSeek-R1-Distill-data-110k
# DATA_PATH=/mnt/jfs/zhangshengzhuo/rl/verl/examples/data/sft_long_32k128k

# MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-0.5B-Instruct"
# MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-0.5B-Instruct"
# MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-7B"
MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/Qwen2.5-0.5B"
# MODEL_PATH="/mnt/jfs/xingmt/model/source_models/Qwen_family/Qwen2.5/DeepSeek-R1-Distill-Qwen-32B"

nproc_per_node=8
save_path=output

# max_length=4096
# max_length=8192
# max_length=8192
# max_length=16384
# max_length=32768
max_length=131072

# Shift the arguments so $@ refers to the rest
# shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.max_length=$max_length \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-5 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=16 \
    model.partial_pretrain=$MODEL_PATH \
    model.use_liger=true \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    trainer.default_local_dir=$save_path \
    trainer.total_epochs=1 \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct-sp2-liger \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
