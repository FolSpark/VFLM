set -x

export VERL_SFT_LOGGING_LEVEL='ERROR'

# nccl settings
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=dfc00c655cf72be6f64d2cdc2e0c19eb1fa607ff # dumingxuan

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

hostfile=./hostfile_txh800_12345678

np=$(( `grep -v '#' $hostfile | grep -v ^$ | wc -l` * 8 )) 

# Change for multinode config
MASTER_ADDR=`ifconfig eth0 | sed -nr '2s/.*inet ([0-9.]+) .*/\1/p'`
MASTER_ADDR=`ifconfig eth0 | grep inet | grep -v inet6 | awk '{print $2}'`
MASTER_PORT=9000


# DATA_PATH=/mnt/yscfs/zhangshengzhuo/rl/dataset/math
DATA_PATH=/mnt/yscfs/xubenfeng/ys_agent_dev/thinkingagent/data/version/thinkingagent_v01_0221.jsonl

# change max_position_embeddings/sliding_window from 32768 to 131072
# tokenizer.json add 4 token: <think></think> <answer></answer>
MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-72B"
# MODEL_PATH="/mnt/yscfs/zhangshengzhuo/model/Qwen2.5-7B"
DATASET_NAME=$(basename $DATA_PATH | sed -e 's/sft_//g' | sed -e 's/.jsonl//g')
TRAIN_DATE=$(date "+%Y%m%d")
BASE_NAME="Qwen25-72B"
LR=1e-5
BSZ=128
EPS=2

save_path="outputs/${TRAIN_DATE}_${BASE_NAME}_${LR}_${EPS}_${DATASET_NAME}/"
LOG_PATH="outputs/${TRAIN_DATE}_${BASE_NAME}_${LR}_${EPS}_${DATASET_NAME}.log"


# 5e-6 1ep
# max_length=4096
# max_length=8192
# max_length=8192
# max_length=16384
# max_length=32768
# max_length=65536
max_length=131072

mpirun --allow-run-as-root -np $np \
        -hostfile $hostfile \
        -mca plm_rsh_args "-p 3391"  \
        --tag-output \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_PXN_DISABLE=0 \
        -x NCCL_DEBUG=WARN \
        -x GLOO_SOCKET_IFNAME=eth0 \
        -x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
        -x NCCL_COLLNET_ENABLE=1 \
        -x LD_LIBRARY_PATH -x PATH \
    python -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.max_length=$max_length \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.system_key=extra_info \
    data.task=agent \
    optim.lr=$LR \
    +data.system_dict_keys=['system'] \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=$BSZ \
    data.micro_batch_size_per_gpu=1 \
    data.truncation='right' \
    model.partial_pretrain=$MODEL_PATH \
    model.use_liger=true \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.cpu_offload=true \
    model.fsdp_config.offload_params=true \
    trainer.default_local_dir=$save_path \
    trainer.total_epochs=$EPS \
    trainer.project_name=think-sft \
    trainer.experiment_name=think-sft-qwen-2.5-72b \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    +master_addr=$MASTER_ADDR \
    +master_port=$MASTER_PORT 2>&1 | tee ${LOG_PATH}
