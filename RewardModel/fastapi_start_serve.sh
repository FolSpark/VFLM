#!/bin/bash

# Configuration
BASE_PORT=30930  # Starting port number
# BASE_PORT=9502
NUM_INSTANCES=2  # Number of instances to launch
SCRIPT_PATH="rm_fastapi.py"
LOG_DIR="logs"   # Directory to store logs
# GPUs to use - one for each instance
# GPU_LIST=(0 1 2 3 4 5 6 7)
GPU_LIST=(6 7)

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

echo "Starting $NUM_INSTANCES FastAPI instances..."

# Launch instances
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$(($BASE_PORT + $i))
    GPU=${GPU_LIST[$i]}
    LOG_FILE="$LOG_DIR/fastapi_$PORT.log"
    
    # Start the FastAPI service with uvicorn specifying the port and GPU
    cd $(dirname $SCRIPT_PATH) && \
    CUDA_VISIBLE_DEVICES=$GPU NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
    uvicorn rm_fastapi:app --host 0.0.0.0 --port $PORT > $LOG_FILE 2>&1 &
    
    echo "Started FastAPI instance on port $PORT using GPU $GPU "
done

echo "All instances started."
