#!/bin/bash

echo "Stopping FastAPI services..."

# Find and kill the FastAPI/uvicorn processes
pids=$(ps aux | grep 'rm_fastapi' | grep -v 'grep' | awk '{print $2}')
# 读取fastapi_pids.txt获得pids
# if [ -f "fastapi_pids.txt" ]; then
#     pids=$(cat fastapi_pids.txt)
# fi

if [ -z "$pids" ]; then
    echo "No FastAPI services are running."
    exit 0
fi

echo "Found FastAPI processes with PIDs: $pids"

# Kill the processes
for pid in $pids; do
    echo "Killing process $pid..."
    kill -15 $pid
    sleep 1
    
    # Check if process still exists, force kill if necessary
    if ps -p $pid > /dev/null; then
        echo "Process $pid didn't terminate gracefully, forcing..."
        kill -9 $pid
    fi
done

echo "All FastAPI services have been stopped."