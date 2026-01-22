#!/bin/bash
set -e

# Kaggle-specific setup with dual T4 GPUs
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
PORT=${PORT:-8000}

echo "Starting vLLM server on Kaggle (2xT4)..."

HUGGINGFACE_HUB_TOKEN=$HF_TOKEN \
CUDA_VISIBLE_DEVICES=0,1 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 2048 \
  --enable-chunked-prefill \
  --port $PORT \
  > /kaggle/working/vllm.log 2>&1 &

echo $! > /kaggle/working/vllm.pid
echo "Server started with PID: $(cat /kaggle/working/vllm.pid)"
echo "Logs: tail -f /kaggle/working/vllm.log"

# Wait for server to start
sleep 10
tail -n 50 /kaggle/working/vllm.log
