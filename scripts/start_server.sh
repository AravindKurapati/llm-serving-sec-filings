#!/bin/bash
set -e

# Load environment variables
source .env 2>/dev/null || echo "No .env file found, using environment variables"

MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
PORT=${PORT:-8000}
GPU=${GPU:-0}

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU: $GPU"

HUGGINGFACE_HUB_TOKEN=$HF_TOKEN \
CUDA_VISIBLE_DEVICES=$GPU \
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --port $PORT
