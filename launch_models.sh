#!/bin/bash
# ~/Documents/Development/git_repos/titan-ai/launch_models.sh

LLAMA_CPP=~/Documents/Development/git_repos/llama.cpp/build/bin/llama-server
MODELS=~/models

# GPU 0: Router (qwen2.5-3b) — fast classification
$LLAMA_CPP \
  --model $MODELS/qwen2.5-7b-instruct.Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --split-mode none \
  --main-gpu 0 \
  --port 8001 \
  --ctx-size 2048 \
  --alias "router" \
  --log-disable &

# GPU 1: Code Agent (qwen2.5-coder-7b)
$LLAMA_CPP \
  --model $MODELS/qwen2.5-coder-7b-instruct.Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --split-mode none \
  --main-gpu 1 \
  --port 8002 \
  --ctx-size 16384 \
  --alias "code" \
  --log-disable &

# GPU 2: Research Agent (qwen2.5-7b)
$LLAMA_CPP \
  --model $MODELS/qwen2.5-7b-instruct.Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --split-mode none \
  --main-gpu 2 \
  --port 8003 \
  --ctx-size 16384 \
  --alias "research" \
  --log-disable &

# GPU 3: Review Agent (llama-3.1-8b)
$LLAMA_CPP \
  --model $MODELS/llama-3.1-8b-instruct.Q5_K_M.gguf \
  --n-gpu-layers 999 \
  --split-mode none \
  --main-gpu 3 \
  --port 8004 \
  --ctx-size 8192 \
  --alias "review" \
  --log-disable &

echo "All models launched. Waiting for servers to be ready..."
sleep 5

# Health check — verify each server is responding
SERVERS=("Router:8001" "Code:8002" "Research:8003" "Review:8004")
for server in "${SERVERS[@]}"; do
    name="${server%%:*}"
    port="${server##*:}"
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ $name (port $port) — UP"
    else
        echo "❌ $name (port $port) — DOWN (check logs)"
    fi
done

echo ""
echo "Verify GPU assignment: nvidia-smi"
wait
