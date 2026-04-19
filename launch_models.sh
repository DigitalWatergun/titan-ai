#!/bin/bash
# ~/Documents/Development/git_repos/titan-ai/launch_models.sh

LLAMA_CPP=~/Documents/Development/git_repos/llama.cpp/build/bin/llama-server
MODELS=~/models
MODEL=$MODELS/qwen3.5-27b.Q4_K_M.gguf


$LLAMA_CPP \
  --model $MODEL \
  --port 8001 \
  --log-disable &

echo "Waiting for server to be ready..."
sleep 15

# Health check — verify each server is responding
if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "✅ $name (port 8001) — UP"
else
    echo "❌ $name (port 8001) — DOWN (check logs)"
fi

echo ""
echo "Verify GPU assignment: nvidia-smi"
wait
