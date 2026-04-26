#!/bin/bash
# Run this script on the Titan workstation

LLAMA_CPP=~/Documents/Development/git_repos/llama.cpp/build/bin/llama-server
MODELS=~/models
MODEL=$MODELS/qwen3.6-27b.Q4_K_M.gguf


$LLAMA_CPP \
  --model $MODEL \
  --port 8001 \
  --host 0.0.0.0 \
  --log-disable &

echo "Waiting for server to be ready..."
sleep 15

# Health check — verify each server is responding
if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "✅ Titan (port 8001) — UP"
else
    echo "❌ Titan (port 8001) — DOWN (check logs)"
fi

echo ""
echo "Verify GPU assignment: nvidia-smi"
wait
