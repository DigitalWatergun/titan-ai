#!/bin/bash

LLAMA_CPP=$HOME/code/llama.cpp/build/bin/llama-server
MODELS=$HOME/models
MODEL=$MODELS/Qwen3.6-27B-UD-Q4_K_XL.gguf

$LLAMA_CPP \
  --model $MODEL \
  --alias titan \
  --port 8001 \
  --host 0.0.0.0 \
  -ngl 99 \
  --jinja \
  --log-disable &

echo "Waiting for server to be ready..."
sleep 15

if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "✅ Titan (port 8001) — UP"
else
    echo "❌ Titan (port 8001) — DOWN (check logs)"
fi

echo ""
echo "Verify GPU assignment: nvidia-smi"
echo "Monitor with: nvtop"
wait
