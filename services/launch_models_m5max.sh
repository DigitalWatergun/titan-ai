#!/bin/bash

LLAMA_CPP=$HOME/code/llama.cpp/build/bin/llama-server
MODELS=$HOME/models
MODEL=$MODELS/Qwen3.6-27B-UD-Q4_K_XL.gguf

$LLAMA_CPP \
  --model $MODEL \
  --alias qwen3.6-27b \
  --port 8001 \
  --host 0.0.0.0 \
  -ngl 99 \
  -fa on \
  -np 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --log-disable &

echo "Waiting for server to be ready..."
sleep 15

if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "✅ M5 Max (port 8001) — UP"
else
    echo "❌ M5 Max (port 8001) — DOWN (check logs)"
fi

echo ""
echo "Monitor with: mactop"
wait
