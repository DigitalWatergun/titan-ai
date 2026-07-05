#!/bin/bash

LLAMA_CPP=$HOME/code/llama.cpp/build/bin/llama-server
MODELS=$HOME/models

# Qwen3-Coder-Next — 80B MoE (~3B active), agentic-coding specialist.
# Download from: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF  (confirm the exact filename)
# Config: Q6 + full 256K context + f16 KV cache (the default — no -ctk/-ctv flags).
#   • UD-Q6_K_XL ≈ 66 GB — SHARDED. Download the folder into a model-named dir:
#       hf download unsloth/Qwen3-Coder-Next-GGUF --include "UD-Q6_K_XL/*" --local-dir ~/models/Qwen3-Coder-Next
#     (hf keeps the UD-Q6_K_XL/ subfolder.) Point at the FIRST shard below; llama.cpp auto-loads 00002/00003.
#   • ~66 GB model + 256K f16 KV (~15 GB) ≈ ~94 GB → comfortable on 128 GB (~34 GB free).
#   • Tight on RAM? add  -ctk q8_0 -ctv q8_0  (≈ half the KV memory, near-lossless; needs -fa on).
#   • Alt: UD-Q4_K_XL (49.6 GB) is a SINGLE root file if you'd rather skip shards (one step lower quality).
# NOTE: no --spec-type draft-mtp here — Coder-Next has no MTP head, so speculative decoding doesn't apply.
MODEL=$MODELS/Qwen3-Coder-Next/UD-Q6_K_XL/Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf

$LLAMA_CPP \
  --model $MODEL \
  --alias qwen3-coder-next \
  --port 8001 \
  --host 0.0.0.0 \
  -ngl 99 \
  --jinja \
  -fa on \
  -np 1 \
  -c 262144 &

echo "Waiting for server to be ready..."
sleep 15

if curl -s "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "✅ M5 Max Coder-Next (port 8001) — UP"
else
    echo "❌ M5 Max Coder-Next (port 8001) — DOWN (check logs)"
fi

echo ""
echo "Monitor with: mactop"
wait
