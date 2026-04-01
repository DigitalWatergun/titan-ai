# TitanAI

A local multi-agent coding assistant built with LangGraph, RAG, and local LLMs running on 4x Titan XP GPUs. Works as both a coding assistant and a general-purpose AI.

## Overview

Titan uses specialized AI agents that each run on their own GPU:

| Agent       | Role                                                    | Model            |
| ----------- | ------------------------------------------------------- | ---------------- |
| **Bridget** | Router — directs requests to the right specialist       | qwen2.5:3b       |
| **Cody**    | Code — writes, edits, and debugs code                   | qwen2.5-coder:7b |
| **Paige**   | Research — answers questions, searches web and codebase | mistral:7b       |
| **Mark**    | Review — reviews code and suggests improvements         | llama3:8b        |

For complex tasks, the router chains multiple agents in sequence (e.g., research → code → review).

## Requirements

- 4x NVIDIA Titan XP (12GB each)
- 128GB RAM
- Ubuntu 24.04
- NVIDIA Driver 535+
- CUDA 12.2+

## Setup

```bash
# Clone the repo
git clone git@github.com:youruser/titan-ai.git
cd titan-ai

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Pull the model (Phase 1)
ollama pull qwen2.5-coder:7b

# Run
titan
```

# Run

Project Status

Currently in Phase 1 — single agent with tool use (read/write files, run commands).

See the full project guide for the roadmap through all 5 phases.
