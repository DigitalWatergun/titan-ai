# Titan

A local multi-agent coding assistant built with LangGraph, RAG, and local LLMs running on 4x Titan XP GPUs. Works as both a coding assistant and a general-purpose AI.

## Agents

| Agent       | GPU               | Role                                                    | Model            |
| ----------- | ----------------- | ------------------------------------------------------- | ---------------- |
| **Bridget** | GPU 0 (port 8001) | Router — classifies requests and chains agents          | qwen2.5:7b       |
| **Cody**    | GPU 1 (port 8002) | Code — writes, edits, and debugs code                   | qwen2.5-coder:7b |
| **Paige**   | GPU 2 (port 8003) | Research — answers questions, searches web and codebase | qwen2.5:7b       |
| **Mark**    | GPU 3 (port 8004) | Review — reviews code and suggests improvements         | llama3.1:8b      |

For complex tasks, the router chains multiple agents in sequence (e.g., research -> code -> review).

## Setup

```bash
git clone git@github.com:digitalwatergun/titan-ai.git
cd titan-ai

uv sync
uv tool install --editable .
```

## Usage

```bash
# Start the llama.cpp servers (run in a tmux pane)
./launch_models.sh

# General questions
titan chat

# Work on a codebase with indexing
titan chat --index

# Index a codebase explicitly
titan index
```

## Project Status

Currently in **Phase 3** — Multi-agent system with LangGraph and llama.cpp.

See the full project guide for the roadmap through all 5 phases.
