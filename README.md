# Titan

A local coding assistant built with LangGraph, RAG, and Qwen3.6 27B running across 4x Titan XP GPUs via pipeline parallelism. Textual-based TUI with streaming responses, tool call visibility, and semantic code search.

## Setup

```bash
git clone git@github.com:digitalwatergun/titan-ai.git
cd titan-ai

uv sync
```

## Running

```bash
# Start the llama.cpp server on the Titan workstation
./launch_models.sh

# Run from the project directory
uv run titan

# Or run in a specific directory
uv run titan --working-dir ~/projects/my-app
```

### Global Install

To run `titan` from any directory without activating the venv:

```bash
uv tool install /path/to/titan-ai

# Then just cd into any project and run
cd ~/projects/my-app
titan
```

Re-run `uv tool install` after code changes to update.

### Remote CLI (Mac → Workstation)

Titan can run on your Mac with LLM inference on the workstation over Tailscale. Set the endpoint in `.env`:

```bash
cp .env.example .env
# Edit .env to point TITAN_LLM_URL at the workstation's Tailscale IP
```

## Slash Commands

| Command  | Purpose                                      |
| -------- | -------------------------------------------- |
| `/index` | Index current directory (or `/index <path>`) |
| `/clear` | Clear conversation history                   |
| `/help`  | List available commands                      |
| `/quit`  | Exit                                         |

## Project Status

Currently in **Phase 4** — Textual TUI, conversation persistence, multi-project indexing.

See the full project guide for the roadmap through all 5 phases.
