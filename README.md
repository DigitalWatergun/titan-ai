# Titan

A local coding assistant built with LangGraph, RAG, and Qwen3.6 27B. Textual-based TUI with streaming responses, tool call visibility, and semantic code search.

Runs on three hosts: M5 Max (Metal, primary dev), Astral (RTX 5090), and the original 4× Titan XP box (pipeline parallel).

## Setup

```bash
git clone git@github.com:digitalwatergun/titan-ai.git
cd titan-ai

uv sync
cp .env.example .env
```

## Services

`llama-server` plus the SearXNG + OpenWebUI stack live in [`services/`](services/README.md). Bring them up there before running `titan`:

```bash
# In one terminal — start llama-server
cd services
./launch_models_m5max.sh        # or _astral.sh / _titan.sh

# In another — start the compose stack
docker compose up -d
```

See `services/README.md` for the full bring-up sequence, ports, and troubleshooting.

## Running the agent

```bash
# From the project directory
uv run titan

# Or against a specific working directory
uv run titan --working-dir ~/projects/my-app
```

### Global install

Run `titan` from any directory without activating the venv:

```bash
uv tool install /path/to/titan-ai

cd ~/projects/my-app
titan
```

Re-run `uv tool install` after code changes to update.

### Remote CLI (Mac → workstation)

`titan` can run on one host with `llama-server` on another over Tailscale. Edit `.env`:

```
TITAN_LLM_URL=http://<workstation-tailscale-ip>:8001/v1
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

See the project guide in the Obsidian vault for the full roadmap.
