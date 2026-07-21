# Titan

A local coding agent — a simplified Claude Code that runs entirely on your own hardware. Native async agent loop (`httpx` + a plain `Agent` dataclass, no framework), Textual TUI with streaming responses and tool-call visibility, RAG over code / notes / past chats via ChromaDB, and a local LLM served by llama.cpp.

Runs on three hosts: M5 Max (Metal, primary — Qwen3-Coder-Next 80B MoE, 256K context), Astral (RTX 5090), and the original 4× Titan XP box (Qwen3.6 27B, pipeline parallel).

## What it can do

- **Agentic tool loop** — read/write files, list directories, run shell commands, all streamed live in the TUI.
- **Semantic search** — `search_codebase` (indexed projects), `search_vault` (Obsidian notes), `search_chats` (past conversations), backed by ChromaDB + nomic-embed-text.
- **Web research** — `web_search` (self-hosted SearXNG) to find pages, `web_fetch` (httpx + trafilatura) to read them as clean markdown.
- **Long sessions** — chats persist to disk, titles auto-generate, context auto-compacts at 80% with old turns recoverable via chat memory.
- **Pre-flight index estimates** — `/index` counts chunks and asks before large embeds.

## Architecture

```
You (Textual TUI)
    ↓
Agent loop (run_turn: stream → tool calls → loop — src/titan/loop.py)
    ↓ OpenAI-compatible API
llama-server (local model via llama.cpp)
    ↓
    ├── Tools: read_file, write_file, list_directory, run_command
    ├── Tools: search_codebase, search_vault, search_chats (ChromaDB RAG)
    └── Tools: web_search (SearXNG), web_fetch (trafilatura)
```

## Setup

```bash
git clone git@github.com:digitalwatergun/titan-ai.git
cd titan-ai

uv sync
cp .env.example .env
```

`.env` variables:

| Variable             | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `TITAN_LLM_URL`      | llama-server endpoint (local, or a workstation over Tailscale) |
| `TITAN_MODEL_NAME`   | Model alias (matches `--alias` in the launch script)           |
| `SEARXNG_URL`        | SearXNG endpoint for `web_search`                              |
| `OBSIDIAN_VAULT_DIR` | Obsidian vault path for `/index vault` and `search_vault`      |

## Services

`llama-server` plus the SearXNG + OpenWebUI stack live in [`services/`](services/README.md). Bring them up before running `titan`:

```bash
# In one terminal — start llama-server
cd services
./launch_models_m5max_coder.sh   # Qwen3-Coder-Next on the M5 Max
                                 # or launch_models_m5max.sh / _astral.sh / _titan.sh

# In another — start the compose stack (SearXNG, OpenWebUI)
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

### Global install (uv tool)

Run `titan` from any directory without activating the venv. Install it as an **editable** uv tool:

```bash
uv tool install --editable /path/to/titan-ai

cd ~/projects/my-app
titan
```

This creates a dedicated venv under `~/.local/share/uv/tools/titan-ai/` — separate from the project `.venv` that `uv run` uses. Because the install is editable:

- **Code changes are live immediately** — no reinstall needed.
- **Dependency changes are not.** After any `uv add` / `uv remove`, sync the tool venv:

```bash
uv tool upgrade titan-ai
```

Skipping this leaves the global `titan` crashing at import (`ModuleNotFoundError` for the new package) while `uv run titan` still works — the tool venv only re-resolves dependencies when told.

Don't omit `--editable` when (re)installing: a plain `uv tool install .` copies a frozen snapshot of the code, and changes stop flowing to the global command until the next reinstall.

### Remote CLI (Mac → workstation)

`titan` can run on one host with `llama-server` on another over Tailscale. Edit `.env`:

```
TITAN_LLM_URL=http://<workstation-tailscale-ip>:8001/v1
```

## Slash Commands

| Command                     | Purpose                                           |
| --------------------------- | ------------------------------------------------- |
| `/index`                    | Index the current directory (or `/index <path>`)  |
| `/index vault`              | Index the Obsidian vault (`.md` only)             |
| `/index chats`              | Index saved chats for `search_chats`              |
| `/index list`               | Show indexed collections and chunk counts         |
| `/index delete <name\|all>` | Remove an index (`all` requires the literal word) |
| `/compact`                  | Summarize old turns and free context now          |
| `/clear`                    | Start a fresh chat                                |
| `/help`                     | List commands and shortcuts                       |
| `/quit`                     | Exit                                              |

Large indexes show a chunk/size estimate and ask for confirmation before embedding.

## Keyboard Shortcuts

| Key         | Action                                  |
| ----------- | --------------------------------------- |
| `Esc`       | Interrupt the current response or index |
| `Ctrl+C`    | Cancel running work; quit when idle     |
| `Ctrl+E`    | Toggle the recent-chats sidebar         |
| `Ctrl+O`    | Toggle thinking display                 |
| `PageUp/Dn` | Scroll the chat log                     |

## Data

Everything lives under `~/.titan/`:

| Path                 | Contents                                      |
| -------------------- | --------------------------------------------- |
| `~/.titan/chats/`    | Chat JSON files + `index.json` manifest       |
| `~/.titan/chromadb/` | Vector store (per-project code, vault, chats) |
