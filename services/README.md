# Services

Everything needed to bring up the Titan stack on a host: per-host `llama-server` launch scripts, plus a Docker Compose stack for SearXNG and OpenWebUI.

Only one host runs the stack at a time. Pick the launch script that matches the current host.

## Contents

| File                             | Purpose                                                                    |
| -------------------------------- | -------------------------------------------------------------------------- |
| `launch_models_m5max.sh`         | Launches `llama-server` on the M5 Max (Metal, MTP enabled)                 |
| `launch_models_astral.sh`        | Launches `llama-server` on Astral (RTX 5090, MTP enabled)                  |
| `launch_models_titan.sh`         | Launches `llama-server` on the 4× Titan XP box (pipeline parallel, no MTP) |
| `docker-compose.yaml`            | SearXNG + OpenWebUI                                                        |
| `.env.example`                   | Compose-stack vars (copy to `.env`)                                        |
| `searxng/settings.yaml.template` | SearXNG config (copy to `settings.yaml`, then generate a secret)           |

## Ports

| Port | Service                                |
| ---- | -------------------------------------- |
| 8001 | `llama-server` (OpenAI-compatible API) |
| 8080 | SearXNG                                |
| 3000 | OpenWebUI                              |

## Bring-up order

```bash
# 1. Per-host bootstrap (first time on a host only)
cd services
cp .env.example .env

SECRET=$(openssl rand -hex 32)
sed "s|REPLACE_WITH_GENERATED_SECRET|$SECRET|" \
    searxng/settings.yaml.template > searxng/settings.yaml

# 2. Start llama-server (separate terminal or tmux pane)
./launch_models_m5max.sh        # or _astral.sh / _titan.sh

# 3. Start the compose stack
docker compose up -d
docker compose ps
```

On Mac, `docker compose` needs Colima running first: `colima start`.

`services/.env` and `services/searxng/settings.yaml` are gitignored — secrets stay local.

## Verify

```bash
curl http://localhost:8001/v1/models                       # llama-server, "titan" alias
curl 'http://localhost:8080/search?q=hello&format=json'    # SearXNG returns JSON
curl http://localhost:3000/                                 # OpenWebUI returns HTML
```

Then open `http://localhost:3000` in a browser to use OpenWebUI.

## Two `.env` files

- Root `.env` — agent vars (`TITAN_LLM_URL`, `TITAN_MODEL_NAME`, `SEARXNG_URL`). Read by the `titan` CLI.
- `services/.env` — compose-stack vars (`SEARXNG_BASE_URL`, `OPENAI_API_BASE_URL`). Auto-discovered by Docker Compose because it sits next to `docker-compose.yaml`.

The two don't overlap. Compose doesn't read the root `.env`; the agent doesn't read `services/.env`.

## Workstation deployment

When the stack later runs on Astral or Titan (instead of the Mac), edit `services/.env` and change `SEARXNG_BASE_URL` to the host's Tailscale IP. `OPENAI_API_BASE_URL` stays the same — `host.docker.internal` resolves correctly on both Mac and Linux thanks to the `extra_hosts` mapping in the compose file.

## Troubleshooting

```bash
docker compose logs -f open-webui
docker compose logs -f searxng
docker compose restart open-webui
docker compose down                       # stop everything (volumes persist)
docker volume inspect services_open-webui # where OpenWebUI's chat history lives
```

OpenWebUI data lives in the named volume `services_open-webui` inside the Colima VM (or directly on disk on Linux hosts). `docker compose down` does not delete it; `docker compose down -v` does — don't run that unless you actually want to wipe chats.
