# Deployment

How IndicationScout is containerised and deployed. The app is a single Docker
image — FastAPI serves both the JSON API and the built React UI — plus a Postgres
(pgvector) database and a persistent volume for the embedding-model cache.

For the click-by-click Railway runbook, see [DEPLOY_RAILWAY.md](DEPLOY_RAILWAY.md).
This document covers the architecture, the files added, and the reasoning.

## Runtime topology

```
                 ┌──────────────────────────────┐
  browser  ───▶  │  app container (Docker image) │
                 │  ├ FastAPI  /api  /health      │
                 │  ├ static React bundle  /      │
                 │  └ BioLORD-2023 (sentence-tx)  │
                 └───────┬───────────────┬────────┘
                         │               │
              DATABASE_URL          /data volume
                         │          ├ /data/_cache  (API/LLM file cache)
                 ┌───────▼──────┐   └ /data/hf      (model weights)
                 │  Postgres    │
                 │  + pgvector  │
                 └──────────────┘
```

- **One image serves API + UI.** `api/main.py` mounts `frontend/dist` as static
  files when that directory exists, so the React bundle is baked into the prod
  image and served by uvicorn — no separate web server.
- **Postgres with pgvector** stores PubMed abstract embeddings. The first
  migration runs `CREATE EXTENSION IF NOT EXISTS vector`.
- **Persistent volume** at `/data` holds the embedding model (~0.5–1.5 GB,
  downloaded once) and the file cache, so neither is lost on restart/redeploy.

## Startup sequence

The image entrypoint (`docker/entrypoint.sh`) runs on every boot:

1. Parse host/port from `DATABASE_URL`, wait for Postgres with `pg_isready`.
2. `alembic upgrade head` — applies migrations (and creates the pgvector
   extension). Idempotent, so safe on every deploy.
3. `exec uvicorn …` on `$PORT`.

Migrations-on-boot means there is no separate migrate step to run or forget.

## Files added for deployment

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build: frontend → shared python base → `api-dev` / `api`. |
| `.dockerignore` | Keeps the build context small; excludes `.venv`, `node_modules`, caches, local `.env`. |
| `docker-compose.yml` | Prod-like local stack: `db` (pgvector) + `api`. Auto-migrates, serves UI on `:8000`. |
| `docker-compose.dev.yml` | Dev override: uvicorn `--reload` + bind mounts, Vite HMR container on `:5173`, `test` / `test-int` services. |
| `docker/entrypoint.sh` | Wait-for-DB → migrate → start uvicorn. |
| `railway.toml` | Tells Railway to build the Dockerfile; the prod `api` stage is the default (last) stage. |
| `DEPLOY_RAILWAY.md` | Step-by-step Railway dashboard runbook. |
| `DEPLOYMENT.md` | This file. |

### Dockerfile stages

- **`frontend-build`** (`node:20-slim`) — `npm install` + `npm run build`, output
  `frontend/dist`. Uses `npm install` not `npm ci`: the lockfile is resolved on
  macOS and `npm ci` strict-checks platform-specific esbuild/rollup binaries,
  which fails on a Linux image.
- **`python-base`** (`python:3.11-slim`) — system libs (`libpq5`,
  `postgresql-client`), **CPU-only PyTorch**, then `pip install -e .`, the
  constants file, and the entrypoint. Shared by both runtime stages.
- **`api-dev`** — no UI bundle; uvicorn `--reload`. Selected explicitly by the
  dev compose file. Intentionally **not** the last stage.
- **`api`** (last stage = default build target) — copies the built UI in, runs
  uvicorn on `$PORT`. This is what Railway builds.

Stage order matters: Railway (and other Dockerfile-only platforms) build the
**last** stage and offer no way to pick one, so `api` must come last.

## Local development

```bash
# Prod-like (single image serves API + UI, auto-migrates):
docker compose up --build
#   → http://localhost:8000   (UI + /api + /health)

# Dev (hot reload both sides):
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
#   → http://localhost:5173   (Vite UI, proxies /api → api container)
#   → http://localhost:8000   (API with --reload)

# Tests (one-shot services):
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test run --rm test       # unit
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test run --rm test-int   # integration
```

## Environment variables

Secrets/credentials are passed at runtime (never baked into the image):

| Variable | Local (compose) | Railway |
|---|---|---|
| `DATABASE_URL` | from `.env`, points at `db:5432` | `${{Postgres.DATABASE_URL}}` |
| `DB_PASSWORD` | from `.env` | any non-empty string (required by config) |
| `SCOUT_CACHE_DIR` | unset → `<root>/_cache` | `/data/_cache` |
| `HF_HOME` | volume `hf_cache` | `/data/hf` |
| `HF_TOKEN` | optional (model usually already cached) | recommended — avoids the anonymous HF rate limit on first download |
| `ANTHROPIC_API_KEY`, `PUBMED_API_KEY`, `NCBI_API_KEY`, `OPENFDA_API_KEY` | from `.env` | set as service variables |
| `PORT` | 8000 (default) | injected by Railway |

`.env.constants` (tunable numeric limits, not secrets) is **baked into the
image** — `config.py` requires those fields with no defaults.

`SCOUT_CACHE_DIR` is an env override added to `constants.py` so the file cache can
live on the mounted volume; unset, it falls back to the original `<root>/_cache`
path, so local/test behaviour is unchanged.

## Problems hit and fixed (so they don't recur)

These surfaced only by building/running the image — they passed locally because
the host had files or hardware the container did not.

1. **GPU PyTorch bloat / build OOM** — `sentence-transformers` pulled the default
   torch with the multi-GB CUDA stack (`nvidia-*-cu13`, `triton`), producing a
   ~6 GB image and exhausting build disk. Fixed by installing CPU-only torch from
   PyTorch's CPU wheel index first (image ~6 GB → ~2 GB).
2. **`.env.constants` missing in image** — config has no defaults for tunable
   limits, so startup crashed with pydantic "field required". Fixed by `COPY
   .env.constants` into the image.
3. **`db/` package absent on GitHub** — a bare `db` pattern in the user's *global*
   gitignore had silently excluded `src/indication_scout/db/`, so clean builds hit
   `ModuleNotFoundError: No module named 'indication_scout.db'`. Fixed by
   force-adding the package and removing the global pattern.
4. **Prompt files absent on GitHub** — a `*.txt` pattern in the local `.gitignore`
   excluded all 14 LLM prompt files (loaded at import time), crashing startup.
   Fixed by force-adding `src/indication_scout/prompts/*.txt` and removing the
   `*.txt` pattern.
5. **Stage selection** — Railway builds the last Dockerfile stage with no override.
   Fixed by ordering the prod `api` stage last.
6. **HF Hub rate limit on first model download** — the first analysis downloads
   BioLORD-2023 from huggingface.co; anonymous requests are rate-limited, and a
   throttled fetch surfaces as "couldn't connect to huggingface.co … and couldn't
   find them in the cached files." Set `HF_TOKEN` so the first download is
   reliable. After it caches on `/data` it persists across deploys.

**Lesson:** when something works locally but fails in a container, suspect what is
actually committed to the repo (gitignore exclusions), not the code.
