# Deploying IndicationScout to Railway

The app is a single Docker image (FastAPI serving the API + the built React UI)
plus a Postgres database. Railway builds the `Dockerfile` from a connected GitHub
repo and runs the prod `api` stage automatically (it is the last stage on purpose).

The container entrypoint waits for Postgres, runs `alembic upgrade head` (which
also `CREATE EXTENSION vector`), then starts uvicorn on `$PORT`.

## One-time setup

### 1. Postgres
1. In the Railway project: **+ New → Database → Add PostgreSQL**.
2. Open the Postgres service → **Data / Query** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   (The migration also runs this, but doing it up front avoids permission surprises.)

### 2. App service (from GitHub)
1. **+ New → GitHub Repo →** select this repository.
2. Railway detects the `Dockerfile` and builds the last stage = prod `api`
   (UI bundled, no `--reload`). Nothing to configure for the build target.

### 3. Persistent volume
1. App service → **Volumes → New Volume**.
2. Mount path: **`/data`**  (holds the BioLORD model cache + the scout `_cache`).

### 4. Environment variables (app service → Variables)
| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (reference the Postgres service) |
| `DB_PASSWORD` | any non-empty string (config requires it; the DB itself ignores it) |
| `SCOUT_CACHE_DIR` | `/data/_cache` |
| `HF_HOME` | `/data/hf` (model cache dir — not HuggingFace hosting) |
| `HF_TOKEN` | a HuggingFace read token — avoids the anonymous rate limit on the first model download (see note below) |
| `ANTHROPIC_API_KEY` | your key |
| `PUBMED_API_KEY` | your key |
| `NCBI_API_KEY` | your key |
| `OPENFDA_API_KEY` | your key |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | optional (tracing; leave unset to disable) |

`PORT` is injected by Railway automatically — the image already binds to it.

### 5. Domain
App service → **Settings → Networking → Generate Domain**. Visit the URL — the
React UI loads at `/` and the API is under `/api` (health at `/health`).

## Deploys
Push to the connected branch → Railway rebuilds and redeploys. The volume, DB,
and variables persist across deploys; the model downloads once and is cached on
`/data`.

## Notes
- `DATABASE_URL` from Railway is `postgresql://…`; SQLAlchemy uses psycopg2 by
  default for that scheme, so no `+psycopg2` suffix is needed.
- First request after a cold start is slow (~10s) while BioLORD-2023 loads into
  memory; subsequent requests are fast.
- **BioLORD model download (`HF_TOKEN`).** On the very first analysis the app
  downloads BioLORD-2023 from huggingface.co into `/data/hf`. Anonymous requests
  to the HF Hub are rate-limited, and a throttled/failed download surfaces as
  "We couldn't connect to 'https://huggingface.co' to load the files, and
  couldn't find them in the cached files." Set `HF_TOKEN` (a free read token from
  https://huggingface.co/settings/tokens) to raise the limit and make the first
  download reliable. Once the model is cached on the `/data` volume it persists
  across deploys, so the token only matters for that first fetch. (Optional
  alternative once cached: set `HF_HUB_OFFLINE=1` to skip the hub entirely — but
  only after the model is present, or loads will fail.)
