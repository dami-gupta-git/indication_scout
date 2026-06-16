# Dev Setup

> See also: [README.md](README.md) (project overview, installation, env/database setup, CLI usage).

Two ways to develop: **fully in Docker** (one command, matches prod) or
**natively** (backend + frontend on the host, only Postgres in Docker — fastest
reload, easiest debugging). Pick one.

---

# Option A — Native (host) + Dockerized Postgres

Fastest loop. `.env` already points `DATABASE_URL` at `localhost:5438`, which is
the Docker Postgres below.

## 1. Postgres (Docker)

Only the DB runs in a container:

```bash
docker compose up db
```

Postgres on `localhost:5438`; the `scout_test` DB is created on first boot.

## 2. Backend (uv + uvicorn)

```bash
uv venv                       # creates .venv (once)
uv pip install -e ".[dev]"    # runtime + dev tools (pytest, ruff, black, mypy)
source .venv/bin/activate

alembic upgrade head          # apply migrations to the scout DB (once / on schema change)
uvicorn indication_scout.api.main:app --reload
```

API on http://localhost:8000 with `--reload` — Python and prompt edits restart live.

The `scout` CLI is also installed by the editable install:

```bash
scout find -d "metformin"
```

## 3. Frontend (npm + Vite)

```bash
cd frontend
npm install                   # once / when package.json changes
npm run dev                   # Vite dev server + HMR
```

UI on http://localhost:5173. Vite proxies `/api` → `localhost:8000`.

## Native quick reference

| Task | Command |
| --- | --- |
| Backend deps | `uv pip install -e ".[dev]"` |
| Frontend deps | `cd frontend && npm install` |
| Migrate DB | `alembic upgrade head` |
| Run API | `uvicorn indication_scout.api.main:app --reload` |
| Run UI | `cd frontend && npm run dev` |
| Unit tests | `pytest tests/unit/` |
| Frontend tests | `cd frontend && npm test` |
| Lint / format (py) | `ruff check src/ tests/` · `black src/ tests/` |

---

# Option B — Fully Dockerized

One command, no image rebuilds for code changes.

## Run (live reload)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

- **Frontend:** http://localhost:5173 — Vite dev server with HMR. React/CSS edits
  appear instantly. **Use this URL for dev, not :8000.**
- **Backend:** http://localhost:8000 — uvicorn `--reload`; `src/` is bind-mounted,
  so Python and prompt (`.txt`) edits restart the server live.
- **DB:** Postgres on `localhost:5438`.

`:8000` still serves the prod-style API (and the baked frontend bundle from the last
image build, which is stale during dev) — hit **:5173** for the UI.

## When a rebuild IS needed (Docker)

| Change | Action |
| --- | --- |
| Python / prompt / CSS / React source | none — auto-reload |
| `pyproject.toml` (pip deps) | `... up --build` |
| `frontend/package.json` (npm deps) | restart the `frontend` service (runs `npm install` on boot) |

## Volumes

- `scout_cache` (`/app/cache`) is shared with the prod compose — the seeded example
  cache and all data-source caches persist across runs.
- `hf_cache` holds the BioLORD-2023 model so it downloads once.

## Tests (one-shot)

```bash
# Unit (no network, no DB)
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test run --rm test

# Integration (needs DB + real API keys + network)
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile test run --rm test-int
```

## Prod image (only to verify the bundled build)

```bash
docker compose up --build
```

Serves the baked frontend bundle at http://localhost:8000. Use only to sanity-check
what Railway will deploy — day-to-day work happens on :5173.
