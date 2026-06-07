# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the React bundle. Output lands in /frontend/dist and is
# copied into the API image so FastAPI can serve it (see api/main.py mount).
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build
WORKDIR /frontend

# Install deps first so this layer caches unless the lockfile changes.
# Use `npm install` rather than `npm ci`: the committed lockfile is resolved on
# the host platform (macOS), and npm ci strict-checks platform-specific optional
# deps (esbuild/rollup binaries), which fails when building on a Linux image.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: shared Python base. Installs the package and its deps once; both
# the prod and dev targets build on top of it.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS python-base
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace model cache — mounted as a volume so BioLORD-2023 (~500MB)
    # downloads once and survives container restarts.
    HF_HOME=/hf-cache \
    # Port uvicorn binds to. Defaults to 8000 for local/compose; PaaS hosts
    # (Render, Railway) inject their own PORT and it is honoured at runtime.
    PORT=8000

# psycopg2 and sentence-transformers need a few system libs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps. Copy only the project metadata first so the dependency
# layer caches unless pyproject changes.
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
RUN pip install -e .

# Tunable numeric limits (not secrets) — config.py loads these from the project
# root at startup, and the fields have no defaults, so the file must be present.
COPY .env.constants ./

# Entrypoint waits for Postgres, applies migrations, then execs the CMD.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

# ---------------------------------------------------------------------------
# Stage 3a: dev image. No frontend bundle (Vite serves it live in its own
# container). src/ is bind-mounted by the dev compose file and uvicorn runs
# with --reload. Selected explicitly via `target: api-dev` in the dev compose
# file — it is intentionally NOT the last stage so platforms that build the
# default (last) stage get the prod `api` image below.
# ---------------------------------------------------------------------------
FROM python-base AS api-dev
COPY alembic/ ./alembic/
COPY alembic.ini ./
EXPOSE 8000
CMD ["uvicorn", "indication_scout.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---------------------------------------------------------------------------
# Stage 3b: production image (LAST stage = the default build target). Bundle the
# built frontend in and serve it from uvicorn alongside the API. Railway and
# other Dockerfile-only platforms build this stage automatically.
# ---------------------------------------------------------------------------
FROM python-base AS api
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY --from=frontend-build /frontend/dist ./frontend/dist
EXPOSE 8000
# Shell form so ${PORT} expands at runtime (PaaS hosts assign it dynamically).
CMD uvicorn indication_scout.api.main:app --host 0.0.0.0 --port ${PORT}
