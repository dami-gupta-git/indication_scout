"""FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from indication_scout import __version__
from indication_scout.api.routes.analyses import router as analyses_router
from indication_scout.api.routes.drilldown import router as drilldown_router
from indication_scout.api.routes.examples import router as examples_router
from indication_scout.api.routes.examples import seed_example_cache
from indication_scout.constants import CORS_ALLOW_ORIGINS, FRONTEND_DIST_DIR

logger = logging.getLogger(__name__)


class _PollingAccessLogFilter(logging.Filter):
    """Suppress uvicorn access-log lines for the analysis polling endpoint."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /api/analyses/" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_PollingAccessLogFilter())

app = FastAPI(
    title="IndicationScout API",
    description="API for drug repurposing and indication discovery",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyses_router)
app.include_router(drilldown_router)
app.include_router(examples_router)


@app.on_event("startup")
async def start_tracing() -> None:
    """Wire OpenTelemetry → Langfuse so web-triggered runs are traced (mirrors the CLI)."""
    from indication_scout.tracing import setup_tracing

    setup_tracing()


@app.on_event("shutdown")
async def stop_tracing() -> None:
    """Flush buffered spans before the process exits."""
    from indication_scout.tracing import shutdown_tracing

    shutdown_tracing()


@app.on_event("startup")
async def seed_examples() -> None:
    """Seed the example cache from committed snapshots when the volume is empty."""
    seed_example_cache()


@app.on_event("startup")
async def preload_embedding_model() -> None:
    """Pre-load the embedding model on startup so it's ready for requests.

    The BioLORD-2023 model is ~500 MB and takes ~10 s to download on first
    deployment. Loading it here — before the server begins accepting traffic —
    means the download happens once at startup rather than on the first
    analysis request, preventing that request from timing out and returning a
    502 to the client.
    """
    from indication_scout.services.embeddings import embed_async

    logger.info("Pre-loading embedding model...")
    await embed_async(["warmup"])
    logger.info("Embedding model ready")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# Serve the built React bundle in prod. Mounted last so it doesn't shadow /api or /health.
# Absent in dev (Vite serves the frontend) — skip the mount so the app still boots.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
