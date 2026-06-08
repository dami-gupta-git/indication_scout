"""FastAPI application."""

import logging

import httpx
from fastapi import FastAPI, Request
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

_geo_cache: dict[str, str] = {}


async def _geolocate(ip: str) -> str:
    """Return "City, Region, Country" for an IP via ip-api.com, cached per IP.

    Returns "" on private/local IPs or any lookup failure — geolocation is best-effort
    and must never break a request.
    """
    if ip in _geo_cache:
        return _geo_cache[ip]
    location = ""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,city,regionName,country"},
            )
        data = resp.json()
        if data.get("status") == "success":
            parts = [data.get("city"), data.get("regionName"), data.get("country")]
            location = ", ".join(p for p in parts if p)
    except Exception as e:
        logger.debug("geolocation failed for %s: %s", ip, e)
    _geo_cache[ip] = location
    return location


@app.middleware("http")
async def _log_client_ip(request: Request, call_next):
    # Skip the high-frequency analysis polling endpoint to avoid log spam.
    if not request.url.path.startswith("/api/analyses/"):
        # Behind Railway's proxy the real client IP is the first entry of X-Forwarded-For;
        # fall back to the direct peer when the header is absent (e.g. local dev).
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (request.client.host if request.client else "unknown")
        )
        location = await _geolocate(client_ip)
        logger.info(
            "request from %s (%s): %s %s",
            client_ip,
            location or "unknown location",
            request.method,
            request.url.path,
        )
    return await call_next(request)


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
