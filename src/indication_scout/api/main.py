"""FastAPI application."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from indication_scout import __version__
from indication_scout.api.routes.analyses import router as analyses_router
from indication_scout.api.routes.drilldown import router as drilldown_router
from indication_scout.api.routes.examples import router as examples_router
from indication_scout.api.routes.examples import seed_example_cache
from indication_scout.config import get_settings
from indication_scout.constants import (
    BOT_USER_AGENT_MARKERS,
    CORS_ALLOW_ORIGINS,
    FRONTEND_DIST_DIR,
    GEO_API_FIELDS,
)
from indication_scout.metrics import record_http_request
from indication_scout.observability import (
    bind_log_context,
    configure_logging,
    reset_log_context,
)

configure_logging(get_settings().log_level)

logger = logging.getLogger(__name__)


class _PollingAccessLogFilter(logging.Filter):
    """Suppress Uvicorn access logs for high-frequency operational requests."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("GET /api/analyses/" in message or "GET /metrics/" in message)


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

# Per-IP geolocation cache: maps IP -> (location string, is_datacenter flag).
_geo_cache: dict[str, tuple[str, bool]] = {}


async def _geolocate(ip: str) -> tuple[str, bool]:
    """Return ("City, Region, Country", is_datacenter) for an IP via ip-api.com.

    `is_datacenter` is True when the IP belongs to a hosting provider or proxy/VPN —
    a strong signal the request is automated rather than a human browser. Cached per IP.
    Returns ("", False) on private/local IPs or any lookup failure — geolocation is
    best-effort and must never break a request.
    """
    if ip in _geo_cache:
        return _geo_cache[ip]
    location = ""
    is_datacenter = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": GEO_API_FIELDS},
            )
        data = resp.json()
        if data.get("status") == "success":
            parts = [data.get("city"), data.get("regionName"), data.get("country")]
            location = ", ".join(p for p in parts if p)
            is_datacenter = bool(data.get("hosting")) or bool(data.get("proxy"))
    except Exception as e:
        logger.debug("geolocation failed for %s: %s", ip, e)
    _geo_cache[ip] = (location, is_datacenter)
    return location, is_datacenter


def _is_bot_user_agent(user_agent: str) -> bool:
    """True when the User-Agent self-identifies as a known crawler / preview fetcher."""
    ua = user_agent.lower()
    return any(marker in ua for marker in BOT_USER_AGENT_MARKERS)


@app.middleware("http")
async def _log_client_ip(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    location, is_datacenter = await _geolocate(client_ip)
    user_agent = request.headers.get("user-agent", "")
    is_bot = _is_bot_user_agent(user_agent) or is_datacenter
    token = bind_log_context(
        request_id=request_id,
        client_ip=client_ip,
        client_location=location or None,
        client_is_automated=is_bot,
    )
    response: Response | None = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", "unmatched")
        duration = time.perf_counter() - started
        is_operational_request = request.url.path in {
            "/health",
            "/metrics",
            "/metrics/",
        }
        if not is_operational_request:
            record_http_request(request.method, route, status_code, duration)
        # Polling is measured but omitted from logs because the UI requests it frequently.
        if not is_operational_request and not (
            request.method == "GET"
            and route in {"/api/analyses/{job_id}", "/api/analyses/{job_id}/report"}
        ):
            logger.info(
                "HTTP request completed",
                extra={
                    "event_name": "http.request.completed",
                    "duration_seconds": duration,
                    "http_method": request.method,
                    "http_route": route,
                    "http_status_code": status_code,
                    "outcome": "success" if status_code < 500 else "error",
                },
            )
        reset_log_context(token)


app.include_router(analyses_router)
app.include_router(drilldown_router)
app.include_router(examples_router)
app.mount("/metrics", make_asgi_app(), name="metrics")


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


@app.on_event("shutdown")
async def close_run_persistence() -> None:
    """Dispose the run-ledger connection pool after request handling stops."""
    from indication_scout.api.routes.analyses import dispose_run_session_factory

    dispose_run_session_factory()


@app.on_event("startup")
async def seed_examples() -> None:
    """Seed the example cache from committed snapshots when the volume is empty."""
    seed_example_cache()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# Serve the built React bundle in prod. Mounted last so it doesn't shadow /api or /health.
# Absent in dev (Vite serves the frontend) — skip the mount so the app still boots.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
