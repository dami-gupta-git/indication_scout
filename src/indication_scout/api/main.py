"""FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from indication_scout import __version__
from indication_scout.api.routes.analyses import router as analyses_router
from indication_scout.api.routes.drilldown import router as drilldown_router
from indication_scout.constants import CORS_ALLOW_ORIGINS, FRONTEND_DIST_DIR

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


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# Serve the built React bundle in prod. Mounted last so it doesn't shadow /api or /health.
# Absent in dev (Vite serves the frontend) — skip the mount so the app still boots.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
