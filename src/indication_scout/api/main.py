"""FastAPI application."""

from fastapi import FastAPI

from indication_scout import __version__
from indication_scout.api.routes.analyses import router as analyses_router

app = FastAPI(
    title="IndicationScout API",
    description="API for drug repurposing and indication discovery",
    version=__version__,
)

app.include_router(analyses_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}
