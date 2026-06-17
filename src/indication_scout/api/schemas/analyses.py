"""Request/response schemas for the analyses API.

The response wraps the existing `SupervisorOutput` — its shape is NOT re-derived here.
"""

from pydantic import BaseModel, Field

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.api.schemas.progress import ProgressEvent
from indication_scout.services.job_store import JobStatus


class AnalysisRequest(BaseModel):
    """Body of `POST /api/analyses`."""

    drug_name: str = Field(min_length=1)


class AnalysisCreatedResponse(BaseModel):
    """`202` response from `POST /api/analyses`."""

    job_id: str
    status: JobStatus


class AnalysisStatusResponse(BaseModel):
    """Response from `GET /api/analyses/{job_id}` (the polling path).

    `result` is set only when `status == "done"`; `error` only when `status == "error"`.
    """

    job_id: str
    drug_name: str
    status: JobStatus
    result: SupervisorOutput | None = None
    error: str | None = None
    progress: list[ProgressEvent] = []
