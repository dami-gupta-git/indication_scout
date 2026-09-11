"""In-memory task registry for API analysis runs.

Postgres owns durable lifecycle state. This registry retains live `asyncio.Task` handles so the
API process that started a run can cancel it. It is lost on restart and remains single-process.
"""

import logging
import uuid
from asyncio import Task
from dataclasses import dataclass, field
from typing import Literal

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.api.schemas.progress import ProgressEvent

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "done", "error", "cancelled"]


@dataclass
class Job:
    """A single analysis run and its lifecycle state.

    `task` is the live background `asyncio.Task` running the analysis; it is retained so the run
    can be cancelled (F5). `result` is populated only when `status == "done"`; `error` only when
    `status == "error"`. `progress` is a live feed of user-facing milestones, appended by the
    runner via `emit` and returned on every poll. This is not an external-data model — it
    carries a live Task handle — so it is a dataclass, not a Pydantic ingestion model.
    """

    job_id: str
    drug_name: str
    status: JobStatus = "pending"
    result: SupervisorOutput | None = None
    error: str | None = None
    task: Task | None = field(default=None, repr=False)
    progress: list[ProgressEvent] = field(default_factory=list)

    def emit(self, phase: str, message: str) -> None:
        """Append a progress milestone. Latest event per phase wins on the frontend; we keep
        the full append-only list so the UI can render counts as each phase completes.
        """
        self.progress.append(ProgressEvent(phase=phase, message=message))


class JobStore:
    """Process-local registry of live tasks and their convenience state mirror."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, drug_name: str, *, job_id: str | None = None) -> Job:
        """Register a pending runtime job, using a supplied durable id when present."""
        resolved_job_id = job_id or uuid.uuid4().hex
        job = Job(job_id=resolved_job_id, drug_name=drug_name)
        self._jobs[resolved_job_id] = job
        logger.info("Created job %s for drug=%s", resolved_job_id, drug_name)
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job, or None if unknown."""
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        """Return all jobs (insertion order)."""
        return list(self._jobs.values())


# Module-level singleton shared across the API layer.
job_store = JobStore()
