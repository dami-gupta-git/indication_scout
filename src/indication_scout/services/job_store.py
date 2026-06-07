"""In-memory job store for analysis runs.

Backs the async job model: `POST /api/analyses` creates a `Job` and launches the runner in a
background `asyncio.Task`; the frontend polls `GET /api/analyses/{job_id}` for status/result.

In-memory by design: jobs live in a module-level dict keyed by `job_id`, lost on restart, no
persisted history. Single-worker only — multiple uvicorn workers would not share this dict.
"""

import logging
import uuid
from asyncio import Task
from dataclasses import dataclass, field
from typing import Literal

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "done", "error", "cancelled"]


@dataclass
class Job:
    """A single analysis run and its lifecycle state.

    `task` is the live background `asyncio.Task` running the analysis; it is retained so the run
    can be cancelled (F5). `result` is populated only when `status == "done"`; `error` only when
    `status == "error"`. This is not an external-data model — it carries a live Task handle — so
    it is a dataclass, not a Pydantic ingestion model.
    """

    job_id: str
    drug_name: str
    status: JobStatus = "pending"
    result: SupervisorOutput | None = None
    error: str | None = None
    task: Task | None = field(default=None, repr=False)


class JobStore:
    """Module-singleton dict of `job_id -> Job`. Not thread-safe; single-worker asyncio only."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, drug_name: str) -> Job:
        """Create a `pending` job with a fresh id and register it."""
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, drug_name=drug_name)
        self._jobs[job_id] = job
        logger.info("Created job %s for drug=%s", job_id, drug_name)
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job, or None if unknown."""
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        """Return all jobs (insertion order)."""
        return list(self._jobs.values())


# Module-level singleton shared across the API layer.
job_store = JobStore()
