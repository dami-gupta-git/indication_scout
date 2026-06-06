"""Unit tests for api/routes/analyses — no network/LLM/DB; the runner is mocked."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.api.routes.analyses import _execute
from indication_scout.services.job_store import JobStore


@pytest.fixture(autouse=True)
def fresh_store():
    """Swap the module-level job_store for an empty one per test."""
    store = JobStore()
    with patch("indication_scout.api.routes.analyses.job_store", store):
        yield store


# --- _execute lifecycle (background runner) ---


async def test_execute_sets_done_and_result_on_success(fresh_store):
    job = fresh_store.create("metformin")
    output = SupervisorOutput(drug_name="metformin", summary="ok")

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(return_value=(output, "report")),
    ):
        await _execute(job)

    assert job.status == "done"
    assert job.result is output
    assert job.error is None


async def test_execute_sets_error_on_failure(fresh_store):
    job = fresh_store.create("metformin")

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        await _execute(job)

    assert job.status == "error"
    assert job.error == "boom"
    assert job.result is None


async def test_execute_sets_cancelled_on_cancellation(fresh_store):
    job = fresh_store.create("metformin")

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await _execute(job)

    assert job.status == "cancelled"


# --- route layer (TestClient) ---


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from indication_scout.api.main import app

    return TestClient(app)


def test_post_returns_202_with_job_id(client, fresh_store):
    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(return_value=(SupervisorOutput(), "report")),
    ):
        resp = client.post("/api/analyses", json={"drug_name": "metformin"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] in {j.job_id for j in fresh_store.all()}
    assert body["status"] in {"pending", "running", "done"}


def test_get_unknown_job_returns_404(client, fresh_store):
    resp = client.get("/api/analyses/nope")
    assert resp.status_code == 404


def test_report_for_unfinished_job_returns_409(client, fresh_store):
    job = fresh_store.create("metformin")
    job.status = "running"

    resp = client.get(f"/api/analyses/{job.job_id}/report.md")
    assert resp.status_code == 409


def test_delete_unknown_job_returns_404(client, fresh_store):
    resp = client.delete("/api/analyses/nope")
    assert resp.status_code == 404


def test_delete_finished_job_is_idempotent_204(client, fresh_store):
    job = fresh_store.create("metformin")
    job.status = "done"
    job.result = SupervisorOutput()

    resp = client.delete(f"/api/analyses/{job.job_id}")
    assert resp.status_code == 204
    assert job.status == "done"
