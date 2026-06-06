"""Unit tests for api/routes/analyses — no network/LLM/DB; the runner is mocked."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
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


# --- end-to-end: structured result through POST -> poll -> GET ---


def _poll_until_done(client, job_id, max_polls=50):
    """Poll GET like the frontend does until the job reaches a terminal status."""
    for _ in range(max_polls):
        body = client.get(f"/api/analyses/{job_id}").json()
        if body["status"] in {"done", "error", "cancelled"}:
            return body
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_structured_result_round_trips_through_post_and_get(client, fresh_store):
    output = SupervisorOutput(
        drug_name="duloxetine",
        candidate_diseases=["alcohol dependence", "obesity", "bipolar disorder"],
        disease_findings=[
            CandidateFindings(disease="alcohol dependence", source="mechanism"),
            CandidateFindings(disease="obesity", source="both"),
        ],
        top_diseases=["alcohol dependence", "obesity"],
        summary="Duloxetine shows mechanism-grounded signals in mood and metabolic indications.",
    )

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(return_value=(output, "report")),
    ):
        created = client.post("/api/analyses", json={"drug_name": "duloxetine"})
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        body = _poll_until_done(client, job_id)

    assert body["status"] == "done"
    assert body["error"] is None
    result = body["result"]
    assert result["drug_name"] == "duloxetine"
    assert result["candidate_diseases"] == ["alcohol dependence", "obesity", "bipolar disorder"]
    assert result["top_diseases"] == ["alcohol dependence", "obesity"]
    assert result["summary"].startswith("Duloxetine shows mechanism-grounded")
    assert len(result["disease_findings"]) == 2
    assert result["disease_findings"][0] == {
        "disease": "alcohol dependence",
        "source": "mechanism",
        "literature": None,
        "clinical_trials": None,
        "blurb": None,
    }
