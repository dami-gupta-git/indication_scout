"""Unit tests for api/routes/analyses — no network/LLM/DB; the runner is mocked."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.api.routes.analyses import _execute
from indication_scout.services.job_store import JobStore


class FakeRunRepository:
    """In-memory stand-in for the durable repository used by route tests."""

    def __init__(self):
        self.runs = {}
        self.attempts = {}
        self.events = {}

    def create_run(
        self,
        drug_name,
        execution_mode,
        *,
        submission_source,
        analysis_kind,
        disease_name,
        date_before,
        run_id=None,
    ):
        resolved_id = run_id or f"{len(self.runs) + 1:032x}"
        run = SimpleNamespace(
            run_id=resolved_id,
            drug_name=drug_name,
            disease_name=disease_name,
            status="pending",
            execution_mode=execution_mode,
            submission_source=submission_source,
            analysis_kind=analysis_kind,
            date_before=date_before,
            result=None,
            integrity_status=None,
            cancellation_requested_at=None,
        )
        self.runs[resolved_id] = run
        self.attempts[resolved_id] = []
        self.events[resolved_id] = []
        return run

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def list_attempts(self, run_id):
        return self.attempts[run_id]

    def list_events(self, run_id):
        return self.events[run_id]

    def start_attempt(self, run_id, **identity):
        run = self.runs[run_id]
        if run.status != "pending":
            from indication_scout.services.run_repository import RunStateError

            raise RunStateError("not pending")
        run.status = "running"
        attempt = SimpleNamespace(
            attempt_id=f"{len(self.attempts[run_id]) + 101:032x}",
            error_message=None,
            **identity,
        )
        self.attempts[run_id].append(attempt)
        return attempt

    def append_event(
        self,
        run_id,
        event_name,
        severity,
        *,
        attempt_id=None,
        stage=None,
        attributes=None,
        **unused,
    ):
        event = SimpleNamespace(
            event_name=event_name,
            severity=severity,
            attempt_id=attempt_id,
            stage=stage,
            attributes=attributes,
        )
        self.events[run_id].append(event)
        return event

    def complete_validated_attempt(self, run_id, attempt_id, result):
        run = self.runs[run_id]
        run.status = "done"
        run.result = result.model_dump(mode="json")
        run.integrity_status = "passed"
        return run

    def record_attempt_cost(self, run_id, attempt_id, snapshot):
        self.attempts[run_id][-1].cost_snapshot = snapshot
        return self.attempts[run_id][-1]

    def fail_attempt(self, run_id, attempt_id, **failure):
        run = self.runs[run_id]
        run.status = "error"
        self.attempts[run_id][-1].error_message = failure["error_message"]
        return run

    def request_cancellation(self, run_id):
        self.runs[run_id].cancellation_requested_at = object()
        return self.runs[run_id]

    def cancel_pending_run(self, run_id):
        self.runs[run_id].status = "cancelled"
        return self.runs[run_id]

    def cancel_attempt(self, run_id, attempt_id):
        self.runs[run_id].status = "cancelled"
        return self.runs[run_id]


@pytest.fixture(autouse=True)
def fresh_store():
    """Swap the module-level job_store for an empty one per test."""
    store = JobStore()
    with patch("indication_scout.api.routes.analyses.job_store", store):
        yield store


@pytest.fixture(autouse=True)
def run_ledger():
    """Replace database repository scopes with one shared in-memory ledger."""
    ledger = FakeRunRepository()

    @contextmanager
    def scope():
        yield ledger

    with patch("indication_scout.api.routes.analyses._repository_scope", scope):
        yield ledger


def create_persisted_job(store, ledger, drug="metformin", mode="live"):
    run = ledger.create_run(
        drug,
        mode,
        submission_source="api",
        analysis_kind="find",
        disease_name=None,
        date_before=None,
    )
    return store.create(drug, job_id=run.run_id)


@pytest.fixture(autouse=True)
def no_seed_report():
    """Default the seed-report shortcut to a miss so _execute hits run_analysis.

    Tests that exercise the seed-hit path patch this themselves.
    """
    with patch(
        "indication_scout.api.routes.analyses.load_fresh_seed_report",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def drug_resolves():
    """Default the fail-fast existence check to a hit so POST reaches the job path.

    Tests that exercise the not-found path patch this themselves.
    """
    with patch(
        "indication_scout.api.routes.analyses.resolve_drug_name",
        new=AsyncMock(return_value="CHEMBL1431"),
    ):
        yield


# --- _execute lifecycle (background runner) ---


async def test_execute_sets_done_and_result_on_success(fresh_store, run_ledger):
    job = create_persisted_job(fresh_store, run_ledger)
    output = SupervisorOutput(drug_name="metformin", summary="ok")

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(return_value=(output, "report")),
    ):
        await _execute(job)

    assert job.status == "done"
    assert job.result is output
    assert job.error is None


async def test_execute_sets_error_on_failure(fresh_store, run_ledger):
    job = create_persisted_job(fresh_store, run_ledger)

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        await _execute(job)

    assert job.status == "error"
    assert job.error == "boom"
    assert job.result is None


async def test_execute_sets_cancelled_on_cancellation(fresh_store, run_ledger):
    job = create_persisted_job(fresh_store, run_ledger)

    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await _execute(job)

    assert job.status == "cancelled"


async def test_execute_serves_seed_report_without_running(fresh_store, run_ledger):
    """A fresh seed report short-circuits the run: run_analysis is never called."""
    job = create_persisted_job(fresh_store, run_ledger, mode="seed")
    seed = SupervisorOutput(drug_name="metformin", summary="seeded")
    run = AsyncMock()

    with (
        patch(
            "indication_scout.api.routes.analyses.load_fresh_seed_report",
            return_value=seed,
        ),
        patch("indication_scout.api.routes.analyses.run_analysis", new=run),
        patch("indication_scout.api.routes.analyses.asyncio.sleep", new=AsyncMock()),
    ):
        await _execute(job)

    run.assert_not_awaited()
    assert job.status == "done"
    assert job.result is seed
    assert job.error is None


async def test_execute_runs_live_when_no_seed_report(fresh_store, run_ledger):
    """No seed report falls through to run_analysis."""
    job = create_persisted_job(fresh_store, run_ledger)
    output = SupervisorOutput(drug_name="metformin", summary="live")
    run = AsyncMock(return_value=(output, "report"))

    with (
        patch(
            "indication_scout.api.routes.analyses.load_fresh_seed_report",
            return_value=None,
        ),
        patch("indication_scout.api.routes.analyses.run_analysis", new=run),
    ):
        await _execute(job)

    run.assert_awaited_once()
    assert job.status == "done"
    assert job.result is output


# --- route layer (TestClient) ---


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from indication_scout.api.main import app

    return TestClient(app)


def test_post_returns_202_with_job_id(client, fresh_store, run_ledger):
    with patch(
        "indication_scout.api.routes.analyses.run_analysis",
        new=AsyncMock(return_value=(SupervisorOutput(), "report")),
    ):
        resp = client.post("/api/analyses", json={"drug_name": "metformin"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] in {j.job_id for j in fresh_store.all()}
    assert body["status"] in {"pending", "running", "done"}
    persisted = run_ledger.runs[body["job_id"]]
    assert persisted.drug_name == "metformin"
    assert persisted.execution_mode == "live"
    assert persisted.submission_source == "api"
    assert persisted.analysis_kind == "find"
    assert persisted.disease_name is None
    assert persisted.date_before is None


def test_post_unknown_drug_returns_422_without_creating_job(
    client, fresh_store, run_ledger
):
    from indication_scout.data_sources.base_client import DataSourceError

    run = AsyncMock()
    with (
        patch(
            "indication_scout.api.routes.analyses.resolve_drug_name",
            new=AsyncMock(
                side_effect=DataSourceError("chembl", "No drug found for 'zzzqq'")
            ),
        ),
        patch("indication_scout.api.routes.analyses.run_analysis", new=run),
    ):
        resp = client.post("/api/analyses", json={"drug_name": "zzzqq"})

    assert resp.status_code == 422
    assert resp.json()["detail"] == "No drug found matching 'zzzqq'."
    assert list(fresh_store.all()) == []
    assert run_ledger.runs == {}
    run.assert_not_awaited()


def test_get_unknown_job_returns_404(client, fresh_store):
    resp = client.get("/api/analyses/nope")
    assert resp.status_code == 404


def test_report_for_unfinished_job_returns_409(client, fresh_store, run_ledger):
    job = create_persisted_job(fresh_store, run_ledger)
    run_ledger.runs[job.job_id].status = "running"

    resp = client.get(f"/api/analyses/{job.job_id}/report.md")
    assert resp.status_code == 409


def test_delete_unknown_job_returns_404(client, fresh_store):
    resp = client.delete("/api/analyses/nope")
    assert resp.status_code == 404


def test_delete_finished_job_is_idempotent_204(client, fresh_store, run_ledger):
    job = create_persisted_job(fresh_store, run_ledger)
    run_ledger.runs[job.job_id].status = "done"
    run_ledger.runs[job.job_id].result = SupervisorOutput().model_dump(mode="json")

    resp = client.delete(f"/api/analyses/{job.job_id}")
    assert resp.status_code == 204
    assert run_ledger.runs[job.job_id].status == "done"


def test_get_uses_durable_ledger_after_runtime_state_is_lost(
    client, fresh_store, run_ledger
):
    output = SupervisorOutput(drug_name="metformin", summary="Persisted.")
    run = run_ledger.create_run(
        "metformin",
        "live",
        submission_source="api",
        analysis_kind="find",
        disease_name=None,
        date_before=None,
    )
    attempt = run_ledger.start_attempt(
        run.run_id,
        worker_id=None,
        deployment_id=None,
        release=None,
    )
    run_ledger.append_event(
        run.run_id,
        "analysis.progress",
        "info",
        attempt_id=attempt.attempt_id,
        stage="summary",
        attributes={"message": "Writing summary"},
    )
    run_ledger.complete_validated_attempt(run.run_id, attempt.attempt_id, output)

    assert fresh_store.all() == []
    response = client.get(f"/api/analyses/{run.run_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["result"]["summary"] == "Persisted."
    assert response.json()["progress"] == [
        {"phase": "summary", "message": "Writing summary"}
    ]


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
    assert result["candidate_diseases"] == [
        "alcohol dependence",
        "obesity",
        "bipolar disorder",
    ]
    assert result["top_diseases"] == ["alcohol dependence", "obesity"]
    assert result["summary"].startswith("Duloxetine shows mechanism-grounded")
    assert len(result["disease_findings"]) == 2
    assert result["disease_findings"][0] == {
        "disease": "alcohol dependence",
        "source": "mechanism",
        "approval_relationship": "none",
        "literature": None,
        "clinical_trials": None,
        "blurb": None,
    }
