"""Database integration tests for durable analysis-run persistence."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.services.run_repository import (
    AnalysisRunRepository,
    RunStateError,
)


def test_create_run_persists_initial_state_and_event(run_db_session):
    now = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    repository = AnalysisRunRepository(run_db_session)

    with patch("indication_scout.services.run_repository._utcnow", return_value=now):
        created = repository.create_run("metformin", "live", run_id="a" * 32)

    persisted = repository.get_run(created.run_id)
    events = repository.list_events(created.run_id)
    attempts = repository.list_attempts(created.run_id)
    assert persisted is not None
    assert persisted.run_id == "a" * 32
    assert persisted.drug_name == "metformin"
    assert persisted.status == "pending"
    assert persisted.execution_mode == "live"
    assert persisted.result is None
    assert persisted.integrity_status is None
    assert persisted.cancellation_requested_at is None
    assert persisted.created_at == now
    assert persisted.started_at is None
    assert persisted.finished_at is None
    assert persisted.updated_at == now
    assert attempts == []
    assert len(events) == 1
    event = events[0]
    assert event.event_id == 1
    assert event.run_id == persisted.run_id
    assert event.attempt_id is None
    assert event.occurred_at == now
    assert event.event_name == "analysis.created"
    assert event.stage is None
    assert event.severity == "info"
    assert event.duration_ms is None
    assert event.dependency is None
    assert event.attributes is None


def test_successful_attempt_persists_progress_heartbeat_and_result(run_db_session):
    times = [datetime(2026, 9, 10, 15, minute, tzinfo=UTC) for minute in range(5)]
    repository = AnalysisRunRepository(run_db_session)
    result = SupervisorOutput(
        drug_name="metformin",
        candidate_diseases=["polycystic ovary syndrome"],
        top_diseases=["polycystic ovary syndrome"],
        summary="Stored report.",
    )

    with patch("indication_scout.services.run_repository._utcnow", side_effect=times):
        run = repository.create_run("metformin", "live", run_id="a" * 32)
        attempt = repository.start_attempt(
            run.run_id,
            worker_id="worker-1",
            deployment_id="deployment-1",
            release="release-1",
            attempt_id="b" * 32,
        )
        progress = repository.append_event(
            run.run_id,
            "analysis.stage.completed",
            "info",
            attempt_id=attempt.attempt_id,
            stage="literature",
            duration_ms=60000,
            dependency="pubmed",
            attributes={"papers": 15},
        )
        heartbeat = repository.heartbeat(run.run_id, attempt.attempt_id)
        heartbeat_at = heartbeat.heartbeat_at
        completed = repository.complete_validated_attempt(
            run.run_id, attempt.attempt_id, result
        )

    attempts = repository.list_attempts(run.run_id)
    events = repository.list_events(run.run_id)
    assert completed.run_id == run.run_id
    assert completed.drug_name == "metformin"
    assert completed.status == "done"
    assert completed.execution_mode == "live"
    assert completed.result == result.model_dump(mode="json")
    assert completed.integrity_status == "passed"
    assert completed.cancellation_requested_at is None
    assert completed.created_at == times[0]
    assert completed.started_at == times[1]
    assert completed.finished_at == times[4]
    assert completed.updated_at == times[4]
    assert heartbeat_at == times[3]
    assert len(attempts) == 1
    persisted_attempt = attempts[0]
    assert persisted_attempt.attempt_id == "b" * 32
    assert persisted_attempt.run_id == run.run_id
    assert persisted_attempt.attempt_number == 1
    assert persisted_attempt.status == "done"
    assert persisted_attempt.worker_id == "worker-1"
    assert persisted_attempt.deployment_id == "deployment-1"
    assert persisted_attempt.release == "release-1"
    assert persisted_attempt.started_at == times[1]
    assert persisted_attempt.finished_at == times[4]
    assert persisted_attempt.heartbeat_at == times[4]
    assert persisted_attempt.duration_ms == int(
        (times[4] - times[1]).total_seconds() * 1000
    )
    assert persisted_attempt.error_code is None
    assert persisted_attempt.error_message is None
    assert persisted_attempt.retryable is None
    assert progress.run_id == run.run_id
    assert progress.attempt_id == attempt.attempt_id
    assert progress.occurred_at == times[2]
    assert progress.event_name == "analysis.stage.completed"
    assert progress.stage == "literature"
    assert progress.severity == "info"
    assert progress.duration_ms == 60000
    assert progress.dependency == "pubmed"
    assert progress.attributes == {"papers": 15}
    assert [event.event_name for event in events] == [
        "analysis.created",
        "analysis.started",
        "analysis.stage.completed",
        "analysis.completed",
    ]


def test_failure_is_terminal_and_preserves_classification(run_db_session):
    start = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    failure = start + timedelta(seconds=90)
    repository = AnalysisRunRepository(run_db_session)

    with patch(
        "indication_scout.services.run_repository._utcnow",
        side_effect=[start, start, failure],
    ):
        run = repository.create_run("metformin", "live", run_id="a" * 32)
        attempt = repository.start_attempt(
            run.run_id,
            worker_id=None,
            deployment_id=None,
            release=None,
            attempt_id="b" * 32,
        )
        failed = repository.fail_attempt(
            run.run_id,
            attempt.attempt_id,
            error_code="upstream_timeout",
            error_message="PubMed request timed out.",
            retryable=True,
            integrity_failed=False,
        )

    persisted_attempt = repository.list_attempts(run.run_id)[0]
    events = repository.list_events(run.run_id)
    assert failed.run_id == run.run_id
    assert failed.drug_name == "metformin"
    assert failed.status == "error"
    assert failed.execution_mode == "live"
    assert failed.result is None
    assert failed.integrity_status is None
    assert failed.cancellation_requested_at is None
    assert failed.created_at == start
    assert failed.started_at == start
    assert failed.finished_at == failure
    assert failed.updated_at == failure
    assert persisted_attempt.attempt_id == attempt.attempt_id
    assert persisted_attempt.run_id == run.run_id
    assert persisted_attempt.attempt_number == 1
    assert persisted_attempt.status == "error"
    assert persisted_attempt.worker_id is None
    assert persisted_attempt.deployment_id is None
    assert persisted_attempt.release is None
    assert persisted_attempt.started_at == start
    assert persisted_attempt.finished_at == failure
    assert persisted_attempt.heartbeat_at == failure
    assert persisted_attempt.duration_ms == 90000
    assert persisted_attempt.error_code == "upstream_timeout"
    assert persisted_attempt.error_message == "PubMed request timed out."
    assert persisted_attempt.retryable is True
    assert events[-1].event_name == "analysis.failed"
    assert events[-1].severity == "error"
    assert events[-1].attributes == {
        "error_code": "upstream_timeout",
        "retryable": True,
    }
    with pytest.raises(RunStateError, match="is error"):
        repository.start_attempt(
            run.run_id,
            worker_id=None,
            deployment_id=None,
            release=None,
        )


def test_cancellation_request_is_idempotent_and_cancel_is_terminal(run_db_session):
    times = [datetime(2026, 9, 10, 15, minute, tzinfo=UTC) for minute in range(5)]
    repository = AnalysisRunRepository(run_db_session)

    with patch("indication_scout.services.run_repository._utcnow", side_effect=times):
        run = repository.create_run("metformin", "seed", run_id="a" * 32)
        attempt = repository.start_attempt(
            run.run_id,
            worker_id="worker-1",
            deployment_id="deployment-1",
            release="release-1",
            attempt_id="b" * 32,
        )
        requested = repository.request_cancellation(run.run_id)
        first_requested_at = requested.cancellation_requested_at
        requested_again = repository.request_cancellation(run.run_id)
        cancelled = repository.cancel_attempt(run.run_id, attempt.attempt_id)

    persisted_attempt = repository.list_attempts(run.run_id)[0]
    events = repository.list_events(run.run_id)
    assert first_requested_at == times[2]
    assert requested_again.cancellation_requested_at == times[2]
    assert cancelled.run_id == run.run_id
    assert cancelled.drug_name == "metformin"
    assert cancelled.status == "cancelled"
    assert cancelled.execution_mode == "seed"
    assert cancelled.result is None
    assert cancelled.integrity_status is None
    assert cancelled.cancellation_requested_at == times[2]
    assert cancelled.created_at == times[0]
    assert cancelled.started_at == times[1]
    assert cancelled.finished_at == times[4]
    assert cancelled.updated_at == times[4]
    assert persisted_attempt.attempt_id == attempt.attempt_id
    assert persisted_attempt.run_id == run.run_id
    assert persisted_attempt.attempt_number == 1
    assert persisted_attempt.status == "cancelled"
    assert persisted_attempt.worker_id == "worker-1"
    assert persisted_attempt.deployment_id == "deployment-1"
    assert persisted_attempt.release == "release-1"
    assert persisted_attempt.started_at == times[1]
    assert persisted_attempt.finished_at == times[4]
    assert persisted_attempt.heartbeat_at == times[4]
    assert persisted_attempt.duration_ms == int(
        (times[4] - times[1]).total_seconds() * 1000
    )
    assert persisted_attempt.error_code is None
    assert persisted_attempt.error_message is None
    assert persisted_attempt.retryable is None
    assert [event.event_name for event in events] == [
        "analysis.created",
        "analysis.started",
        "analysis.cancellation_requested",
        "analysis.cancelled",
    ]
