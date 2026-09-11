"""Unit tests for durable analysis-run SQLAlchemy models."""

from datetime import UTC, datetime

from indication_scout.sqlalchemy.analysis_runs import (
    AnalysisAttempt,
    AnalysisEvent,
    AnalysisRun,
)


def test_analysis_run_mapping_and_fields():
    """The run model should expose the complete durable run record."""
    now = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    record = AnalysisRun(
        run_id="a" * 32,
        drug_name="metformin",
        status="done",
        execution_mode="live",
        result={"drug_name": "metformin"},
        integrity_status="passed",
        cancellation_requested_at=None,
        created_at=now,
        started_at=now,
        finished_at=now,
        updated_at=now,
    )

    assert AnalysisRun.__tablename__ == "analysis_runs"
    assert [column.name for column in AnalysisRun.__table__.primary_key] == ["run_id"]
    assert {column.name for column in AnalysisRun.__table__.columns} == {
        "run_id",
        "drug_name",
        "status",
        "execution_mode",
        "result",
        "integrity_status",
        "cancellation_requested_at",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
    }
    assert record.run_id == "a" * 32
    assert record.drug_name == "metformin"
    assert record.status == "done"
    assert record.execution_mode == "live"
    assert record.result == {"drug_name": "metformin"}
    assert record.integrity_status == "passed"
    assert record.cancellation_requested_at is None
    assert record.created_at == now
    assert record.started_at == now
    assert record.finished_at == now
    assert record.updated_at == now


def test_analysis_attempt_mapping_and_fields():
    """The attempt model should retain execution and failure metadata."""
    now = datetime(2026, 9, 10, 15, 1, tzinfo=UTC)
    record = AnalysisAttempt(
        attempt_id="b" * 32,
        run_id="a" * 32,
        attempt_number=2,
        status="error",
        worker_id="worker-1",
        deployment_id="deployment-1",
        release="release-1",
        started_at=now,
        finished_at=now,
        heartbeat_at=now,
        duration_ms=1250,
        error_code="upstream_timeout",
        error_message="The upstream request timed out.",
        retryable=True,
    )

    assert AnalysisAttempt.__tablename__ == "analysis_attempts"
    assert [column.name for column in AnalysisAttempt.__table__.primary_key] == [
        "attempt_id"
    ]
    assert {column.name for column in AnalysisAttempt.__table__.columns} == {
        "attempt_id",
        "run_id",
        "attempt_number",
        "status",
        "worker_id",
        "deployment_id",
        "release",
        "started_at",
        "finished_at",
        "heartbeat_at",
        "duration_ms",
        "error_code",
        "error_message",
        "retryable",
    }
    assert record.attempt_id == "b" * 32
    assert record.run_id == "a" * 32
    assert record.attempt_number == 2
    assert record.status == "error"
    assert record.worker_id == "worker-1"
    assert record.deployment_id == "deployment-1"
    assert record.release == "release-1"
    assert record.started_at == now
    assert record.finished_at == now
    assert record.heartbeat_at == now
    assert record.duration_ms == 1250
    assert record.error_code == "upstream_timeout"
    assert record.error_message == "The upstream request timed out."
    assert record.retryable is True


def test_analysis_event_mapping_and_fields():
    """The event model should retain the complete structured event."""
    now = datetime(2026, 9, 10, 15, 2, tzinfo=UTC)
    record = AnalysisEvent(
        event_id=7,
        run_id="a" * 32,
        attempt_id="b" * 32,
        occurred_at=now,
        event_name="dependency.request.completed",
        stage="literature",
        severity="warning",
        duration_ms=3000,
        dependency="pubmed",
        attributes={"status_code": 429, "retry": 1},
    )

    assert AnalysisEvent.__tablename__ == "analysis_events"
    assert [column.name for column in AnalysisEvent.__table__.primary_key] == [
        "event_id"
    ]
    assert {column.name for column in AnalysisEvent.__table__.columns} == {
        "event_id",
        "run_id",
        "attempt_id",
        "occurred_at",
        "event_name",
        "stage",
        "severity",
        "duration_ms",
        "dependency",
        "attributes",
    }
    assert record.event_id == 7
    assert record.run_id == "a" * 32
    assert record.attempt_id == "b" * 32
    assert record.occurred_at == now
    assert record.event_name == "dependency.request.completed"
    assert record.stage == "literature"
    assert record.severity == "warning"
    assert record.duration_ms == 3000
    assert record.dependency == "pubmed"
    assert record.attributes == {"status_code": 429, "retry": 1}
