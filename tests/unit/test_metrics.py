"""Tests for bounded Prometheus application metrics."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from indication_scout.metrics import (
    ACTIVE_ANALYSES,
    ANALYSIS_DURATION,
    ANALYSIS_RUNS,
    DEPENDENCY_REQUESTS,
    HTTP_REQUESTS,
    INTEGRITY_REJECTIONS,
    OLDEST_ACTIVE_ANALYSIS_START_TIME,
    analysis_finished,
    analysis_started,
    record_dependency_request,
    record_http_request,
)


def test_analysis_duration_has_service_level_buckets():
    ANALYSIS_DURATION.labels("api", "live", "done").observe(601.0)

    bucket_bounds = [
        sample.labels["le"]
        for sample in ANALYSIS_DURATION.collect()[0].samples
        if sample.name == "indication_scout_analysis_duration_seconds_bucket"
        and sample.labels["submission_source"] == "api"
        and sample.labels["execution_mode"] == "live"
        and sample.labels["outcome"] == "done"
    ]

    assert bucket_bounds == [
        "30.0",
        "60.0",
        "120.0",
        "300.0",
        "600.0",
        "900.0",
        "1800.0",
        "3600.0",
        "+Inf",
    ]


def test_http_metric_records_route_template_and_status_class():
    counter = HTTP_REQUESTS.labels("GET", "/api/analyses/{job_id}", "2xx")
    before = counter._value.get()

    record_http_request("GET", "/api/analyses/{job_id}", 200, 0.25)

    assert counter._value.get() == before + 1


def test_dependency_metric_bounds_unknown_labels():
    counter = DEPENDENCY_REQUESTS.labels("other", "OTHER", "other")
    before = counter._value.get()

    record_dependency_request("user-value", "CONNECT", "new-outcome", 0.5)

    assert counter._value.get() == before + 1


def test_analysis_metrics_track_active_and_terminal_attempts():
    active = ACTIVE_ANALYSES.labels("api", "live")
    completed = ANALYSIS_RUNS.labels("api", "live", "done")
    active_before = active._value.get()
    completed_before = completed._value.get()

    started_at = analysis_started("api", "live")
    assert active._value.get() == active_before + 1
    assert (
        OLDEST_ACTIVE_ANALYSIS_START_TIME.labels("api", "live")._value.get()
        == started_at
    )

    analysis_finished("api", "live", "done", 2.0, started_at=started_at)

    assert active._value.get() == active_before
    assert completed._value.get() == completed_before + 1


def test_analysis_metrics_keep_oldest_concurrent_start_and_record_integrity():
    integrity = INTEGRITY_REJECTIONS.labels("api", "live")
    integrity_before = integrity._value.get()

    first_started_at = analysis_started("api", "live")
    second_started_at = analysis_started("api", "live")
    analysis_finished(
        "api",
        "live",
        "error",
        2.0,
        started_at=first_started_at,
        integrity_failed=True,
    )

    assert (
        OLDEST_ACTIVE_ANALYSIS_START_TIME.labels("api", "live")._value.get()
        == second_started_at
    )
    assert integrity._value.get() == integrity_before + 1

    analysis_finished("api", "live", "cancelled", 3.0, started_at=second_started_at)


def test_api_exposes_metrics_and_returns_request_id():
    from indication_scout.api.main import app

    unmatched = HTTP_REQUESTS.labels("GET", "unmatched", "2xx")
    unmatched_before = unmatched._value.get()
    with patch(
        "indication_scout.api.main._geolocate",
        new=AsyncMock(return_value=("New York, New York, United States", False)),
    ):
        with TestClient(app) as client:
            health = client.get("/health", headers={"x-request-id": "request-1"})
            metrics = client.get("/metrics/")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "version": "0.1.0"}
    assert health.headers["x-request-id"] == "request-1"
    assert metrics.status_code == 200
    assert "indication_scout_http_requests_total" in metrics.text
    assert unmatched._value.get() == unmatched_before
