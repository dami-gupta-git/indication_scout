"""Bounded-label Prometheus metrics for application operations."""

from __future__ import annotations

import time
from threading import Lock

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "indication_scout_http_requests_total",
    "HTTP requests completed by the API.",
    ("method", "route", "status_class"),
)
HTTP_DURATION = Histogram(
    "indication_scout_http_request_duration_seconds",
    "API request duration in seconds.",
    ("method", "route"),
)
ANALYSIS_RUNS = Counter(
    "indication_scout_analysis_runs_total",
    "Analysis attempts reaching a terminal outcome.",
    ("submission_source", "execution_mode", "outcome"),
)
ANALYSIS_DURATION = Histogram(
    "indication_scout_analysis_duration_seconds",
    "Analysis attempt duration in seconds.",
    ("submission_source", "execution_mode", "outcome"),
)
ACTIVE_ANALYSES = Gauge(
    "indication_scout_active_analyses",
    "Analysis attempts currently executing.",
    ("submission_source", "execution_mode"),
)
OLDEST_ACTIVE_ANALYSIS_START_TIME = Gauge(
    "indication_scout_analysis_oldest_active_start_time_seconds",
    "Unix timestamp when the oldest currently active analysis attempt started.",
    ("submission_source", "execution_mode"),
)
INTEGRITY_REJECTIONS = Counter(
    "indication_scout_report_integrity_rejections_total",
    "Reports rejected before publication.",
    ("submission_source", "execution_mode"),
)
DEPENDENCY_REQUESTS = Counter(
    "indication_scout_dependency_requests_total",
    "External dependency request attempts.",
    ("dependency", "method", "outcome"),
)
DEPENDENCY_DURATION = Histogram(
    "indication_scout_dependency_request_duration_seconds",
    "External dependency request duration in seconds.",
    ("dependency", "method", "outcome"),
)

_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}
_SUBMISSION_SOURCES = {"api", "cli"}
_EXECUTION_MODES = {"live", "seed"}
_ANALYSIS_OUTCOMES = {"done", "error", "cancelled", "interrupted"}
_DEPENDENCIES = {
    "chembl",
    "clinical_trials",
    "drugbank",
    "europepmc",
    "open_targets",
    "openfda",
    "pubmed",
}
_DEPENDENCY_OUTCOMES = {
    "success",
    "retryable_status",
    "error_status",
    "timeout",
    "connection_error",
    "other",
}
_ACTIVE_ANALYSIS_START_TIMES: dict[tuple[str, str], list[float]] = {}
_ACTIVE_ANALYSIS_LOCK = Lock()


def _method(value: str) -> str:
    normalized = value.upper()
    return normalized if normalized in _HTTP_METHODS else "OTHER"


def _dependency(value: str) -> str:
    return value if value in _DEPENDENCIES else "other"


def _bounded(value: str, allowed: set[str]) -> str:
    return value if value in allowed else "other"


def record_http_request(
    method: str, route: str, status_code: int, duration_seconds: float
) -> None:
    """Record one API request using a route template rather than a raw path."""
    bounded_method = _method(method)
    bounded_route = route if route.startswith("/") else "unmatched"
    status_class = f"{status_code // 100}xx"
    HTTP_REQUESTS.labels(bounded_method, bounded_route, status_class).inc()
    HTTP_DURATION.labels(bounded_method, bounded_route).observe(duration_seconds)


def analysis_started(submission_source: str, execution_mode: str) -> float:
    """Record an active attempt and return its metric start-time token."""
    labels = (
        _bounded(submission_source, _SUBMISSION_SOURCES),
        _bounded(execution_mode, _EXECUTION_MODES),
    )
    started_at = time.time()
    with _ACTIVE_ANALYSIS_LOCK:
        start_times = _ACTIVE_ANALYSIS_START_TIMES.setdefault(labels, [])
        start_times.append(started_at)
        ACTIVE_ANALYSES.labels(*labels).inc()
        OLDEST_ACTIVE_ANALYSIS_START_TIME.labels(*labels).set(min(start_times))
    return started_at


def analysis_finished(
    submission_source: str,
    execution_mode: str,
    outcome: str,
    duration_seconds: float,
    *,
    started_at: float,
    integrity_failed: bool = False,
) -> None:
    """Record one terminal attempt and decrement its active gauge."""
    source_label = _bounded(submission_source, _SUBMISSION_SOURCES)
    mode_label = _bounded(execution_mode, _EXECUTION_MODES)
    outcome_label = _bounded(outcome, _ANALYSIS_OUTCOMES)
    labels = (source_label, mode_label)
    with _ACTIVE_ANALYSIS_LOCK:
        start_times = _ACTIVE_ANALYSIS_START_TIMES[labels]
        start_times.remove(started_at)
        ACTIVE_ANALYSES.labels(*labels).dec()
        if start_times:
            OLDEST_ACTIVE_ANALYSIS_START_TIME.labels(*labels).set(min(start_times))
        else:
            del _ACTIVE_ANALYSIS_START_TIMES[labels]
            OLDEST_ACTIVE_ANALYSIS_START_TIME.remove(*labels)
    ANALYSIS_RUNS.labels(source_label, mode_label, outcome_label).inc()
    ANALYSIS_DURATION.labels(source_label, mode_label, outcome_label).observe(
        duration_seconds
    )
    if integrity_failed:
        INTEGRITY_REJECTIONS.labels(source_label, mode_label).inc()


def record_dependency_request(
    dependency: str, method: str, outcome: str, duration_seconds: float
) -> None:
    """Record one external request attempt with bounded labels."""
    bounded_outcome = outcome if outcome in _DEPENDENCY_OUTCOMES else "other"
    labels = (_dependency(dependency), _method(method), bounded_outcome)
    DEPENDENCY_REQUESTS.labels(*labels).inc()
    DEPENDENCY_DURATION.labels(*labels).observe(duration_seconds)
