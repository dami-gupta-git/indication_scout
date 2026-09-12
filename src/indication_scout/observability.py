"""Structured logging configuration and asynchronous correlation context."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from indication_scout.constants import NOISY_THIRD_PARTY_LOGGERS

_SERVICE_NAME = "indication-scout"
_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)
_EXTRA_FIELDS = (
    "event_name",
    "duration_seconds",
    "http_method",
    "http_route",
    "http_status_code",
    "client_ip",
    "client_location",
    "client_is_automated",
    "outcome",
    "retry_count",
    "dependency",
)


def bind_log_context(**fields: Any) -> Token[dict[str, Any] | None]:
    """Add non-null fields to the current asynchronous logging context."""
    merged = dict(_context.get() or {})
    merged.update({key: value for key, value in fields.items() if value is not None})
    return _context.set(merged)


def reset_log_context(token: Token[dict[str, Any] | None]) -> None:
    """Restore the logging context that preceded one bind operation."""
    _context.reset(token)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind logging fields for the duration of a synchronous or asynchronous call."""
    token = bind_log_context(**fields)
    try:
        yield
    finally:
        reset_log_context(token)


def current_log_context() -> dict[str, Any]:
    """Return a copy of the current logging context."""
    return dict(_context.get() or {})


def _trace_id() -> str | None:
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return format(span_context.trace_id, "032x")
    except Exception:  # noqa: BLE001 - logging must not affect application behavior
        return None
    return None


class JsonLogFormatter(logging.Formatter):
    """Render one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname.lower(),
            "logger": record.name,
            "service": _SERVICE_NAME,
            "message": record.getMessage(),
        }
        payload.update(_context.get() or {})
        payload.update(
            {
                "environment": os.environ.get("RAILWAY_ENVIRONMENT_NAME"),
                "release": os.environ.get("RAILWAY_GIT_COMMIT_SHA"),
                "deployment_id": os.environ.get("RAILWAY_DEPLOYMENT_ID"),
                "replica_id": os.environ.get("RAILWAY_REPLICA_ID"),
                "trace_id": _trace_id(),
            }
        )
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            error = record.exc_info[1]
            payload["exception_type"] = type(error).__name__ if error else None
            payload["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(
            {key: value for key, value in payload.items() if value is not None},
            separators=(",", ":"),
            default=str,
        )


def configure_logging(level: str | int) -> None:
    """Configure application and Uvicorn loggers to write single-line JSON."""
    if isinstance(level, int):
        resolved_level = level
    else:
        configured_level = logging.getLevelNamesMapping().get(level.upper())
        if configured_level is None:
            raise ValueError(f"Unknown logging level: {level}")
        resolved_level = configured_level
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    for name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
