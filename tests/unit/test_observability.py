"""Tests for structured logging and correlation context."""

import asyncio
import json
import logging
from datetime import UTC, datetime

import pytest

from indication_scout.constants import NOISY_THIRD_PARTY_LOGGERS
from indication_scout.observability import (
    JsonLogFormatter,
    bind_log_context,
    configure_logging,
    current_log_context,
    reset_log_context,
)


def test_json_formatter_includes_record_and_bound_context():
    record = logging.LogRecord(
        name="indication_scout.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Analysis started",
        args=(),
        exc_info=None,
    )
    record.created = 1_700_000_000.0
    record.event_name = "analysis.started"
    token = bind_log_context(run_id="run-1", attempt_id="attempt-1")
    try:
        payload = json.loads(JsonLogFormatter().format(record))
    finally:
        reset_log_context(token)

    assert payload == {
        "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
        "severity": "info",
        "logger": "indication_scout.test",
        "service": "indication-scout",
        "message": "Analysis started",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "event_name": "analysis.started",
    }


async def test_log_context_is_isolated_between_async_tasks():
    async def read_context(run_id: str) -> dict[str, str]:
        token = bind_log_context(run_id=run_id)
        try:
            await asyncio.sleep(0)
            return current_log_context()
        finally:
            reset_log_context(token)

    first, second = await asyncio.gather(read_context("run-1"), read_context("run-2"))

    assert first == {"run_id": "run-1"}
    assert second == {"run_id": "run-2"}
    assert current_log_context() == {}


def test_configure_logging_rejects_unknown_level():
    with pytest.raises(ValueError, match="Unknown logging level: LOUD"):
        configure_logging("LOUD")


def test_configure_logging_suppresses_successful_third_party_requests():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_root_level = root.level
    original_levels = {
        name: logging.getLogger(name).level for name in NOISY_THIRD_PARTY_LOGGERS
    }
    try:
        configure_logging(logging.INFO)

        assert {
            name: logging.getLogger(name).getEffectiveLevel()
            for name in NOISY_THIRD_PARTY_LOGGERS
        } == dict.fromkeys(NOISY_THIRD_PARTY_LOGGERS, logging.WARNING)
    finally:
        root.handlers.clear()
        root.handlers.extend(original_handlers)
        root.setLevel(original_root_level)
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)
