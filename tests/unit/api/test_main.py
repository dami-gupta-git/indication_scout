"""Tests for API logging behavior."""

import logging

import pytest

from indication_scout.api.main import _UvicornAccessLogFilter


@pytest.mark.parametrize(
    "message",
    [
        '127.0.0.1:1000 - "GET /metrics/ HTTP/1.1" 200',
        '127.0.0.1:1000 - "GET /api/analyses/run-1 HTTP/1.1" 200',
        '127.0.0.1:1000 - "POST /api/analyses HTTP/1.1" 202',
    ],
)
def test_access_log_filter(message: str) -> None:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )

    assert _UvicornAccessLogFilter().filter(record) is False
