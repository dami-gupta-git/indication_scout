"""vcrpy wiring for the regression suite.

vcrpy 7+ supports both aiohttp (data source clients) and httpx (Anthropic SDK,
LangChain) in one cassette. Mode selection is via the SCOUT_CASSETTE_MODE env
var, read once on import.

Modes:
- replay (default): play back the committed cassette. No network. Fails if a
  request isn't in the cassette.
- record: re-record the cassette from scratch. Hits real APIs. Used when the
  pipeline legitimately changes and the golden needs to move.
- live: hit real APIs without recording. Used to sanity-check that current
  code still works against the real services without touching cassettes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.regression.common.constants import (
    CASSETTE_MODE_ENV,
    CASSETTE_MODE_LIVE,
    CASSETTE_MODE_RECORD,
    CASSETTE_MODE_REPLAY,
)


def get_mode() -> str:
    mode = os.environ.get(CASSETTE_MODE_ENV, CASSETTE_MODE_REPLAY).lower()
    if mode not in {CASSETTE_MODE_REPLAY, CASSETTE_MODE_RECORD, CASSETTE_MODE_LIVE}:
        raise ValueError(
            f"{CASSETTE_MODE_ENV}={mode!r} is not one of "
            f"{{{CASSETTE_MODE_REPLAY}, {CASSETTE_MODE_RECORD}, {CASSETTE_MODE_LIVE}}}"
        )
    return mode


def _vcr_record_mode(mode: str) -> str:
    if mode == CASSETTE_MODE_REPLAY:
        # Fail loudly on unrecorded requests so cassette gaps don't slip past.
        return "none"
    if mode == CASSETTE_MODE_RECORD:
        return "all"
    # live mode: bypass entirely (handled by caller — we don't enter the
    # cassette at all).
    return "none"


# Anthropic and most data sources put their auth in headers; strip them so
# committed cassettes don't leak keys.
_SCRUB_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "set-cookie",
    "cookie",
)


def _canonical_body(body: object) -> str:
    """Normalize a request body to a comparable string.

    aiohttp hands vcrpy the *unserialized* body for GraphQL POSTs (a dict, from the `json=`
    kwarg), while a replayed cassette holds the serialized text. vcrpy's own body matcher
    assumes text and raises on the dict, so both sides are normalized here: JSON is compared
    as a key-order-independent structure, anything else as plain text.
    """
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return body
    return json.dumps(body, sort_keys=True)


def _match_body(request_a: Any, request_b: Any) -> bool:
    return _canonical_body(request_a.body) == _canonical_body(request_b.body)


@contextmanager
def use_cassette(
    cassette_path: Path, filter_query_parameters: Sequence[str] = ()
) -> Iterator[None]:
    """Wrap a block of code so all aiohttp + httpx traffic flows through vcrpy.

    In live mode this is a no-op context manager — requests hit the real APIs.

    `filter_query_parameters` names query-string keys to scrub before recording (PubMed and
    openFDA carry their API key there, not in a header). Scrubbed keys are also dropped from
    the match, so leaving the default empty keeps existing cassettes matching as recorded.
    """
    mode = get_mode()
    if mode == CASSETTE_MODE_LIVE:
        yield
        return

    if mode == CASSETTE_MODE_RECORD:
        cassette_path.parent.mkdir(parents=True, exist_ok=True)

    # Deferred import so the default test suite (which excludes -m regression)
    # collects cleanly even when vcrpy isn't installed.
    import vcr

    recorder = vcr.VCR(
        record_mode=_vcr_record_mode(mode),
        filter_headers=list(_SCRUB_HEADERS),
        filter_query_parameters=list(filter_query_parameters),
        decode_compressed_response=True,
    )
    recorder.register_matcher("json_aware_body", _match_body)
    # Match on method + URI + body so that LLM calls with different prompts
    # are treated as distinct requests during replay.
    recorder.match_on = (
        "method",
        "scheme",
        "host",
        "port",
        "path",
        "query",
        "json_aware_body",
    )
    with recorder.use_cassette(str(cassette_path)):
        yield
