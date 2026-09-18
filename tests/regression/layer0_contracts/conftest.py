"""Fixtures for the data-source contract layer.

Every test runs against a committed cassette and an empty cache directory, so the call under
test always goes through the client's own parse path rather than being served from a warm
file cache.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.regression.common.cassette import use_cassette
from tests.regression.common.constants import CONTRACT_CASSETTE_DIR

# PubMed and openFDA pass their API key as a query parameter, so it would otherwise be
# recorded into a committed cassette.
SCRUBBED_QUERY_PARAMS = ("api_key",)


@pytest.fixture
def contract_cache_dir(tmp_path: Path) -> Path:
    """An empty per-test cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def cassette() -> Callable[[str], Iterator[None]]:
    """Return a context manager that plays back `cassettes/<name>.yaml`."""

    @contextmanager
    def _cassette(name: str) -> Iterator[None]:
        path = CONTRACT_CASSETTE_DIR / f"{name}.yaml"
        with use_cassette(path, filter_query_parameters=SCRUBBED_QUERY_PARAMS):
            yield

    return _cassette
