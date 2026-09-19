"""Fixtures for the layer-0 contract tests.

Every test runs against a committed cassette and an empty cache directory, so the call under
test always reaches the recorded HTTP interaction rather than being short-circuited by the
file cache.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Protocol, TypeVar

import pytest

from indication_scout.data_sources.base_client import BaseClient
from tests.regression.common.cassette import SCRUB_QUERY_PARAMS, use_cassette
from tests.regression.common.constants import CONTRACT_CASSETTE_DIR

ClientT = TypeVar("ClientT", bound=BaseClient)


class ContractClient(Protocol):
    """Enter `cassettes/<name>.yaml` and yield a client wired to an empty cache dir."""

    def __call__(
        self, client_cls: type[ClientT], cassette_name: str
    ) -> AbstractAsyncContextManager[ClientT]: ...


@pytest.fixture
def contract_client(tmp_path: Path) -> ContractClient:
    @asynccontextmanager
    async def _client(
        client_cls: type[ClientT], cassette_name: str
    ) -> AsyncIterator[ClientT]:
        path = CONTRACT_CASSETTE_DIR / f"{cassette_name}.yaml"
        with use_cassette(path, filter_query_parameters=SCRUB_QUERY_PARAMS):
            async with client_cls(cache_dir=tmp_path / "cache") as client:
                yield client

    return _client
