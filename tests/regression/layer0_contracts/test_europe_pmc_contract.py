"""Europe PMC client contract: citation counts come back keyed by PMID as recorded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from indication_scout.data_sources.europe_pmc import EuropePMCClient

pytestmark = pytest.mark.contract


async def test_fetch_citation_counts_maps_pmid_to_count(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("epmc_citations"):
        async with EuropePMCClient(cache_dir=contract_cache_dir) as client:
            counts = await client.fetch_citation_counts(["39215927", "27633186"])

    assert counts == {"39215927": 1, "27633186": 4961}
