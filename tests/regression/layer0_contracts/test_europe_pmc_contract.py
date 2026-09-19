"""Europe PMC client contract: citation counts come back keyed by PMID as recorded."""

from __future__ import annotations

import pytest

from indication_scout.data_sources.europe_pmc import EuropePMCClient
from tests.regression.layer0_contracts.conftest import ContractClient

pytestmark = pytest.mark.contract


async def test_fetch_citation_counts_maps_pmid_to_count(
    contract_client: ContractClient,
) -> None:
    async with contract_client(EuropePMCClient, "epmc_citations") as client:
        counts = await client.fetch_citation_counts(["39215927", "27633186"])

    assert counts == {"39215927": 1, "27633186": 4961}
