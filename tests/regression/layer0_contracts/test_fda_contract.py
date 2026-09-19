"""openFDA client contract: label indications and safety sections parse as recorded."""

from __future__ import annotations

import pytest

from indication_scout.data_sources.fda import FDAClient
from tests.regression.layer0_contracts.conftest import ContractClient

pytestmark = pytest.mark.contract


async def test_get_label_indications_returns_one_string_per_label(
    contract_client: ContractClient,
) -> None:
    async with contract_client(FDAClient, "fda_labels") as client:
        indications = await client.get_label_indications("pioglitazone")

    assert len(indications) == 5
    assert all(
        "INDICATIONS AND USAGE" in text and "type 2 diabetes mellitus" in text
        for text in indications
    )


async def test_get_label_safety_parses_boxed_warnings(
    contract_client: ContractClient,
) -> None:
    async with contract_client(FDAClient, "fda_labels") as client:
        records = await client.get_label_safety("pioglitazone")

    assert len(records) == 5
    record = records[0]
    assert record.set_id == "0331400c-b163-4856-bfb0-965470247cb3"
    assert record.effective_time == "20250517"
    assert record.brand_names == ["PIOGLITAZONE HYDROCHLORIDE"]
    assert record.generic_names == ["PIOGLITAZONE HYDROCHLORIDE"]
    assert len(record.boxed_warnings) == 1
    assert record.boxed_warnings[0].startswith("WARNING: CONGESTIVE HEART FAILURE")
    assert len(record.warnings) == 1
