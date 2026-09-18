"""openFDA client contract: label indications and safety sections parse as recorded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from indication_scout.data_sources.fda import FDAClient

pytestmark = pytest.mark.contract


async def test_get_label_indications_returns_one_string_per_label(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("fda_labels"):
        async with FDAClient(cache_dir=contract_cache_dir) as client:
            indications = await client.get_label_indications("pioglitazone")

    assert len(indications) == 5
    assert all(
        "INDICATIONS AND USAGE" in text and "type 2 diabetes mellitus" in text
        for text in indications
    )


async def test_get_label_safety_parses_boxed_warnings(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("fda_labels"):
        async with FDAClient(cache_dir=contract_cache_dir) as client:
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
