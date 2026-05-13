"""Unit tests for FDAClient — no network calls."""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.fda import FDAClient

WEGOVY_LABEL_FIXTURE = {
    "results": [
        {
            "indications_and_usage": [
                "Wegovy is indicated as an adjunct to a reduced calorie diet and increased physical activity for chronic weight management."
            ]
        }
    ]
}

OZEMPIC_LABEL_FIXTURE = {
    "results": [
        {
            "indications_and_usage": [
                "Ozempic is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus."
            ]
        }
    ]
}

MULTI_INDICATION_FIXTURE = {
    "results": [
        {"indications_and_usage": ["Indicated for condition A."]},
        {"indications_and_usage": ["Indicated for condition B."]},
    ]
}


# --- get_label_indications ---


async def test_get_label_indications_returns_indications_text(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    with patch.object(
        client, "_rest_get", new=AsyncMock(return_value=WEGOVY_LABEL_FIXTURE)
    ):
        result = await client.get_label_indications("Wegovy")

    assert len(result) == 1
    assert "chronic weight management" in result[0]


async def test_get_label_indications_returns_empty_on_404(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("openfda", "HTTP 404", 404)),
    ):
        result = await client.get_label_indications("NonexistentBrand")

    assert result == []


async def test_get_label_indications_reraises_non_404(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("openfda", "HTTP 500", 500)),
    ):
        with pytest.raises(DataSourceError):
            await client.get_label_indications("Wegovy")


async def test_get_label_indications_empty_results(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    with patch.object(client, "_rest_get", new=AsyncMock(return_value={"results": []})):
        result = await client.get_label_indications("SomeBrand")

    assert result == []


async def test_get_label_indications_no_indications_field(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    fixture = {"results": [{"openfda": {"brand_name": ["Wegovy"]}}]}
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_label_indications("Wegovy")

    assert result == []


async def test_get_label_indications_multiple_results(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    with patch.object(
        client, "_rest_get", new=AsyncMock(return_value=MULTI_INDICATION_FIXTURE)
    ):
        result = await client.get_label_indications("SomeDrug")

    assert len(result) == 2
    assert "condition A" in result[0]
    assert "condition B" in result[1]


async def test_get_label_indications_caches_result(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    mock_rest_get = AsyncMock(return_value=WEGOVY_LABEL_FIXTURE)
    with patch.object(client, "_rest_get", new=mock_rest_get):
        first = await client.get_label_indications("Wegovy")
        second = await client.get_label_indications("Wegovy")

    assert first == second
    mock_rest_get.assert_awaited_once()


async def test_get_label_indications_does_not_cache_404(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    mock_rest_get = AsyncMock(side_effect=DataSourceError("openfda", "HTTP 404", 404))
    with patch.object(client, "_rest_get", new=mock_rest_get):
        first = await client.get_label_indications("Ghost")
        second = await client.get_label_indications("Ghost")

    assert first == []
    assert second == []
    assert mock_rest_get.await_count == 2


# --- get_all_label_indications ---


async def test_get_all_label_indications_combines_trade_names(tmp_path):
    client = FDAClient(cache_dir=tmp_path)

    async def mock_get_label(brand_name: str) -> list[str]:
        mapping = {
            "ozempic": ["Ozempic indication text."],
            "wegovy": ["Wegovy indication text."],
        }
        return mapping.get(brand_name.lower(), [])

    with patch.object(client, "get_label_indications", side_effect=mock_get_label):
        result = await client.get_all_label_indications(["Ozempic", "Wegovy"])

    assert len(result) == 2
    assert "Ozempic indication text." in result
    assert "Wegovy indication text." in result


async def test_get_all_label_indications_deduplicates(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    shared_text = "Same indication text across brands."

    with patch.object(
        client,
        "get_label_indications",
        new=AsyncMock(return_value=[shared_text]),
    ):
        result = await client.get_all_label_indications(["BrandA", "BrandB"])

    assert len(result) == 1
    assert result[0] == shared_text


async def test_get_all_label_indications_empty_trade_names(tmp_path):
    client = FDAClient(cache_dir=tmp_path)
    result = await client.get_all_label_indications([])

    assert result == []


async def test_get_all_label_indications_tolerates_404(tmp_path):
    """Per-alias 404s are turned into [] inside get_label_indications and
    must not cause get_all_label_indications to fail."""
    client = FDAClient(cache_dir=tmp_path)

    async def mock_get_label(brand_name: str) -> list[str]:
        # 404 is already converted to [] inside get_label_indications, so we
        # mirror that here: missing alias yields [] without raising.
        if brand_name == "GhostBrand":
            return []
        return ["Good indication text."]

    with patch.object(client, "get_label_indications", side_effect=mock_get_label):
        result = await client.get_all_label_indications(["GhostBrand", "GoodBrand"])

    assert len(result) == 1
    assert result[0] == "Good indication text."


async def test_get_all_label_indications_reraises_non_404_failures(tmp_path):
    """Non-404 errors (e.g. HTTP 500, 429) on any alias must be re-raised so
    the caller knows the result is incomplete."""
    client = FDAClient(cache_dir=tmp_path)

    async def mock_get_label(brand_name: str) -> list[str]:
        if brand_name == "BadBrand":
            raise DataSourceError("openfda", "HTTP 500", 500)
        return ["Good indication text."]

    with patch.object(client, "get_label_indications", side_effect=mock_get_label):
        with pytest.raises(DataSourceError):
            await client.get_all_label_indications(["BadBrand", "GoodBrand"])
