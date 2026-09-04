"""Unit tests for EuropePMCClient.fetch_citation_counts() per-PMID caching."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.europe_pmc import EuropePMCClient


def _response(*records: tuple[str, int]) -> dict:
    return {
        "resultList": {
            "result": [{"pmid": pmid, "citedByCount": count} for pmid, count in records]
        }
    }


async def test_counts_are_cached_per_pmid(tmp_path: Path) -> None:
    """A second call for the same PMIDs hits the cache and issues no request."""
    client = EuropePMCClient(cache_dir=tmp_path)

    with patch.object(
        EuropePMCClient,
        "_rest_get",
        new=AsyncMock(return_value=_response(("111", 42), ("222", 7))),
    ) as first:
        assert await client.fetch_citation_counts(["111", "222"]) == {
            "111": 42,
            "222": 7,
        }
        assert first.await_count == 1

    with patch.object(EuropePMCClient, "_rest_get", new=AsyncMock()) as second:
        assert await client.fetch_citation_counts(["111", "222"]) == {
            "111": 42,
            "222": 7,
        }
        assert second.await_count == 0


async def test_only_uncached_pmids_are_requested(tmp_path: Path) -> None:
    """PMIDs Europe PMC has no record for are cached too, so only new PMIDs are queried."""
    client = EuropePMCClient(cache_dir=tmp_path)

    with patch.object(
        EuropePMCClient, "_rest_get", new=AsyncMock(return_value=_response(("111", 42)))
    ):
        assert await client.fetch_citation_counts(["111", "999"]) == {"111": 42}

    mock = AsyncMock(return_value=_response(("333", 5)))
    with patch.object(EuropePMCClient, "_rest_get", new=mock):
        assert await client.fetch_citation_counts(["111", "999", "333"]) == {
            "111": 42,
            "333": 5,
        }
        assert mock.await_count == 1
        assert mock.await_args.args[1]["query"] == "(EXT_ID:333) AND SRC:MED"


async def test_failed_batch_is_not_cached(tmp_path: Path) -> None:
    """A timed-out batch caches nothing, so the next call retries those PMIDs."""
    client = EuropePMCClient(cache_dir=tmp_path)

    with patch.object(
        EuropePMCClient,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("europepmc", "timeout")),
    ):
        assert await client.fetch_citation_counts(["111"]) == {}

    mock = AsyncMock(return_value=_response(("111", 42)))
    with patch.object(EuropePMCClient, "_rest_get", new=mock):
        assert await client.fetch_citation_counts(["111"]) == {"111": 42}
        assert mock.await_count == 1
