"""Integration tests for RetrievalService.fetch_and_cache date_before filtering.

These assert the date_before CONTRACT as a property of the returned PMIDs' publication
dates, not against a fixed PMID list. PubMed sorts most-recent-first and fetch_and_cache
caps at pubmed_max_results, so any hardcoded old PMID eventually drops out of the window
as new papers are indexed — the dates, however, always obey the cutoff. We read each
PMID's pub_date from pgvector via the SAME production helpers fetch_and_cache uses
(_read_pub_dates_from_db + _parse_pub_date_conservative), so the test tracks production's
own date semantics (partial dates resolved to the last day of their range).
"""

import logging
from datetime import date

from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Query with known results verified live. "biguanides AND colon cancer" returns
# PMIDs spanning multiple decades, so a cutoff date will meaningfully reduce results.
_QUERY = "biguanides AND colon cancer"

_CUTOFF = date(2015, 1, 1)


def _parsed_dates_for(svc: RetrievalService, pmids: list[str], db) -> dict[str, date]:
    """Return {pmid -> parsed publication date} for every returned PMID that has an
    abstract row in pgvector. PMIDs without an abstract (letters/editorials) or with an
    unparseable pub_date are omitted — callers assert the resulting set is non-empty so
    the date checks aren't vacuously true."""
    raw_dates = svc._read_pub_dates_from_db(pmids, db)
    parsed: dict[str, date] = {}
    for pmid, raw in raw_dates.items():
        d = svc._parse_pub_date_conservative(raw)
        if d is not None:
            parsed[pmid] = d
    return parsed


async def test_fetch_and_cache_date_before_excludes_recent_pmids(
    db_session_truncating, test_cache_dir
):
    """Every PMID returned with date_before is published strictly before the cutoff.

    _parse_pub_date_conservative resolves partial dates to the last day of their range,
    so this strict-before check errs on the side of catching a too-late paper.
    """
    svc = RetrievalService(test_cache_dir)

    pmids = await svc.fetch_and_cache(
        [_QUERY], db_session_truncating, date_before=_CUTOFF
    )

    parsed = _parsed_dates_for(svc, pmids, db_session_truncating)
    assert parsed, "no dated PMIDs returned — cannot verify the cutoff contract"
    for pmid, d in parsed.items():
        assert d < _CUTOFF, f"PMID {pmid} dated {d} is not before cutoff {_CUTOFF}"


async def test_fetch_and_cache_without_date_before_includes_recent_pmids(
    db_session_truncating, test_cache_dir
):
    """Without date_before, the result is NOT silently capped at the cutoff — at least
    one returned PMID is published on/after _CUTOFF, proving the unbounded path differs
    from the bounded one."""
    svc = RetrievalService(test_cache_dir)

    pmids = await svc.fetch_and_cache([_QUERY], db_session_truncating)

    parsed = _parsed_dates_for(svc, pmids, db_session_truncating)
    assert parsed, "no dated PMIDs returned — cannot verify the unbounded contract"
    assert any(
        d >= _CUTOFF for d in parsed.values()
    ), f"unbounded fetch returned no PMID dated on/after {_CUTOFF}"
