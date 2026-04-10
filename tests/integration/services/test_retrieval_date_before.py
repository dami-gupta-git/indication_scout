"""Integration tests for RetrievalService.fetch_and_cache date_before filtering."""

import logging
from datetime import date

from sqlalchemy import text

from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Query with known results verified live. "biguanides AND colon cancer" returns
# PMIDs spanning multiple decades, so a cutoff date will meaningfully reduce results.
_QUERY = "biguanides AND colon cancer"

_CUTOFF = date(2015, 1, 1)
# PMID 17571619 : Publication date Jun-2007 : A cutoff of 2015-01-01 must include it.
# PMID 28423651: Published 2017-Apr-25. A cutoff of 2020-01-01 must exclude it.
# PMID 27039825: Published 2016. A cutoff of 2020-01-01 must exclude it.
_PMID_INCLUDED_AFTER_CUTOFF = ["17571619"]
_PMID_EXCLUDED_AFTER_CUTOFF = ["28423651", "27039825"]


async def test_single(db_session_truncating, test_cache_dir):
    # 28423651 was published in 2017-Apr-25
    svc = RetrievalService(test_cache_dir)
    cutoff_1 = date(2015, 4, 10)
    PMID = "28423651"
    pmids = await svc.fetch_and_cache(
        [_QUERY], db_session_truncating, date_before=cutoff_1
    )
    assert PMID not in pmids

    cutoff_2 = date(2017, 5, 30)
    pmids = await svc.fetch_and_cache(
        [_QUERY], db_session_truncating, date_before=cutoff_2
    )
    assert PMID in pmids


async def test_fetch_and_cache_date_before_excludes_recent_pmids(
    db_session_truncating, test_cache_dir
):
    """fetch_and_cache with date_before excludes PMIDs published after the cutoff.

    Uses a stable query with two known PMIDs: one published before and one after
    the cutoff date. Asserts the pre-cutoff PMID is included and the post-cutoff
    PMID is excluded.
    """
    svc = RetrievalService(test_cache_dir)

    pmids = await svc.fetch_and_cache(
        [_QUERY], db_session_truncating, date_before=_CUTOFF
    )

    assert all(p in pmids for p in _PMID_INCLUDED_AFTER_CUTOFF)
    assert all(excluded not in pmids for excluded in _PMID_EXCLUDED_AFTER_CUTOFF)


async def test_fetch_and_cache_without_date_before_includes_recent_pmids(
    db_session_truncating, test_cache_dir
):
    """fetch_and_cache without date_before includes both old and recent PMIDs."""
    svc = RetrievalService(test_cache_dir)

    pmids = await svc.fetch_and_cache([_QUERY], db_session_truncating)

    assert all(
        p in pmids for p in _PMID_INCLUDED_AFTER_CUTOFF + _PMID_EXCLUDED_AFTER_CUTOFF
    )
