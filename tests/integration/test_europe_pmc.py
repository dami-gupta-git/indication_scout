"""Integration tests for Europe PMC drug-scoped literature retrieval.

Hits the live Europe PMC search API. Tirofiban is used throughout: its pool is large enough to
span more than one page but small enough to fetch quickly, and its literature is historical so
bounded windows are closed and their contents no longer change.
"""

import logging
from datetime import date

from indication_scout.constants import EUROPE_PMC_PAGE_SIZE
from indication_scout.models.model_europe_pmc import EuropePMCArticle

logger = logging.getLogger(__name__)

# A 1999 record inside the bounded window. Every asserted field is fixed for good; citation count
# and open-access status are deliberately not asserted, since both can change after publication.
ANCHOR_KEY = "MED:10064007"


async def test_search_by_drug_parses_known_article(europe_pmc_client):
    """Field parsing against a specific record in a closed historical window."""
    articles = await europe_pmc_client.search_by_drug(
        "tirofiban", date_before=date(2000, 1, 1)
    )
    by_key = {a.article_key: a for a in articles}
    assert ANCHOR_KEY in by_key

    article = by_key[ANCHOR_KEY]
    assert article.source == "MED"
    assert article.record_id == "10064007"
    assert article.pmid == "10064007"
    assert article.doi == "10.1055/s-0037-1614458"
    assert article.title == (
        "Modulation of platelet-neutrophil interaction with pharmacological inhibition of "
        "fibrinogen binding to platelet GPIIb/IIIa receptor."
    )
    assert article.journal == "Thrombosis and haemostasis"
    assert article.first_publication_date == "1999-02-01"
    assert article.pub_year == 1999
    assert article.pub_types == [
        "Clinical Trial",
        "Randomized Controlled Trial",
        "Journal Article",
    ]
    assert article.abstract.startswith(
        "The study investigated how drug inhibition of the GPIIb/IIIa"
    )


async def test_search_by_drug_respects_date_bound(europe_pmc_client):
    """No record first published after the cutoff is returned.

    The bound is inclusive, so records dated exactly on the cutoff belong in the pool — the
    assertion is on the full date, not the year, because a record first published on the cutoff
    day reports a pub_year equal to the cutoff year.
    """
    cutoff = date(2000, 1, 1)
    articles = await europe_pmc_client.search_by_drug("tirofiban", date_before=cutoff)

    assert len(articles) >= 70
    assert all(a.first_publication_date <= cutoff.isoformat() for a in articles)

    unbounded = await europe_pmc_client.search_by_drug("tirofiban")
    assert len(unbounded) > len(articles)


async def test_search_by_drug_paginates_past_one_page(europe_pmc_client):
    """The unbounded tirofiban pool exceeds one page, so the cursor loop must run more than once.

    Asserts the pool is larger than a single page and that record identity is unique across it —
    a duplicate would mean a page was re-requested, and would corrupt the per-article extraction
    cache downstream.
    """
    articles = await europe_pmc_client.search_by_drug("tirofiban")

    assert len(articles) > EUROPE_PMC_PAGE_SIZE
    assert len({a.article_key for a in articles}) == len(articles)
    assert all(isinstance(a, EuropePMCArticle) for a in articles)


async def test_search_by_drug_unmatched_drug_returns_empty(europe_pmc_client):
    """A drug name with no literature yields an empty pool rather than raising."""
    articles = await europe_pmc_client.search_by_drug("zzzznotadrugzzzz")
    assert articles == []
