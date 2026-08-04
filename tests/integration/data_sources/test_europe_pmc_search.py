"""Integration tests for EuropePMCClient.search_by_drug against the real Europe PMC API.

Pinned against minoxidil's pre-1988 pool: the drug is old, so the pool is closed — no new papers
can be published into a window that ended in 1987 — which makes exact counts stable.
"""

from datetime import date

import pytest

from indication_scout.models.model_europe_pmc import EuropePMCArticle

MINOXIDIL_CUTOFF = date(1988, 1, 1)
MINOXIDIL_POOL_SIZE = 355


async def test_search_by_drug_parses_a_known_article(europe_pmc_client):
    """Every field of the earliest minoxidil paper, including the nested journal title."""
    articles = await europe_pmc_client.search_by_drug(
        "minoxidil", date_before=MINOXIDIL_CUTOFF
    )
    by_key = {a.article_key: a for a in articles}

    article = by_key["MED:1197258"]
    assert isinstance(article, EuropePMCArticle)
    assert article.source == "MED"
    assert article.record_id == "1197258"
    assert article.pmid == "1197258"
    assert article.title == (
        "Severe hypertension in chronic renal failure treated successfully with Minoxidil."
    )
    assert article.first_publication_date == "1975-01-01"
    assert article.pub_year == 1975
    assert article.abstract
    assert "Journal Article" in article.pub_types


async def test_search_by_drug_returns_the_whole_pool(europe_pmc_client):
    """No result cap: the full closed pool comes back, with unique record identities."""
    articles = await europe_pmc_client.search_by_drug(
        "minoxidil", date_before=MINOXIDIL_CUTOFF
    )
    assert len(articles) == MINOXIDIL_POOL_SIZE
    assert len({a.article_key for a in articles}) == MINOXIDIL_POOL_SIZE


@pytest.mark.parametrize(
    "cutoff, expected_count, latest_date",
    [
        (date(1980, 1, 1), 83, "1979-12-01"),
        (date(1988, 1, 1), MINOXIDIL_POOL_SIZE, "1987-12-22"),
    ],
)
async def test_search_by_drug_date_bound_is_exclusive(
    europe_pmc_client, cutoff, expected_count, latest_date
):
    """`date_before` excludes the cutoff day itself, matching PubMedClient.search.

    Europe PMC's FIRST_PDATE range is inclusive at both ends, so the client sends the preceding
    day. Passing the cutoff straight through returned 18 papers dated exactly 1980-01-01 — a
    holdout cut at an approval date would have seen papers published on that date.
    """
    articles = await europe_pmc_client.search_by_drug("minoxidil", date_before=cutoff)

    assert len(articles) == expected_count
    assert max(a.first_publication_date for a in articles) == latest_date
    assert not [
        a for a in articles if a.first_publication_date >= cutoff.strftime("%Y-%m-%d")
    ]


async def test_search_by_drug_paginates_past_one_page(europe_pmc_client):
    """A pool larger than one page is assembled across cursor pages without loss or duplication.

    Sildenafil bounded at 2004 exceeds the 1000-record page size, so this exercises the cursor
    loop and its termination check (Europe PMC repeats the cursor on the final page rather than
    omitting it).
    """
    articles = await europe_pmc_client.search_by_drug(
        "sildenafil", date_before=date(2004, 1, 1)
    )

    assert len(articles) > 1000
    assert len({a.article_key for a in articles}) == len(articles)
    assert all(a.first_publication_date < "2004-01-01" for a in articles)


async def test_search_by_drug_unmatched_drug_returns_empty(europe_pmc_client):
    assert await europe_pmc_client.search_by_drug("zzzznotadrugzzzz") == []
