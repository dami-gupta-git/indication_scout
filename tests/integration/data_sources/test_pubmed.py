"""Integration tests for PubMedClient."""

import pytest

# --- Main functionality ---


@pytest.mark.parametrize(
    "query, expected_pmids",
    [
        (
            "sildenafil AND diabetic nephropathy",
            ["21133896", "18812693", "35671809", "27261587"],
        ),
    ],
)
async def test_search_returns_known_pmids(pubmed_client, query, expected_pmids):
    """Search results must include all known PMIDs for a specific query."""
    pmids = await pubmed_client.search(query, max_results=50)

    for pmid in expected_pmids:
        assert pmid in pmids


async def test_search_returns_pmids(pubmed_client):
    """Test search returns a list of PMIDs for a query."""
    pmids = await pubmed_client.search("semaglutide diabetes", max_results=10)

    assert len(pmids) == 10
    # PMIDs are numeric strings
    for pmid in pmids:
        assert pmid.isdigit()


async def test_get_count_returns_total(pubmed_client):
    """Test get_count returns total count without fetching articles."""
    count = await pubmed_client.get_count("semaglutide diabetes")

    # Should be many articles about semaglutide and diabetes
    assert count > 500


async def test_fetch_articles_parses_correctly(pubmed_client):
    """Test fetch_articles returns parsed PubmedAbstract objects."""
    # First search for a known article
    pmids = await pubmed_client.search(
        "semaglutide obesity clinical trial", max_results=5
    )
    articles = await pubmed_client.fetch_abstracts(pmids)

    assert len(articles) == 5
    [article] = [a for a in articles if a.pmid == '41754149']


    assert article.pmid.isdigit()
    assert article.title == 'A Narrative Review on GLP-1 Receptor Agonists for Obesity in Older Women: Maximizing Weight Loss While Preserving Lean Mass.'
    assert isinstance(article.authors, list)
    assert 'Menichelli, Danilo' in article.authors
    expected_keywords = ['GLP-1 receptor agonist', 'cardiovascular prevention', 'lifestyle for prevention', 'obesity', 'old women', 'sarcopenia prevention']
    for k in expected_keywords:
        assert k in article.keywords
    expected_fields = ['cardiovascular disease','GLP-1 RAs','obesity','phase 3','weight loss','osteoarthritis','muscle mass']
    for e in expected_fields:
        assert e in article.abstract


async def test_fetch_specific_article(pubmed_client):
    """Test fetching a specific known article by PMID."""
    # PMID 27633186 - SUSTAIN-6 semaglutide cardiovascular trial (NEJM 2016)
    articles = await pubmed_client.fetch_abstracts(["27633186"])

    assert len(articles) == 1
    article = articles[0]

    assert article.pmid == "27633186"
    assert "semaglutide" in article.title.lower()
    assert article.journal == "The New England journal of medicine"
    assert len(article.authors) > 0
    # First author is Marso
    assert article.authors[0].startswith("Marso")


async def test_fetch_articles_empty_list_returns_empty(pubmed_client):
    """Test that fetching empty list returns empty list."""
    articles = await pubmed_client.fetch_abstracts([])

    assert articles == []


# --- Edge cases and weird inputs ---


async def test_search_nonexistent_term_returns_empty(pubmed_client):
    """Test that a nonexistent search term returns empty list."""
    pmids = await pubmed_client.search(
        "xyzzy_fake_drug_99999_not_real_term", max_results=10
    )

    assert pmids == []


async def test_get_count_nonexistent_term_returns_zero(pubmed_client):
    """Test that a nonexistent term returns zero count."""
    count = await pubmed_client.get_count("xyzzy_fake_drug_99999_not_real_term")

    assert count == 0


async def test_search_empty_query_returns_empty(pubmed_client):
    """Test that empty query returns empty list."""
    pmids = await pubmed_client.search("", max_results=10)

    # Empty search returns nothing
    assert pmids == []


async def test_fetch_articles_invalid_pmid_returns_empty(pubmed_client):
    """Test that invalid PMIDs return empty list (no matching articles)."""
    articles = await pubmed_client.fetch_abstracts(["99999999999"])

    # Invalid PMID returns no articles
    assert articles == []


async def test_search_special_characters_returns_empty(pubmed_client):
    """Test that special characters in query returns empty list."""
    pmids = await pubmed_client.search("!!!@@@###$$$", max_results=10)

    assert pmids == []
