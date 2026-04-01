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
        "semaglutide obesity clinical trial"
    )
    articles = await pubmed_client.fetch_abstracts(pmids)

    assert len(articles) >= 5
    [article] = [a for a in articles if a.pmid == "41664890"]

    assert article.pmid.isdigit()
    assert (
        article.title
        == "Who Wins the Battle Against Obesity? A Network Meta-Analysis Comparing Tirzepatide and Semaglutide."
    )
    assert isinstance(article.authors, list)
    assert "Huntermann, Ramon" in article.authors
    expected_keywords = [
        "obese patients",
        "semaglutide",
        "tirzepatide",
    ]
    for k in expected_keywords:
        assert k in article.keywords

    expected_fields = [
        "obesity",
        "absolute weight loss",
        "GLP-1",
        "percentage weight reduction",
    ]
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


async def test_fetch_abstracts_parses_biguanide_colon_cancer_article(pubmed_client):
    """Verify field parsing for PMID 39215927 (metformin / small intestine / prostate cancer).

    Checks title, authors, journal, pub_date, mesh_terms, keywords, and two
    phrases that must appear in the abstract body.
    """
    articles = await pubmed_client.fetch_abstracts(["39215927"])

    assert len(articles) == 1
    article = articles[0]

    assert article.pmid == "39215927"
    assert article.title == (
        "Metformin protects against small intestine damage induced by diabetes and "
        "dunning's prostate cancer: A biochemical and histological study."
    )
    assert article.journal == "Journal of molecular histology"
    assert article.pub_date == "2024-Dec"

    assert article.authors == [
        "Dagsuyu, Eda",
        "Koroglu, Pinar",
        "Bulan, Omur Karabulut",
        "Gul, Ilknur Bugan",
        "Yanardag, Refiye",
    ]

    expected_mesh = [
        "Metformin",
        "Animals",
        "Male",
        "Intestine, Small",
        "Prostatic Neoplasms",
        "Rats",
        "Diabetes Mellitus, Experimental",
        "Oxidative Stress",
        "Lipid Peroxidation",
        "Antioxidants",
        "Hypoglycemic Agents",
        "Glutathione",
        "Reactive Oxygen Species",
    ]
    for term in expected_mesh:
        assert term in article.mesh_terms, f"missing mesh term: {term}"

    assert article.keywords == [
        "Cancer",
        "Diabetes",
        "Metformin",
        "Rat",
        "Small intestine",
    ]

    assert "glutathione (reduced) levels" in article.abstract
    assert "histopathological damage" in article.abstract


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
