"""Unit tests for PubMedClient."""

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.utils.cache import cache_get

_GOOD_XML = (
    '<?xml version="1.0"?>'
    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
    "<PMID>12345678</PMID><Article><ArticleTitle>T</ArticleTitle></Article>"
    "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
)
_MALFORMED_XML = "<?xml version='1.0'?>\n<PubmedArticleSet>\n<broken & token"

# --- complete direct-query search regression ---


async def test_search_complete_retrieves_supporting_pmid_beyond_first_page(
    tmp_path: Path,
) -> None:
    """A directly matched trial after the first 200 results remains in the evidence pool."""
    query = 'metformin AND "Breast Neoplasms"'
    supporting_pmid = "40579605"
    all_pmids = [str(10_000_000 + index) for index in range(672)]
    all_pmids[529] = supporting_pmid
    requested_offsets: list[int] = []

    async def search_page(url: str, params: dict[str, Any]) -> dict[str, Any]:
        assert url == client.SEARCH_URL
        assert params["datetype"] == "pdat"
        assert params["mindate"] == "1900/01/01"
        assert params["maxdate"] == "2023/12/31"
        offset = int(params.get("retstart", 0))
        page_size = int(params["retmax"])
        requested_offsets.append(offset)
        return {
            "esearchresult": {
                "count": str(len(all_pmids)),
                "retmax": str(page_size),
                "retstart": str(offset),
                "idlist": all_pmids[offset : offset + page_size],
            }
        }

    client = PubMedClient(tmp_path)
    with (
        patch.object(client, "_rest_get_json_tolerant", new=search_page),
        patch("indication_scout.data_sources.pubmed.asyncio.sleep", new=AsyncMock()),
    ):
        result = await client.search_complete(
            query, page_size=200, date_before=date(2024, 1, 1)
        )
        cached_result = await client.search_complete(
            query, page_size=200, date_before=date(2024, 1, 1)
        )

    assert result == all_pmids
    assert cached_result == all_pmids
    assert result[529] == supporting_pmid
    assert requested_offsets == [0, 200, 400, 600]


async def test_search_complete_rejects_more_than_pubmed_limit(tmp_path: Path) -> None:
    """A complete direct query never returns PubMed's partial first 10,000 records."""
    client = PubMedClient(tmp_path)
    response = {
        "esearchresult": {
            "count": "10001",
            "retmax": "200",
            "retstart": "0",
            "idlist": [str(10_000_000 + index) for index in range(200)],
        }
    }
    with (
        patch.object(
            client, "_rest_get_json_tolerant", new=AsyncMock(return_value=response)
        ),
        patch("indication_scout.data_sources.pubmed.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(DataSourceError) as exc_info:
            await client.search_complete('drug AND "Disease"', page_size=200)

    assert exc_info.value.source == "pubmed"
    assert (
        str(exc_info.value)
        == "[pubmed] Direct query 'drug AND \"Disease\"' matched 10001 records; "
        "PubMed ESearch exposes at most 10000"
    )


# --- _parse_pubmed_xml ---


def test_invalid_xml_raises_error(tmp_path):
    """Test that invalid XML raises DataSourceError."""
    client = PubMedClient(tmp_path)
    invalid_xml = "not valid xml <unclosed"

    with pytest.raises(DataSourceError) as exc_info:
        client._parse_pubmed_xml(invalid_xml)

    assert exc_info.value.source == "pubmed"
    assert "Failed to parse XML" in str(exc_info.value)


def test_empty_xml_raises_error(tmp_path):
    """Test that empty string raises DataSourceError."""
    client = PubMedClient(tmp_path)

    with pytest.raises(DataSourceError) as exc_info:
        client._parse_pubmed_xml("")

    assert exc_info.value.source == "pubmed"
    assert "Failed to parse XML" in str(exc_info.value)


def test_valid_xml_no_articles_returns_empty_list(tmp_path):
    """Test that valid XML with no PubmedArticle elements returns empty list."""
    client = PubMedClient(tmp_path)
    xml = "<PubmedArticleSet></PubmedArticleSet>"

    result = client._parse_pubmed_xml(xml)

    assert result == []


def test_valid_xml_parses_article(tmp_path):
    """Test that valid XML with article is parsed correctly."""
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Test Article Title</ArticleTitle>
                    <Abstract>
                        <AbstractText>This is the abstract text.</AbstractText>
                    </Abstract>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <ForeName>John</ForeName>
                        </Author>
                    </AuthorList>
                    <Journal>
                        <Title>Test Journal</Title>
                    </Journal>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <History>
                    <PubMedPubDate PubStatus="pubmed">
                        <Year>2023</Year>
                        <Month>06</Month>
                        <Day>15</Day>
                    </PubMedPubDate>
                </History>
            </PubmedData>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert len(result) == 1
    article = result[0]
    assert article.pmid == "12345678"
    assert article.title == "Test Article Title"
    assert article.abstract == "This is the abstract text."
    assert article.authors == ["Smith, John"]
    assert article.journal == "Test Journal"


def test_article_without_pmid_is_skipped(tmp_path):
    """Test that articles without PMID are skipped."""
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>No PMID Article</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert result == []


def test_valid_xml_parses_book_article(tmp_path):
    """PubmedBookArticle elements are parsed into PubmedAbstract objects.

    Book chapters have a different XML structure: BookDocument instead of
    MedlineCitation, BookTitle instead of Journal/Title, and separate
    AuthorList elements for authors vs editors.
    """
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedBookArticle>
            <BookDocument>
                <PMID>20301421</PMID>
                <Book>
                    <BookTitle>GeneReviews</BookTitle>
                    <PubDate><Year>1993</Year></PubDate>
                    <AuthorList Type="editors">
                        <Author><LastName>Adam</LastName><ForeName>Margaret P</ForeName></Author>
                    </AuthorList>
                </Book>
                <ArticleTitle>ALS2-Related Disorder</ArticleTitle>
                <AuthorList Type="authors">
                    <Author><LastName>Orrell</LastName><ForeName>Richard W</ForeName></Author>
                </AuthorList>
                <Abstract>
                    <AbstractText Label="CLINICAL CHARACTERISTICS">Spasticity with increased reflexes.</AbstractText>
                    <AbstractText Label="MANAGEMENT">Multidisciplinary care.</AbstractText>
                </Abstract>
                <KeywordList>
                    <Keyword>ALS2</Keyword>
                    <Keyword>Alsin</Keyword>
                </KeywordList>
            </BookDocument>
            <PubmedBookData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">20301421</ArticleId>
                </ArticleIdList>
            </PubmedBookData>
        </PubmedBookArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert len(result) == 1
    article = result[0]
    assert article.pmid == "20301421"
    assert article.title == "ALS2-Related Disorder"
    assert (
        article.abstract
        == "CLINICAL CHARACTERISTICS: Spasticity with increased reflexes. MANAGEMENT: Multidisciplinary care."
    )
    assert article.authors == ["Orrell, Richard W"]  # editors excluded
    assert article.journal == "GeneReviews"
    assert article.pub_date == "1993"
    assert article.mesh_terms == []  # not present in book articles
    assert article.keywords == ["ALS2", "Alsin"]


def test_book_article_excludes_editors_from_authors(tmp_path):
    """AuthorList Type='editors' must not appear in the authors field."""
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedBookArticle>
            <BookDocument>
                <PMID>20301421</PMID>
                <Book>
                    <BookTitle>GeneReviews</BookTitle>
                    <AuthorList Type="editors">
                        <Author><LastName>Editor</LastName><ForeName>One</ForeName></Author>
                        <Author><LastName>Editor</LastName><ForeName>Two</ForeName></Author>
                    </AuthorList>
                </Book>
                <ArticleTitle>Test Chapter</ArticleTitle>
                <AuthorList Type="authors">
                    <Author><LastName>Author</LastName><ForeName>One</ForeName></Author>
                </AuthorList>
            </BookDocument>
        </PubmedBookArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert len(result) == 1
    assert result[0].authors == ["Author, One"]


# --- _parse_pubmed_xml warms the pubtypes cache ---


def test_parse_warms_pubtypes_cache_for_article(tmp_path):
    """Parsing a journal article writes its PublicationType list to the pubtypes cache,
    so a later fetch_pubtypes() hits the cache instead of an esummary round-trip."""
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Test Article Title</ArticleTitle>
                    <PublicationTypeList>
                        <PublicationType UI="D016428">Journal Article</PublicationType>
                        <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
                    </PublicationTypeList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert len(result) == 1
    assert result[0].pmid == "12345678"
    cached = cache_get("pubmed_pubtypes", {"pmid": "12345678"}, tmp_path)
    assert cached == ["Journal Article", "Randomized Controlled Trial"]


def test_parse_warms_pubtypes_cache_empty_when_absent(tmp_path):
    """A book article with no PublicationType caches an empty list (a real answer,
    not a miss), so fetch_pubtypes() won't refetch it. cache_get returns [] not None."""
    client = PubMedClient(tmp_path)
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedBookArticle>
            <BookDocument>
                <PMID>20301421</PMID>
                <Book><BookTitle>GeneReviews</BookTitle></Book>
                <ArticleTitle>ALS2-Related Disorder</ArticleTitle>
            </BookDocument>
        </PubmedBookArticle>
    </PubmedArticleSet>
    """

    result = client._parse_pubmed_xml(xml)

    assert len(result) == 1
    assert result[0].pmid == "20301421"
    cached = cache_get("pubmed_pubtypes", {"pmid": "20301421"}, tmp_path)
    assert cached == []


# --- fetch_abstracts re-fetches on malformed efetch XML ---


async def test_fetch_abstracts_retries_malformed_then_succeeds(tmp_path):
    """A malformed efetch body (HTTP 200) is re-fetched; the next good body parses."""
    client = PubMedClient(tmp_path)
    mock_get = AsyncMock(side_effect=[_MALFORMED_XML, _GOOD_XML])
    with (
        patch.object(client, "_rest_get_xml", mock_get),
        patch("indication_scout.data_sources.pubmed.asyncio.sleep", new=AsyncMock()),
    ):
        result = await client.fetch_abstracts(["12345678"])

    assert mock_get.await_count == 2
    assert len(result) == 1
    assert result[0].pmid == "12345678"


async def test_fetch_abstracts_raises_after_exhausting_retries(tmp_path):
    """Persistently malformed bodies raise DataSourceError after the retry budget."""
    client = PubMedClient(tmp_path)
    mock_get = AsyncMock(return_value=_MALFORMED_XML)
    with (
        patch.object(client, "_rest_get_xml", mock_get),
        patch("indication_scout.data_sources.pubmed.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(DataSourceError) as exc_info:
            await client.fetch_abstracts(["12345678"])

    # PUBMED_EFETCH_PARSE_RETRIES=3 → 1 initial + 3 retries = 4 fetch attempts.
    assert mock_get.await_count == 4
    assert "Failed to parse XML" in str(exc_info.value)


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (
            400,
            "<eFetchResult><ERROR> Error occurred: cannot get document summary</ERROR></eFetchResult>",
            True,
        ),
        (
            400,
            "<eFetchResult><ERROR>ID list is empty! Possibly it has no correct IDs.</ERROR></eFetchResult>",
            False,
        ),
        (200, "<eFetchResult><ERROR>Error occurred: x</ERROR></eFetchResult>", False),
        (400, "Bad Request", False),
    ],
)
def test_is_transient_error_body(tmp_path, status, body, expected):
    client = PubMedClient(tmp_path)
    assert client._is_transient_error_body(status, body) is expected
