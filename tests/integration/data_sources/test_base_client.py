"""Integration tests for base_client module."""

import pytest

from indication_scout.constants import OPEN_TARGETS_BASE_URL
from indication_scout.data_sources.base_client import BaseClient, DataSourceError


class ConcreteTestClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    @property
    def _source_name(self) -> str:
        return "test_client"


# --- BaseClient integration tests (requires network) ---


async def test_get_request_to_httpbin():
    """Test GET request to a real endpoint."""
    async with ConcreteTestClient(timeout=30.0) as client:
        result = await client._rest_get(
            "https://httpbin.org/get",
            params={"test_param": "test_value"},
        )

        assert result is not None
        assert result["args"]["test_param"] == "test_value"


async def test_post_request_to_httpbin():
    """Test POST request to a real endpoint."""
    async with ConcreteTestClient(timeout=30.0) as client:
        result = await client._request(
            "POST",
            "https://httpbin.org/post",
            json_body={"key": "value"},
            headers={"Content-Type": "application/json"},
        )

        assert result is not None
        assert result["json"]["key"] == "value"


async def test_graphql_successful_query():
    """Test _graphql with Open Targets GraphQL endpoint."""
    query = """
    query($q: String!) {
        search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
            hits { id entity }
        }
    }
    """
    async with ConcreteTestClient(timeout=30.0) as client:
        result = await client._graphql(
            OPEN_TARGETS_BASE_URL,
            query,
            variables={"q": "imatinib"},
        )

        hits = result["data"]["search"]["hits"]
        assert len(hits) == 1
        assert hits[0]["id"] == "CHEMBL941"
        assert hits[0]["entity"] == "drug"


async def test_graphql_invalid_query_raises_datasource_error():
    """Test _graphql raises DataSourceError for an invalid query.

    Open Targets returns HTTP 400 for malformed queries, so the error
    is raised by _request before _graphql's own error handling.
    """
    invalid_query = "{ invalidField }"
    async with ConcreteTestClient(timeout=30.0) as client:
        with pytest.raises(DataSourceError, match="HTTP 400") as exc_info:
            await client._graphql(
                OPEN_TARGETS_BASE_URL,
                invalid_query,
                variables={},
            )
        assert exc_info.value.status_code == 400


async def test_rest_get_xml_returns_xml_string():
    """Test _rest_get_xml with PubMed efetch endpoint."""
    async with ConcreteTestClient(timeout=30.0) as client:
        result = await client._rest_get_xml(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": "33914610",
                "retmode": "xml",
                "rettype": "abstract",
            },
        )

        assert isinstance(result, str)
        assert "<PubmedArticleSet>" in result
        assert "<PMID" in result
        assert "33914610" in result


async def test_timeout_raises_error():
    """Test that timeouts raise DataSourceError."""
    async with ConcreteTestClient(timeout=0.001, max_retries=0) as client:
        with pytest.raises(DataSourceError):
            await client._rest_get(
                "https://httpbin.org/delay/10",  # 10 second delay
                params={},
            )
