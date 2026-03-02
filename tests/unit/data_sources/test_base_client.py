"""Unit tests for base_client module."""

from unittest.mock import AsyncMock, patch


import pytest

from indication_scout.data_sources.base_client import BaseClient, DataSourceError


class ConcreteTestClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    @property
    def _source_name(self) -> str:
        return "test_client"


# --- BaseClient session lifecycle (no network calls) ---


async def test_client_context_manager():
    """Test that client can be used as async context manager."""
    async with ConcreteTestClient() as client:
        assert client._session is None  # Session created lazily
        session = await client._get_session()

        assert session is not None
        assert not session.closed

    # Session should be closed after exiting context
    assert client._session.closed


async def test_session_reuse():
    """Test that session is reused across requests."""
    client = ConcreteTestClient()

    session1 = await client._get_session()
    session2 = await client._get_session()

    assert session1 is session2
    await client.close()


# --- _graphql error handling ---


async def test_graphql_errors_in_response_raises_datasource_error():
    """Test _graphql raises DataSourceError when response contains GraphQL errors.

    Some GraphQL servers return HTTP 200 with an 'errors' key for
    partial or runtime failures. This path cannot be triggered by
    Open Targets (which returns HTTP 400 instead), so we mock _request.
    """
    client = ConcreteTestClient()
    mock_response = {
        "data": None,
        "errors": [
            {"message": "Field 'x' not found on type 'Query'"},
            {"message": "Unauthorized access"},
        ],
    }

    with patch.object(
        client, "_request", new_callable=AsyncMock, return_value=mock_response
    ):
        with pytest.raises(DataSourceError, match="GraphQL") as exc_info:
            await client._graphql(
                "https://example.com/graphql",
                "{ x }",
                variables={},
            )

        assert "Field 'x' not found on type 'Query'" in str(exc_info.value)
        assert "Unauthorized access" in str(exc_info.value)
        assert exc_info.value.source == "test_client"


# --- _rest_get_xml ---


async def test_rest_get_xml_returns_xml_text_on_success():
    """Test _rest_get_xml returns raw text for a 200 response."""
    xml_body = "<PubmedArticleSet><PubmedArticle></PubmedArticle></PubmedArticleSet>"
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value=xml_body)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_resp)

    client = ConcreteTestClient()
    with patch.object(
        client, "_get_session", new_callable=AsyncMock, return_value=mock_session
    ):
        result = await client._rest_get_xml(
            "https://example.com/xml", params={"id": "1"}
        )

    assert result == xml_body


async def test_rest_get_xml_raises_datasource_error_on_4xx():
    """Test _rest_get_xml raises DataSourceError for non-retryable 4xx."""
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.text = AsyncMock(return_value="Not Found")

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_resp)

    client = ConcreteTestClient(max_retries=0)
    with patch.object(
        client, "_get_session", new_callable=AsyncMock, return_value=mock_session
    ):
        with pytest.raises(DataSourceError, match="HTTP 404") as exc_info:
            await client._rest_get_xml("https://example.com/xml", params={})

    assert exc_info.value.status_code == 404
    assert exc_info.value.source == "test_client"


async def test_rest_get_xml_retries_on_5xx_then_succeeds():
    """Test _rest_get_xml retries on 500 and succeeds on next attempt."""
    xml_body = "<root>OK</root>"

    error_resp = AsyncMock()
    error_resp.status = 500

    ok_resp = AsyncMock()
    ok_resp.status = 200
    ok_resp.text = AsyncMock(return_value=xml_body)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=[error_resp, ok_resp])

    client = ConcreteTestClient(max_retries=1)
    with patch.object(
        client, "_get_session", new_callable=AsyncMock, return_value=mock_session
    ):
        with patch(
            "indication_scout.data_sources.base_client.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await client._rest_get_xml("https://example.com/xml", params={})

    assert result == xml_body
    assert mock_session.get.call_count == 2


async def test_rest_get_xml_raises_after_exhausting_retries_on_5xx():
    """Test _rest_get_xml raises DataSourceError after all retries fail with 5xx."""
    error_resp = AsyncMock()
    error_resp.status = 503

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=error_resp)

    client = ConcreteTestClient(max_retries=2)
    with patch.object(
        client, "_get_session", new_callable=AsyncMock, return_value=mock_session
    ):
        with patch(
            "indication_scout.data_sources.base_client.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(DataSourceError, match="HTTP 503") as exc_info:
                await client._rest_get_xml("https://example.com/xml", params={})

    assert exc_info.value.status_code == 503
    assert mock_session.get.call_count == 3


# --- DataSourceError ---


def test_error_message_format():
    """Test error message includes source."""
    error = DataSourceError("pubmed", "Connection failed")
    assert "[pubmed]" in str(error)
    assert "Connection failed" in str(error)


def test_error_with_status_code():
    """Test error can include status code."""
    error = DataSourceError("api", "Not found", status_code=404)
    assert error.source == "api"
    assert error.status_code == 404
