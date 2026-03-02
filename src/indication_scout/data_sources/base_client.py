"""
Base client for external data source clients.

Provides: retry with exponential backoff and session management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from indication_scout.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT

logger = logging.getLogger("indication_scout.data_sources")


class DataSourceError(Exception):
    """Exception for data source failures."""

    def __init__(self, source: str, message: str, status_code: int | None = None):
        self.source = source
        self.status_code = status_code
        super().__init__(f"[{source}] {message}")


class BaseClient(ABC):
    """
    Abstract base for data source clients.

    Subclasses set `_source_name` and use `_rest_get()` or `_graphql()`.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    @property
    @abstractmethod
    def _source_name(self) -> str:
        """Identifier for this data source."""
        ...

    # -- Session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- HTTP requests with retry --------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make HTTP request with retry. Returns parsed JSON or raises DataSourceError."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()

                if method.upper() == "GET":
                    resp = await session.get(url, params=params, headers=headers)
                else:
                    resp = await session.post(
                        url, json=json_body, params=params, headers=headers
                    )

                # Retry on 429/5xx
                if resp.status in {429, 500, 502, 503, 504}:
                    last_error = DataSourceError(
                        self._source_name, f"HTTP {resp.status}", resp.status
                    )
                    if attempt < self.max_retries:
                        delay = min(2**attempt, 30)
                        await asyncio.sleep(delay)
                        continue
                    raise last_error

                if resp.status >= 400:
                    body = await resp.text()
                    raise DataSourceError(
                        self._source_name,
                        f"HTTP {resp.status}: {body[:200]}",
                        resp.status,
                    )

                return await resp.json()

            except asyncio.TimeoutError:
                last_error = DataSourceError(self._source_name, "Request timeout")
            except aiohttp.ClientError as e:
                last_error = DataSourceError(
                    self._source_name, f"Connection error: {e}"
                )

            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt, 30))

        raise last_error or DataSourceError(self._source_name, "Unknown error")

    async def _rest_get(self, url: str, params: dict[str, Any]) -> Any:
        """REST GET request."""
        return await self._request("GET", url, params=params)

    async def _graphql(self, url: str, query: str, variables: dict[str, Any]) -> Any:
        """GraphQL POST request."""
        data = await self._request(
            "POST",
            url,
            json_body={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
        )

        if data and "errors" in data:
            errors = [e.get("message", str(e)) for e in data["errors"]]
            raise DataSourceError(self._source_name, f"GraphQL: {errors}")

        return data

    async def _rest_get_xml(self, url: str, params: dict[str, Any]) -> str:
        """REST GET that returns XML text instead of JSON."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                resp = await session.get(url, params=params)

                if resp.status in {429, 500, 502, 503, 504}:
                    last_error = DataSourceError(
                        self._source_name, f"HTTP {resp.status}", resp.status
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 30))
                        continue
                    raise last_error

                if resp.status >= 400:
                    body = await resp.text()
                    raise DataSourceError(
                        self._source_name,
                        f"HTTP {resp.status}: {body[:200]}",
                        resp.status,
                    )

                return await resp.text()

            except asyncio.TimeoutError:
                last_error = DataSourceError(self._source_name, "Request timeout")
            except aiohttp.ClientError as e:
                last_error = DataSourceError(
                    self._source_name, f"Connection error: {e}"
                )

            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt, 30))

        raise last_error or DataSourceError(self._source_name, "Unknown error")
