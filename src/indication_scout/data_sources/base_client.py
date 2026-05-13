"""
Base client for external data source clients.

Provides: retry with exponential backoff and session management.
"""

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR

logger = logging.getLogger("indication_scout.data_sources")

_settings = get_settings()


def log_data_source_failure(
    source: str,
    url: str,
    context: str,
    error: str | Exception,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Append a tab-separated failure record to data_source_failures.log.

    One line per failure: <iso-timestamp>\\t<source>\\t<url>\\t<context>\\t<error>.
    Best-effort: a write failure logs a warning but does not raise, so the
    caller can continue (or exit) without being blocked on the log itself.
    """
    log_path = cache_dir / "data_source_failures.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(
                f"{datetime.now().isoformat()}\t{source}\t{url}\t{context}\t{error}\n"
            )
    except OSError as log_err:
        logger.warning("Could not write to data_source_failures.log: %s", log_err)


# Keys searched, in order, when building a request-context summary string.
# The first match wins. Covers the common identifying fields across our
# data sources: NCBI eutils (term, id), GraphQL (variables), generic
# search/query params.
_CONTEXT_PRIORITY_KEYS: tuple[str, ...] = (
    "term",
    "query",
    "q",
    "search",
    "id",
    "ids",
    "expr",
)
_CONTEXT_MAX_LEN: int = 200


def _build_context_string(
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
) -> str:
    """Return a short, human-readable summary of the most identifying request field.

    Used to enrich both retry-warning log lines and the persistent fatal
    failure log so a reader can tell *which* call failed (e.g.
    `term='metformin AND obesity'` vs `id='12345'`) without losing the
    URL+source signal that was already there.

    Strategy: walk `_CONTEXT_PRIORITY_KEYS` against params and json_body's
    `variables` (GraphQL convention); return the first hit as `key=repr(value)`,
    truncated. If nothing matches, fall back to a repr of the merged dict.
    """
    candidates: list[dict[str, Any]] = []
    if params:
        candidates.append(params)
    if json_body:
        # GraphQL: identifying args live under `variables`. Body itself
        # also gets searched as a fallback for non-GraphQL POSTs.
        if isinstance(json_body.get("variables"), dict):
            candidates.append(json_body["variables"])
        candidates.append(json_body)

    for source_dict in candidates:
        for key in _CONTEXT_PRIORITY_KEYS:
            if key in source_dict and source_dict[key] not in (None, ""):
                value = source_dict[key]
                rendered = f"{key}={value!r}"
                if len(rendered) > _CONTEXT_MAX_LEN:
                    rendered = rendered[: _CONTEXT_MAX_LEN - 3] + "..."
                return rendered

    if not candidates:
        return ""
    fallback = repr(candidates[0])
    if len(fallback) > _CONTEXT_MAX_LEN:
        fallback = fallback[: _CONTEXT_MAX_LEN - 3] + "..."
    return fallback


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

    If `exit_on_retry_exhausted` is True, a request that fails all retries
    will append a record to data_source_failures.log and call sys.exit
    instead of raising DataSourceError. Use this for hard dependencies
    where downstream analysis cannot proceed correctly without the source
    (e.g. NCBI for MeSH resolution / PubMed efetch).
    """

    exit_on_retry_exhausted: bool = False

    def __init__(self):
        self.timeout = _settings.default_timeout
        self.max_retries = _settings.default_max_retries
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
        as_text: bool = False,
    ) -> Any:
        """Make HTTP request with retry. Returns parsed JSON, raw text, or raises DataSourceError."""
        last_error: Exception | None = None
        # Build once: identifying field summary used in retry warnings and
        # in the persistent failure log so a reader can tell which call
        # failed without losing the URL+source signal.
        context = _build_context_string(params, json_body)

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
                    if attempt < self.max_retries:
                        # 429s usually need real time to clear (server-side rate limits) —
                        # enforce a 90s floor so the first retry doesn't fire before the
                        # window resets. 5xx is more transient, no floor needed.
                        delay = min(2**attempt, 90)
                        if resp.status == 429:
                            delay = max(delay, 90)
                        ctx_suffix = f" ({context})" if context else ""
                        logger.debug(
                            "%s: HTTP %d on %s%s; sleeping %ds and retrying (attempt %d/%d)",
                            self._source_name,
                            resp.status,
                            url,
                            ctx_suffix,
                            delay,
                            attempt + 1,
                            self.max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    err = DataSourceError(
                        self._source_name, f"HTTP {resp.status}", resp.status
                    )
                    if self.exit_on_retry_exhausted:
                        log_data_source_failure(
                            source=self._source_name,
                            url=url,
                            context=context,
                            error=err,
                        )
                        sys.exit(
                            f"FATAL: {self._source_name} unreachable after "
                            f"{self.max_retries + 1} attempts on {url} "
                            f"({context}): {err}"
                        )
                    raise err

                if resp.status >= 400:
                    body = await resp.text()
                    raise DataSourceError(
                        self._source_name,
                        f"HTTP {resp.status}: {body[:200]}",
                        resp.status,
                    )

                return await resp.text() if as_text else await resp.json()

            except asyncio.TimeoutError:
                last_error = DataSourceError(self._source_name, "Request timeout")
            except aiohttp.ClientError as e:
                last_error = DataSourceError(
                    self._source_name, f"Connection error: {e}"
                )

            if attempt < self.max_retries:
                delay = min(2**attempt, 90)
                ctx_suffix = f" ({context})" if context else ""
                logger.debug(
                    "%s: %s on %s%s; sleeping %ds and retrying (attempt %d/%d)",
                    self._source_name,
                    last_error,
                    url,
                    ctx_suffix,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)

        final_error = last_error or DataSourceError(self._source_name, "Unknown error")
        if self.exit_on_retry_exhausted:
            log_data_source_failure(
                source=self._source_name,
                url=url,
                context=context,
                error=final_error,
            )
            sys.exit(
                f"FATAL: {self._source_name} unreachable after "
                f"{self.max_retries + 1} attempts on {url} "
                f"({context}): {final_error}"
            )
        raise final_error

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
        return await self._request("GET", url, params=params, as_text=True)
