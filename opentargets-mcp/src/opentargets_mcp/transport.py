"""GraphQL transport for the Open Targets Platform API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
REQUEST_TIMEOUT_SECONDS = 30.0


class OpenTargetsError(RuntimeError):
    """Raised when a query cannot be completed or the API returns GraphQL errors."""


async def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Post a query and return its `data` payload. Raises OpenTargetsError on transport or GraphQL failure."""
    payload = {"query": query, "variables": variables}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPStatusError as exc:
        raise OpenTargetsError(f"Open Targets returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise OpenTargetsError(f"Could not reach Open Targets: {exc}") from exc
    except ValueError as exc:
        raise OpenTargetsError("Open Targets returned a body that is not JSON") from exc

    errors = body.get("errors")
    if errors:
        messages = "; ".join(e.get("message", "unknown error") for e in errors)
        raise OpenTargetsError(f"Open Targets rejected the query: {messages}")

    data = body.get("data")
    if data is None:
        raise OpenTargetsError("Open Targets returned no data for the query")
    return data
