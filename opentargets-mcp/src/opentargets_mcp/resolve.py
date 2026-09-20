"""Turn free text into an Open Targets identifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from opentargets_mcp.constants import RESOLVE_HITS
from opentargets_mcp.queries import SEARCH_QUERY
from opentargets_mcp.transport import OpenTargetsError, graphql

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entity:
    """One search hit: its identifier, display name, kind, and description where the API gives one."""

    id: str
    name: str
    entity: str
    description: str | None


@dataclass(frozen=True)
class Resolution:
    """The chosen match plus the other candidates, so a caller can see what else the query could have meant."""

    match: Entity
    alternatives: list[Entity]


async def search(text: str, kind: str, size: int = RESOLVE_HITS) -> list[Entity]:
    """Return search hits for `text` restricted to one entity kind, best match first."""
    data = await graphql(SEARCH_QUERY, {"q": text, "entities": [kind], "size": size})
    hits = (data.get("search") or {}).get("hits") or []
    return [
        Entity(
            id=hit["id"],
            name=hit.get("name") or hit["id"],
            entity=hit.get("entity") or kind,
            description=hit.get("description"),
        )
        for hit in hits
        if hit.get("id")
    ]


async def resolve(text: str, kind: str) -> Resolution:
    """Resolve free text to a single entity of `kind`. Raises OpenTargetsError when nothing matches."""
    hits = await search(text, kind)
    if not hits:
        raise OpenTargetsError(f"No {kind} in Open Targets matches {text!r}")
    logger.info("Resolved %r to %s (%s)", text, hits[0].id, hits[0].name)
    return Resolution(match=hits[0], alternatives=hits[1:])
