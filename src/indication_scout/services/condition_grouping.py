"""Collapse extracted condition names into candidate indications.

Extraction preserves each abstract's own wording, so one condition arrives under many names
("alopecia", "male pattern baldness", "androgenetic alopecia"). Those names are merged in a single
`merge_duplicate_diseases` call, which also reports which of them are already-approved indications.

One call, not batches. Measured: 81 duloxetine names in 3.7s, 257 colchicine names in 13.7s, both
well inside the response limit. Batching was tried and abandoned — splitting the set meant no call
could compare all variants of a term at once, and re-merging the survivors of earlier rounds
ratcheted toward generic categories (three distinct animal pain assays collapsed into a bare
"pain"). See design_europe_pmc.md.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from indication_scout.services.disease_helper import merge_duplicate_diseases

logger = logging.getLogger(__name__)


class GroupedCondition(BaseModel):
    """One candidate indication and the articles it was extracted from.

    ``aliases`` are the raw extracted strings that collapsed into ``name``, kept so a candidate can
    be traced back to the wording the literature actually used.
    """

    name: str
    aliases: list[str] = []
    article_keys: list[str] = []

    @property
    def paper_count(self) -> int:
        return len(self.article_keys)


async def group_conditions(
    conditions: dict[str, list[str]],
    approved_indications: list[str],
) -> list[GroupedCondition]:
    """Merge synonymous condition names and drop those already approved.

    ``conditions`` maps a raw extracted name to the article keys it came from.
    ``approved_indications`` must already be resolved as of the retrieval cutoff — under a holdout,
    approvals granted after the cutoff are not approved yet and their conditions must survive.

    Returns candidates sorted by paper count, descending. Propagates the ``DataSourceError`` raised
    when the merge response is unparseable: a candidate list that silently kept its approved
    indications would present them as novel repurposing hits.
    """
    if not conditions:
        return []

    names = sorted(conditions)
    result = await merge_duplicate_diseases(names, approved_indications)

    # alias -> canonical, for the names the merge actually collapsed. Aliases the LLM returns that
    # were never in the input are ignored; they are not ours to map.
    canonical_of: dict[str, str] = {}
    for canonical, aliases in result["merge"].items():
        for alias in aliases:
            if alias in conditions and alias != canonical:
                canonical_of[alias] = canonical

    removed = {r.lower().strip() for r in result["remove"]}

    grouped: dict[str, GroupedCondition] = {}
    for raw_name, article_keys in conditions.items():
        canonical = canonical_of.get(raw_name, raw_name)
        if canonical.lower().strip() in removed or raw_name.lower().strip() in removed:
            continue
        entry = grouped.setdefault(canonical, GroupedCondition(name=canonical))
        if raw_name != canonical:
            entry.aliases.append(raw_name)
        entry.article_keys.extend(article_keys)

    for entry in grouped.values():
        entry.aliases = sorted(set(entry.aliases))
        entry.article_keys = list(dict.fromkeys(entry.article_keys))

    logger.info(
        "condition grouping: %s raw names -> %s candidates (%s removed as approved)",
        len(names),
        len(grouped),
        len(removed),
    )
    return sorted(grouped.values(), key=lambda c: (-c.paper_count, c.name))
