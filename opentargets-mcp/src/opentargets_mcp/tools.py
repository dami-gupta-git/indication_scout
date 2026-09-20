"""The six tool implementations. Each resolves free text, runs one query, and renders a compact text result."""

from __future__ import annotations

import logging

from opentargets_mcp import queries
from opentargets_mcp.constants import (
    DRUG_DISEASE_ROWS,
    EVIDENCE_LITERATURE_IDS,
    FUNCTION_DESCRIPTION_CHARS,
    MAX_ASSOCIATION_ROWS,
    MAX_DRUG_ROWS,
    MAX_EVIDENCE_ROWS,
)
from opentargets_mcp.resolve import Resolution, resolve, search
from opentargets_mcp.transport import OpenTargetsError, graphql

logger = logging.getLogger(__name__)


def _matched_line(resolution: Resolution) -> str:
    """One line naming what the free text resolved to, and what else it could have been."""
    match = resolution.match
    line = f"Matched **{match.name}** (`{match.id}`)."
    if resolution.alternatives:
        others = ", ".join(f"{alt.name} (`{alt.id}`)" for alt in resolution.alternatives)
        line += f" Other candidates: {others}."
    return line


def _shown_line(shown: int, total: int, noun: str) -> str:
    """State how much of the result was returned, so a truncated list is never mistaken for the whole list."""
    if total > shown:
        return f"Showing {shown} of {total} {noun}."
    return f"{total} {noun}."


def _clamp(requested: int, maximum: int) -> int:
    """Hold a caller-supplied row count inside the tool's range."""
    return max(1, min(requested, maximum))


def _number(value: float | None) -> str:
    """Render an API float at three decimals; an absent value stays visibly absent."""
    if value is None:
        return ""
    return f"{value:.3f}"


def _truncate(text: str, limit: int) -> str:
    """Cut long prose at `limit` characters, marking that it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " […]"


async def tool_resolve(text: str, kind: str | None) -> str:
    """Report what Open Targets identifier a name maps to, across one entity kind or all three."""
    kinds = [kind] if kind else ["target", "disease", "drug"]
    sections: list[str] = []
    for entity_kind in kinds:
        hits = await search(text, entity_kind)
        if not hits:
            sections.append(f"**{entity_kind}** — no match.")
            continue
        rows = "\n".join(
            f"| {hit.name} | `{hit.id}` | {_truncate(hit.description or '', 120)} |" for hit in hits
        )
        sections.append(f"**{entity_kind}**\n\n| Name | Id | Description |\n|---|---|---|\n{rows}")
    return f"Lookup for {text!r}:\n\n" + "\n\n".join(sections)


async def tool_target_profile(gene: str) -> str:
    """Identity, function, tractability, safety liabilities, and genetic constraint for one target."""
    resolution = await resolve(gene, "target")
    data = await graphql(queries.TARGET_PROFILE_QUERY, {"id": resolution.match.id})
    target = data.get("target")
    if not target:
        raise OpenTargetsError(f"Open Targets has no target record for `{resolution.match.id}`")

    parts = [_matched_line(resolution), ""]
    parts.append(
        f"**{target.get('approvedSymbol')}** — {target.get('approvedName')} ({target.get('biotype')})"
    )

    descriptions = target.get("functionDescriptions") or []
    if descriptions:
        parts += ["", "**Function**", "", _truncate(descriptions[0], FUNCTION_DESCRIPTION_CHARS)]

    tractable = [t for t in (target.get("tractability") or []) if t.get("value")]
    if tractable:
        by_modality: dict[str, list[str]] = {}
        for entry in tractable:
            by_modality.setdefault(entry["modality"], []).append(entry["label"])
        parts += ["", "**Tractability** (labels where the evidence is present)", ""]
        parts += [f"- {modality}: {', '.join(labels)}" for modality, labels in by_modality.items()]

    constraints = target.get("geneticConstraint") or []
    if constraints:
        rows = "\n".join(
            f"| {c.get('constraintType')} | {_number(c.get('score'))} | {_number(c.get('oe'))} | "
            f"{_number(c.get('oeLower'))}–{_number(c.get('oeUpper'))} |"
            for c in constraints
        )
        parts += [
            "",
            "**Genetic constraint** (observed/expected variant counts; lower means more constrained)",
            "",
            "| Type | Score | o/e | o/e bounds |",
            "|---|---|---|---|",
            rows,
        ]

    liabilities = target.get("safetyLiabilities") or []
    if liabilities:
        parts += ["", f"**Safety liabilities** ({len(liabilities)} recorded)", ""]
        for entry in liabilities:
            effects = ", ".join(
                f"{e.get('direction')}/{e.get('dosing')}" for e in (entry.get("effects") or []) if e
            )
            detail = f" — {effects}" if effects else ""
            parts.append(f"- {entry.get('event')} (source: {entry.get('datasource')}){detail}")
    else:
        parts += ["", "**Safety liabilities** — none recorded."]

    return "\n".join(parts)


async def tool_target_diseases(gene: str, count: int) -> str:
    """Diseases most strongly associated with a target, by Open Targets overall score."""
    size = _clamp(count, MAX_ASSOCIATION_ROWS)
    resolution = await resolve(gene, "target")
    data = await graphql(queries.TARGET_DISEASES_QUERY, {"id": resolution.match.id, "size": size})
    target = data.get("target")
    if not target:
        raise OpenTargetsError(f"Open Targets has no target record for `{resolution.match.id}`")

    block = target.get("associatedDiseases") or {}
    rows = block.get("rows") or []
    if not rows:
        return f"{_matched_line(resolution)}\n\nNo disease associations recorded."

    lines = [
        _matched_line(resolution),
        "",
        _shown_line(len(rows), block.get("count", len(rows)), "associated diseases"),
        "Overall score runs 0–1, aggregated by Open Targets across evidence datatypes; higher means stronger evidence.",
        "",
        "| Disease | Id | Score | Therapeutic areas | Datatypes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        disease = row.get("disease") or {}
        areas = ", ".join(a.get("name", "") for a in (disease.get("therapeuticAreas") or [])[:3])
        datatypes = ", ".join(d.get("id", "") for d in (row.get("datatypeScores") or []))
        lines.append(
            f"| {disease.get('name')} | `{disease.get('id')}` | {_number(row.get('score'))} | {areas} | {datatypes} |"
        )
    return "\n".join(lines)


async def tool_disease_targets(disease: str, count: int) -> str:
    """Targets most strongly associated with a disease, by Open Targets overall score."""
    size = _clamp(count, MAX_ASSOCIATION_ROWS)
    resolution = await resolve(disease, "disease")
    data = await graphql(queries.DISEASE_TARGETS_QUERY, {"id": resolution.match.id, "size": size})
    record = data.get("disease")
    if not record:
        raise OpenTargetsError(f"Open Targets has no disease record for `{resolution.match.id}`")

    block = record.get("associatedTargets") or {}
    rows = block.get("rows") or []
    if not rows:
        return f"{_matched_line(resolution)}\n\nNo target associations recorded."

    lines = [
        _matched_line(resolution),
        "",
        _shown_line(len(rows), block.get("count", len(rows)), "associated targets"),
        "Overall score runs 0–1, aggregated by Open Targets across evidence datatypes; higher means stronger evidence.",
        "",
        "| Target | Ensembl id | Score | Name | Datatypes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        target = row.get("target") or {}
        datatypes = ", ".join(d.get("id", "") for d in (row.get("datatypeScores") or []))
        lines.append(
            f"| {target.get('approvedSymbol')} | `{target.get('id')}` | {_number(row.get('score'))} | "
            f"{target.get('approvedName')} | {datatypes} |"
        )
    return "\n".join(lines)


def _mechanisms(drug: dict) -> str:
    """Join a drug's mechanisms of action into one cell."""
    rows = ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
    seen = list(dict.fromkeys(f"{r.get('mechanismOfAction')}" for r in rows if r.get("mechanismOfAction")))
    return "; ".join(seen)


async def tool_known_drugs(name: str, kind: str, count: int) -> str:
    """Drugs and clinical candidates recorded against a target or a disease, with their highest clinical stage."""
    size = _clamp(count, MAX_DRUG_ROWS)
    resolution = await resolve(name, kind)
    if kind == "target":
        data = await graphql(queries.TARGET_DRUGS_QUERY, {"id": resolution.match.id})
        record = data.get("target")
    else:
        data = await graphql(queries.DISEASE_DRUGS_QUERY, {"id": resolution.match.id})
        record = data.get("disease")
    if not record:
        raise OpenTargetsError(f"Open Targets has no {kind} record for `{resolution.match.id}`")

    block = record.get("drugAndClinicalCandidates") or {}
    all_rows = block.get("rows") or []
    if not all_rows:
        return f"{_matched_line(resolution)}\n\nNo drugs or clinical candidates recorded."

    rows = all_rows[:size]
    header = "| Drug | ChEMBL id | Type | Max stage | Mechanism |"
    separator = "|---|---|---|---|---|"
    if kind == "target":
        header = "| Drug | ChEMBL id | Type | Max stage | Mechanism | Diseases |"
        separator = "|---|---|---|---|---|---|"

    lines = [
        _matched_line(resolution),
        "",
        _shown_line(len(rows), block.get("count", len(all_rows)), "drugs and clinical candidates"),
        "",
        header,
        separator,
    ]
    for row in rows:
        drug = row.get("drug") or {}
        cells = [
            drug.get("name") or "",
            f"`{drug.get('id')}`",
            drug.get("drugType") or "",
            row.get("maxClinicalStage") or "",
            _mechanisms(drug),
        ]
        if kind == "target":
            entries = [d.get("disease") for d in (row.get("diseases") or []) if d.get("disease")]
            named = ", ".join(d.get("name", "") for d in entries[:DRUG_DISEASE_ROWS])
            if len(entries) > DRUG_DISEASE_ROWS:
                named += f" (+{len(entries) - DRUG_DISEASE_ROWS} more)"
            cells.append(named)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def tool_evidence(gene: str, disease: str, count: int) -> str:
    """The evidence rows behind one target-disease pair, with datatype, score, and direction."""
    size = _clamp(count, MAX_EVIDENCE_ROWS)
    target_resolution = await resolve(gene, "target")
    disease_resolution = await resolve(disease, "disease")
    data = await graphql(
        queries.EVIDENCE_QUERY,
        {"id": target_resolution.match.id, "efoIds": [disease_resolution.match.id], "size": size},
    )
    target = data.get("target")
    if not target:
        raise OpenTargetsError(f"Open Targets has no target record for `{target_resolution.match.id}`")

    block = target.get("evidences") or {}
    rows = block.get("rows") or []
    lines = [
        f"Target: {_matched_line(target_resolution)}",
        f"Disease: {_matched_line(disease_resolution)}",
        "",
    ]
    if not rows:
        lines.append("No evidence recorded for this pair.")
        return "\n".join(lines)

    lines += [
        _shown_line(len(rows), block.get("count", len(rows)), "evidence rows"),
        "Direction on target is the change in target activity; direction on trait is its effect on the disease.",
        "",
        "| Datatype | Datasource | Score | On target | On trait | Literature |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        literature = (row.get("literature") or [])[:EVIDENCE_LITERATURE_IDS]
        lines.append(
            f"| {row.get('datatypeId')} | {row.get('datasourceId')} | {_number(row.get('score'))} | "
            f"{row.get('directionOnTarget') or ''} | {row.get('directionOnTrait') or ''} | "
            f"{', '.join(literature)} |"
        )
    return "\n".join(lines)
