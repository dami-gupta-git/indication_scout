"""Hierarchical LLM dedup pass for candidate diseases.

`merge_and_dedup()` calls `run_hierarchical_dedup` after the deterministic exact-match passes (ID,
name, OT name-resolve). The LLM sees the full merged candidate list and identifies super/subtype
pairs the exact-match passes can't catch (e.g. "ulcerative colitis" ⊂ "inflammatory bowel
disease"; "type 2 diabetes mellitus" ⊂ "diabetes mellitus"), picking ONE survivor per pair by
which level is most actionable for the drug's mechanism. On failure (LLM error, unparseable JSON,
unknown survivors) it logs a WARNING and yields zero decisions, so the caller keeps all candidates
(error by omission, not inaccuracy).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, model_validator

from indication_scout.services.llm import parse_last_json_object, query_llm

logger = logging.getLogger(__name__)


class HierarchyDecision(BaseModel):
    """One survivor + the entries it subsumes (super- or subtypes)."""

    survivor: str = ""
    dropped: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None:
                if field_info.default_factory is not None:
                    values[field_name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[field_name] = field_info.default
        return values


class HierarchyDedupOutput(BaseModel):
    """Decisions returned by the hierarchical LLM pass."""

    decisions: list[HierarchyDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None:
                if field_info.default_factory is not None:
                    values[field_name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[field_name] = field_info.default
        return values


async def run_hierarchical_dedup(
    drug_name: str,
    mechanism_targets: list[tuple[str, str]],
    candidates: list[tuple[str, str, str | None]],
) -> HierarchyDedupOutput:
    """Ask an LLM to identify hierarchical (super/subtype) overlaps among candidates.

    Args:
        drug_name: the drug under analysis (for MoA-aware survivor selection).
        mechanism_targets: list of (gene, action_type) pairs from the mechanism agent.
        candidates: full merged list of (canonical_name, source, efo_id_or_None);
            source ∈ {"competitor", "mechanism", "both"}.

    Returns:
        HierarchyDedupOutput. On any failure (LLM/parse error, missing fields), returns
        `decisions=[]` with a WARNING logged; the caller then keeps every candidate.

    Idempotence guards applied here (not the caller):
        - Decisions naming a survivor not in `candidates` are dropped.
        - In `dropped`, the survivor's own name and any unknown name are filtered out; the
          decision is kept only if at least one valid drop remains.
        - A name in `dropped` across multiple decisions is still removed once at apply time
          (caller responsibility).
    """
    if len(candidates) < 2:
        return HierarchyDedupOutput(decisions=[])

    prompt = _build_prompt(drug_name, mechanism_targets, candidates)
    try:
        raw = await query_llm(prompt)
    except Exception as e:
        logger.warning(
            "hierarchical dedup LLM call failed for drug=%r (%d candidates): %r — "
            "keeping all candidates",
            drug_name,
            len(candidates),
            e,
        )
        return HierarchyDedupOutput(decisions=[])

    parsed = parse_last_json_object(raw)
    if parsed is None:
        logger.warning(
            "hierarchical dedup returned unparseable JSON for drug=%r: %r — "
            "keeping all candidates",
            drug_name,
            raw[:300],
        )
        return HierarchyDedupOutput(decisions=[])

    decisions_raw = parsed.get("decisions")
    if not isinstance(decisions_raw, list):
        logger.warning(
            "hierarchical dedup JSON missing 'decisions' list for drug=%r: %r — "
            "keeping all candidates",
            drug_name,
            parsed,
        )
        return HierarchyDedupOutput(decisions=[])

    known_names = {name for (name, _src, _eid) in candidates}
    cleaned: list[HierarchyDecision] = []
    for item in decisions_raw:
        if not isinstance(item, dict):
            continue
        survivor = (item.get("survivor") or "").strip()
        dropped_raw = item.get("dropped") or []
        reason = (item.get("reason") or "").strip()

        if survivor not in known_names:
            logger.warning(
                "hierarchical dedup names unknown survivor %r for drug=%r — "
                "ignoring decision",
                survivor,
                drug_name,
            )
            continue
        if not isinstance(dropped_raw, list):
            continue

        dropped = [
            d.strip()
            for d in dropped_raw
            if isinstance(d, str) and d.strip() in known_names and d.strip() != survivor
        ]
        if not dropped:
            continue

        cleaned.append(
            HierarchyDecision(survivor=survivor, dropped=dropped, reason=reason)
        )

    return HierarchyDedupOutput(decisions=cleaned)


def _build_prompt(
    drug_name: str,
    mechanism_targets: list[tuple[str, str]],
    candidates: list[tuple[str, str, str | None]],
) -> str:
    """Build the hierarchical-dedup prompt."""
    target_str = (
        ", ".join(f"{g} ({a})" for g, a in mechanism_targets)
        if mechanism_targets
        else "(no mechanism targets recorded)"
    )
    rows = "\n".join(
        f"- name: {name!r}, source: {source}, efo_id: {efo_id or '(unknown)'}"
        for (name, source, efo_id) in candidates
    )
    return f"""Task: identify hierarchical (super/subtype) overlaps among the candidate
diseases below and decide which single entry to KEEP per overlap. Return
decisions so a downstream filter can drop the rest.

Drug under analysis: {drug_name!r}
Drug mechanism targets: {target_str}

Candidate list (each line is one entry):
{rows}

Rules:
- A "hierarchical overlap" is any pair where one entry is a clinical SUBTYPE of
  another (e.g. ulcerative colitis ⊂ inflammatory bowel disease; type 2
  diabetes mellitus ⊂ diabetes mellitus; non-alcoholic steatohepatitis ⊂
  non-alcoholic fatty liver disease).
- Synonyms (NAFLD ↔ non-alcoholic fatty liver disease) are also overlaps and
  should produce one decision picking either spelling.
- DO NOT group sibling diseases (two distinct subtypes of the same parent) —
  only true super/subtype relationships.
- HARD RULE: every entry in a decision's `dropped` list MUST be either a
  super/subtype or a synonym of that decision's `survivor`. Never put a
  sibling, cousin, or otherwise unrelated entry in `dropped`, even if the
  entry seems clinically irrelevant to the drug's mechanism. Clinical
  relevance is NOT your concern here — this step ONLY removes redundant
  hierarchical overlaps. A different downstream step filters by relevance.
  Example: with candidates {{diabetes mellitus, type 2 diabetes mellitus,
  type 1 diabetes mellitus}}, the ONLY valid decision is
  {{survivor: "type 2 diabetes mellitus" OR "diabetes mellitus",
   dropped: [the other one]}}. T1DM is a SIBLING of T2DM and MUST be left
  alone, regardless of how poorly it fits the drug.
- For each overlap, pick the ONE survivor. DEFAULT: prefer the SUBTYPE — it is
  more actionable as a repurposing candidate than its parent. The parent only
  wins when the subtype is itself an approved/standard indication for this
  drug (so dropping it removes a non-opportunity) AND the parent has a
  clinically distinct uncovered population not subsumed by the subtype.
  Heuristics:
    * Default: keep the subtype, drop the parent. Broad parent terms like
      "metabolic disease", "cancer", "autoimmune disease" are rarely useful
      as repurposing candidates and should lose to any specific child.
    * Exception: if the subtype IS the drug's approved indication (e.g.
      type 2 diabetes for metformin) AND the parent encompasses additional
      uncovered populations, prefer the parent.
    * For synonyms (NAFLD ↔ non-alcoholic fatty liver disease), either
      spelling is fine — pick the more clinically standard form.
    * Source tag is informative but not decisive: "both" entries are slightly
      preferred (they had independent support) but can still be dropped if
      another entry is the better match.
- One decision per overlap. Do not produce decisions that don't drop anyone.
- Use the EXACT canonical names shown in the candidate list above (do not
  reword, abbreviate, or add/remove articles).

Return a single JSON object:

{{"decisions": [
  {{"survivor": "<canonical name>",
    "dropped": ["<canonical name>", ...],
    "reason": "<short clinical reason, log-only>"}}
]}}

If there are NO hierarchical overlaps, return {{"decisions": []}}.

Return only the JSON object. Do not include any other text.
"""
