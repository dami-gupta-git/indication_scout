"""Drug-specific literature strength — an isolated LLM judgment that grades the evidence for
THIS EXACT DRUG, so a class-level (other-drug) RCT body never inflates the card to "strong".

The bug this fixes (snapshot semaglutide_2026-06-14_19-41-15.md, Parkinson): synthesize set
strength="strong" while its own prose said "no direct clinical evidence for semaglutide in
Parkinson's disease" — the strong RCTs were lixisenatide / exenatide / NLY01 (same GLP-1 class),
and the one semaglutide abstract was for depression. Strength must grade THIS drug's evidence;
class-level evidence is surfaced as evidence_basis="class_level", never as drug strength.

Proven in scratch/literature_strength_harness.py (7/7 on Sonnet over REAL abstracts across three
drugs: the Parkinson class-level bug, two cross-drug class-level traps (erlotinib fed gefitinib
RCTs, tadalafil fed sildenafil RCTs), and four genuine drug-specific sets including a
drug-specific negative — none over- or under-corrected).

Mirrors services/dev_stage.py: prompt const, parse, cache, frozen dataclass, async fn. The same
top abstracts synthesize sees are sent; cached on the sorted PMID set + drug + indication.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from indication_scout.constants import JUDGMENT_CACHE_TTL
from indication_scout.services.llm import query_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_BASIS_VALUES = ("drug_specific", "class_level", "none")
_STRENGTH_VALUES = ("strong", "moderate", "weak", "none")
_DIRECTION_VALUES = ("supports", "contradicts", "mixed", "none")

_LITERATURE_STRENGTH_PROMPT = """You are a biomedical evidence analyst. Grade the literature \
evidence for repurposing ONE drug to treat ONE disease, using ONLY the abstracts below.

Drug: {drug}
Disease: {disease}

CRITICAL — strength and direction grade evidence for THIS EXACT DRUG only:
- An abstract about a DIFFERENT drug — even one in the same mechanistic class (e.g. another GLP-1
  receptor agonist) — is NOT direct evidence for this drug. It is class-level context.
- An abstract about this drug but a DIFFERENT disease is NOT relevant evidence for this pair.
- evidence_basis:
  - "drug_specific": at least one abstract reports clinical/preclinical evidence for THIS drug in
    THIS disease.
  - "class_level": the disease-relevant evidence is for OTHER drugs in the class; there is no
    direct evidence for this drug in this disease.
  - "none": no relevant evidence at all (neither drug-specific nor class-level for this disease).
- strength grades DRUG-SPECIFIC evidence quantity/quality only:
  - "strong": multiple drug-specific clinical studies (RCTs, large cohorts) for THIS drug in THIS
    disease. NEVER "strong" when evidence_basis is "class_level" or "none".
  - "moderate": small drug-specific clinical studies, case series, or strong drug-specific
    preclinical data.
  - "weak": drug-specific case reports only, or drug-specific in-vitro/animal data only.
  - "none": no drug-specific evidence (set this whenever evidence_basis != "drug_specific").
- direction (of the drug-specific evidence): "supports" | "contradicts" | "mixed" | "none". When
  evidence_basis != "drug_specific", direction is "none".
- is_observational: true if the relevant drug-specific clinical evidence is exclusively
  observational; false if at least one drug-specific RCT/controlled trial; null if no relevant
  drug-specific clinical evidence.

Abstracts:
{abstracts}

Respond with ONLY a JSON object:
{{"evidence_basis": "drug_specific"|"class_level"|"none", \
"strength": "strong"|"moderate"|"weak"|"none", \
"direction": "supports"|"contradicts"|"mixed"|"none", \
"is_observational": true|false|null, "reason": "<one short sentence>"}}"""


def _format_abstracts(abstracts: list[dict]) -> str:
    """Format abstracts as PMID/Title/Abstract blocks — same shape synthesize sends.

    Each dict has keys "pmid", "title", "abstract".
    """
    return "\n\n".join(
        f"PMID: {a.get('pmid', '')}\nTitle: {a.get('title', '')}\n"
        f"Abstract: {a.get('abstract', '')}"
        for a in abstracts
    )


@dataclass(frozen=True)
class LiteratureStrength:
    """The isolated drug-specific read of an abstract set: how strong the evidence is FOR THIS
    DRUG, which way it points, and whether it is drug-specific or only class-level."""

    strength: Literal["strong", "moderate", "weak", "none"]
    direction: Literal["supports", "contradicts", "mixed", "none"]
    evidence_basis: Literal["drug_specific", "class_level", "none"]
    is_observational: bool | None


def _parse_strength(text: str) -> LiteratureStrength | None:
    """Extract the judgment from the LLM JSON. None on parse failure or an out-of-enum value
    (caller keeps the synthesize values as fallback — never fabricates a stronger grade).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
    try:
        data = json.loads(stripped)
        basis = data.get("evidence_basis")
        strength = data.get("strength")
        direction = data.get("direction")
        is_obs = data.get("is_observational")
    except (json.JSONDecodeError, AttributeError):
        return None
    if (
        basis not in _BASIS_VALUES
        or strength not in _STRENGTH_VALUES
        or direction not in _DIRECTION_VALUES
    ):
        return None
    if not isinstance(is_obs, bool) and is_obs is not None:
        return None
    # ENFORCE the invariant the prompt states: only drug_specific evidence carries a
    # strength/direction. The model is told to set strength="none"/direction="none" whenever
    # basis != "drug_specific", but don't trust it — every consumer (the supervisor ranking path
    # reads es.strength directly) depends on this, so force it here. Otherwise a class_level pair
    # with a stray strength="moderate" would rank as if it had direct drug evidence.
    if basis != "drug_specific":
        strength = "none"
        direction = "none"
    return LiteratureStrength(
        strength=strength,
        direction=direction,
        evidence_basis=basis,
        is_observational=is_obs,
    )


async def judge_literature_strength(
    abstracts: list[dict],
    *,
    drug: str,
    indication: str,
    cache_dir: Path,
) -> LiteratureStrength | None:
    """Return the DRUG-SPECIFIC strength/direction/basis for the abstract set, or None.

    `abstracts` are the same top abstracts synthesize sees (dicts with pmid/title/abstract).
    Returns None when there are no abstracts, or on a parse failure (the caller keeps the
    existing synthesize values rather than fabricating). Cached on the sorted PMID set + drug +
    indication (PMIDs are stable; mirrors synthesize's own cache key) under JUDGMENT_CACHE_TTL.
    """
    if not abstracts:
        return None

    pmids = sorted(str(a.get("pmid", "")) for a in abstracts)
    cache_params = {"drug": drug, "indication": indication, "pmids": pmids}
    cached = cache_get("literature_strength", cache_params, cache_dir)
    if isinstance(cached, dict) and cached.get("evidence_basis") in _BASIS_VALUES:
        return LiteratureStrength(
            strength=cached.get("strength", "none"),
            direction=cached.get("direction", "none"),
            evidence_basis=cached["evidence_basis"],
            is_observational=cached.get("is_observational"),
        )

    prompt = _LITERATURE_STRENGTH_PROMPT.format(
        drug=drug, disease=indication, abstracts=_format_abstracts(abstracts)
    )
    response = await query_llm(prompt)
    judgment = _parse_strength(response)
    if judgment is None:
        logger.warning(
            "judge_literature_strength: could not parse a valid judgment for %s x %s; "
            "keeping the synthesize values. Response was: %s",
            drug,
            indication,
            response,
        )
        return None

    cache_set(
        "literature_strength",
        cache_params,
        {
            "strength": judgment.strength,
            "direction": judgment.direction,
            "evidence_basis": judgment.evidence_basis,
            "is_observational": judgment.is_observational,
        },
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return judgment
