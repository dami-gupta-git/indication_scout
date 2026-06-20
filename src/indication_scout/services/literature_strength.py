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

_BASIS_VALUES = ("drug_specific", "approved", "class_level", "none")
_STRENGTH_VALUES = ("strong", "moderate", "weak", "none")
_DIRECTION_VALUES = ("supports", "contradicts", "mixed", "none")

_LITERATURE_STRENGTH_PROMPT = """You are a biomedical evidence analyst. Grade the literature \
evidence for repurposing ONE drug to treat ONE disease, using ONLY the abstracts below.

Drug: {drug}
Disease: {disease}
Drug's FDA-approved indications: {approved_indications}

CRITICAL — strength and direction grade evidence for THIS EXACT DRUG only:
- An abstract about a DIFFERENT drug — even one in the same mechanistic class (e.g. another GLP-1
  receptor agonist) — is NOT direct evidence for this drug. It is class-level context.
- An abstract about this drug but a DIFFERENT disease is NOT relevant evidence for this pair.
- THERAPEUTIC INTENT: evidence that the drug was studied IN patients who HAVE this disease but
  FOR a different condition (a comorbidity, smoking cessation, weight loss, etc.) is NOT evidence
  that the drug TREATS this disease. It does not support the repurposing hypothesis. Set
  evidence_basis="none" in that case (there is no relevant evidence for treating this disease),
  even though the abstracts are about this drug and mention this disease.
- APPROVED-INDICATION EXCLUSION (direction matters): this fires ONLY when the CANDIDATE is a BROAD
  term that CONTAINS an approved indication as a narrower part (e.g. candidate "mood disorder"
  contains approved "major depressive disorder"; candidate "NAFLD" contains approved "MASH"). In
  that case an abstract studying THIS DRUG for the APPROVED sub-indication (the narrower part) is
  ALREADY-APPROVED evidence — do NOT count it; only abstracts about the candidate's NON-approved
  scope count. DO NOT fire it in the OPPOSITE direction: if the candidate is NARROWER than (a child
  of) an approved indication — i.e. the candidate itself is NOT approved but its broader PARENT is
  (e.g. candidate "diabetic nephropathy"/"diabetic kidney disease" when only the broad parent
  "chronic kidney disease" is approved) — then abstracts about the broad approved PARENT ARE the
  candidate's own evidence (the candidate's patients are studied within the parent's trials), so
  COUNT them normally. Test: exclude an abstract only when its subject is the approved indication
  itself or NARROWER than it AND narrower-than-or-equal-to the candidate; never exclude an abstract
  about a disease BROADER than the candidate.
  SEVERITY/STAGE QUALIFIER: ignore a severity/stage/biomarker QUALIFIER on the approved indication
  when matching — it does NOT create a separable disease. If the approval is "X with
  <severity/stage>" (e.g. "MASH with moderate-to-advanced fibrosis"), an abstract about the bare
  disease X or its near-synonyms (e.g. "NASH"/"steatohepatitis") IS the approved sub-indication and
  is EXCLUDED. Example: approved "MASH (NASH) with fibrosis"; candidate "NAFLD" -> abstracts on
  "NASH"/"MASH"/"steatohepatitis" are EXCLUDED (approved); only abstracts on the broad
  NAFLD/simple-steatosis spectrum count as repurposing evidence.
  SIBLINGS of an approved indication (e.g. type 1 diabetes vs approved type 2 diabetes), and
  evidence BROADER than a MINORITY-BIOMARKER approval (e.g. all-comers NSCLC vs approved
  EGFR-mutated NSCLC, biomarker ~10-15%) are NOT approved sub-indications — they count as
  repurposing evidence. (A severity grade is NOT a minority biomarker: "NASH" is the approved
  disease, not a broad parent of it.)
- evidence_basis:
  - "drug_specific": at least one abstract reports clinical/preclinical evidence for THIS drug
    used to TREAT THIS disease in the candidate's NON-approved scope.
  - "approved": the only relevant this-drug evidence studies an APPROVED sub-indication of the
    broad candidate; there is no repurposing evidence for the candidate's uncovered scope.
  - "class_level": the disease-relevant evidence is for OTHER drugs in the class; there is no
    direct evidence for this drug in this disease.
  - "none": no relevant evidence for treating this disease with this drug — neither drug-specific
    nor class-level, OR the only this-drug evidence is for a different condition in this
    population (the therapeutic-intent case above).
- strength grades DRUG-SPECIFIC evidence quantity/quality only:
  - "strong": multiple drug-specific clinical studies (RCTs, large cohorts) for THIS drug in THIS
    disease. NEVER "strong" when evidence_basis is "approved", "class_level", or "none".
  - "moderate": small drug-specific clinical studies, case series, or strong drug-specific
    preclinical data.
  - "weak": drug-specific case reports only, or drug-specific in-vitro/animal data only.
  - "none": no drug-specific evidence (set this whenever evidence_basis != "drug_specific").
- direction (of the drug-specific evidence): "supports" | "contradicts" | "mixed" | "none". When
  evidence_basis != "drug_specific", direction is "none".
- is_observational: true if the relevant drug-specific clinical evidence is exclusively
  observational; false if at least one drug-specific RCT/controlled trial; null if no relevant
  drug-specific clinical evidence.

PER-ABSTRACT RELEVANCE (do this FIRST, then grade): classify EACH abstract by PMID as
"relevant" or "contaminated", applying the rules above PER ABSTRACT:
- this drug, treating this disease in the candidate's NON-approved scope -> relevant
- this drug for an APPROVED sub-indication of a BROAD candidate (the narrower part) -> contaminated
- this drug for the broad approved PARENT of a NARROWER candidate (CKD abstract under a DKD
  candidate) -> relevant (the candidate's own evidence — the parent-direction rule above)
- a DIFFERENT drug (same class or not) -> contaminated for the per-PMID split, but note class
  context: if the ONLY disease-relevant abstracts are other-drug, set evidence_basis="class_level"
- therapeutic-intent mismatch (this drug in these patients but FOR another condition) -> contaminated
Then grade strength / direction / is_observational / evidence_basis over the RELEVANT abstracts
ONLY. If NO abstract is relevant, evidence_basis is "approved" (only approved sub-indication
evidence), "class_level" (only other-drug evidence), or "none"; strength/direction are "none".

Abstracts:
{abstracts}

Respond with ONLY a JSON object. "verdicts" maps EVERY PMID above to its per-abstract verdict;
the strength fields describe the RELEVANT set only:
{{"verdicts": {{"<pmid>": "relevant"|"contaminated", ...}}, \
"evidence_basis": "drug_specific"|"approved"|"class_level"|"none", \
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
    DRUG, which way it points, and whether it is drug-specific or only class-level.

    relevant_pmids / contaminated_pmids are the per-abstract split (mirrors the trial gate's
    relevant_ncts / contaminated_ncts). strength/direction/basis are graded over the RELEVANT
    set only. Stored as tuples so the dataclass stays frozen/hashable.
    """

    strength: Literal["strong", "moderate", "weak", "none"]
    direction: Literal["supports", "contradicts", "mixed", "none"]
    evidence_basis: Literal["drug_specific", "approved", "class_level", "none"]
    is_observational: bool | None
    relevant_pmids: tuple[str, ...] = ()
    contaminated_pmids: tuple[str, ...] = ()


def _extract_json_object(text: str) -> dict | None:
    """Return the JSON object from an LLM response, or None.

    Tolerant of (a) ```json fences and (b) reasoning prose emitted BEFORE the JSON — the model
    sometimes explains its grade first, which broke the old fence-split (it grabbed the prose
    segment). We try a direct parse, then the fenced block, then the LAST balanced {...} block.
    """
    stripped = text.strip()
    # 1) direct
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # 2) ```json ... ``` fence (take the segment AFTER a json fence marker if present)
    if "```" in stripped:
        parts = stripped.split("```")
        for seg in parts:
            seg = seg.strip()
            if seg.lower().startswith("json"):
                seg = seg[4:].strip()
            try:
                obj = json.loads(seg)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # 3) last balanced {...} block anywhere in the text
    end = stripped.rfind("}")
    if end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if stripped[i] == "}":
                depth += 1
            elif stripped[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[i : end + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def _parse_strength(
    text: str, input_pmids: list[str] | None = None
) -> LiteratureStrength | None:
    """Extract the judgment from the LLM JSON. None on parse failure or an out-of-enum value
    (caller keeps the synthesize values as fallback — never fabricates a stronger grade).

    `input_pmids` is the set of PMIDs that were sent to the model. The per-abstract `verdicts`
    map is parsed into relevant_pmids / contaminated_pmids over exactly that set: any input PMID
    the model OMITS is treated as contaminated (conservative — an unclassified abstract never
    counts as evidence), and any verdict for a PMID that was not sent is ignored.
    """
    data = _extract_json_object(text)
    if data is None:
        return None
    try:
        basis = data.get("evidence_basis")
        strength = data.get("strength")
        direction = data.get("direction")
        is_obs = data.get("is_observational")
        verdicts = data.get("verdicts")
    except AttributeError:
        return None
    if (
        basis not in _BASIS_VALUES
        or strength not in _STRENGTH_VALUES
        or direction not in _DIRECTION_VALUES
    ):
        return None
    if not isinstance(is_obs, bool) and is_obs is not None:
        return None
    if not isinstance(verdicts, dict):
        return None

    # Build the per-PMID split over the INPUT set only. Omitted PMID -> contaminated.
    pmids = [str(p) for p in (input_pmids or [])]
    relevant: list[str] = []
    contaminated: list[str] = []
    for pmid in pmids:
        v = verdicts.get(pmid)
        if v not in ("relevant", "contaminated"):
            # missing or invalid verdict for a shown abstract -> conservative: contaminated
            contaminated.append(pmid)
        elif v == "relevant":
            relevant.append(pmid)
        else:
            contaminated.append(pmid)

    # An empty relevant set cannot be drug_specific (nothing this-drug-this-disease survived).
    # Reject so the caller keeps the synthesize values rather than rendering an inconsistent card.
    if not relevant and basis == "drug_specific":
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
        relevant_pmids=tuple(relevant),
        contaminated_pmids=tuple(contaminated),
    )


async def judge_literature_strength(
    abstracts: list[dict],
    *,
    drug: str,
    indication: str,
    cache_dir: Path,
    approved_indications: list[str] | None = None,
) -> LiteratureStrength | None:
    """Return the DRUG-SPECIFIC strength/direction/basis for the abstract set, or None.

    `abstracts` are the same top abstracts synthesize sees (dicts with pmid/title/abstract).
    `approved_indications` is the drug's FDA-approved indication list; a paper about an APPROVED
    sub-indication of a broad candidate is graded evidence_basis="approved" (excluded from
    repurposing strength), not drug_specific. Empty/None → the exclusion never fires.
    Returns None when there are no abstracts, or on a parse failure (the caller keeps the
    existing synthesize values rather than fabricating). Cached on the sorted PMID set + drug +
    indication + sorted approved set (the same PMIDs grade differently under different approved
    lists, so the approved set MUST be in the key) under JUDGMENT_CACHE_TTL.
    """
    if not abstracts:
        return None

    approved = sorted({i.strip() for i in (approved_indications or []) if i.strip()})
    pmids = sorted(str(a.get("pmid", "")) for a in abstracts)
    cache_params = {
        "drug": drug,
        "indication": indication,
        "pmids": pmids,
        "approved_indications": approved,
    }
    cached = cache_get("literature_strength", cache_params, cache_dir)
    if isinstance(cached, dict) and cached.get("evidence_basis") in _BASIS_VALUES:
        return LiteratureStrength(
            strength=cached.get("strength", "none"),
            direction=cached.get("direction", "none"),
            evidence_basis=cached["evidence_basis"],
            is_observational=cached.get("is_observational"),
            relevant_pmids=tuple(cached.get("relevant_pmids", [])),
            contaminated_pmids=tuple(cached.get("contaminated_pmids", [])),
        )

    prompt = _LITERATURE_STRENGTH_PROMPT.format(
        drug=drug,
        disease=indication,
        approved_indications=", ".join(approved) if approved else "(none)",
        abstracts=_format_abstracts(abstracts),
    )
    response = await query_llm(prompt)
    judgment = _parse_strength(response, input_pmids=pmids)
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
            "relevant_pmids": list(judgment.relevant_pmids),
            "contaminated_pmids": list(judgment.contaminated_pmids),
        },
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return judgment
