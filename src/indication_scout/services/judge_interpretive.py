"""LLM-judged INTERPRETIVE blurb fields for a drug x indication candidate.

The interpretive card fields (blocker, key_risk, verdict, prose) repeatedly CONTRADICTED the
authoritative stage when written in the supervisor's monolithic 8-field blurb pass — e.g. stage
"Phase 3 completed" beside prose "no dedicated Phase 2/3 program". Prompt rules in the big pass
did not hold. The fix (proven this session): an ISOLATED call that is HANDED the already-resolved
facts and asked ONLY to interpret them — it then cannot contradict them.

Validated in:
- scratch/interpretive_fields_harness.py  (11/11, incl. contradicts / terminated / Phase-4-only)
- scratch/staged_blurb_harness.py         (9/9 full chain on real-shaped data)
- scratch/approval_input_harness.py       (6/6 with PRODUCTION-shaped approval inputs, incl.
                                           adversarial enum-vs-fact mismatch + false-approval guard)

INPUTS are the resolved facts (NOT free LLM text):
- stage            : the authoritative dev_stage PHRASE (from judge_dev_stage / _DEV_STAGE_PHRASE)
- active_programs  : the "what is still moving" line (from judge_dev_stage)
- literature       : a deterministic one-liner built from typed EvidenceSummary fields
- relationship     : the approval_relationship Literal (best-effort enum; constrained, so it
                     cannot reintroduce a stage contradiction). Pass the RAW enum value.
- approved_indication : ApprovalCheck.matched_indication (typed FDA fact) or None — the grounding
                     anchor so prose references a real label, not a guess.

Cached per the fact-tuple so a candidate is judged once within the TTL window.

FIELD MAPPING (prompt JSON key -> CandidateBlurb field):
  constraint  -> blocker
  assessment  -> verdict
  key_risk    -> key_risk
  prose       -> prose
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from indication_scout.constants import JUDGMENT_CACHE_TTL
from indication_scout.services.llm import parse_last_json_object, query_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_INTERP_PROMPT = """You are a drug-repurposing analyst writing four interpretive fields for a \
candidate card. You are GIVEN the authoritative facts below — already decided. Interpret them; \
do NOT restate, re-derive, or contradict them.

AUTHORITATIVE FACTS (ground truth):
- Development stage: {stage}
- Active programs (what is still moving): {active_programs}
- Relevance-reviewed trial evidence: {trial_evidence}
- Literature: {literature}
- Approval relationship (how this candidate relates to an approved use): {relationship}
- Drug's approved indication that this relates to: {approved_indication}
- Closure verdict (whether the signal is still live): {closure}
- Late-stage terminations: {terminations}

The closure verdict comes from the combined trial and literature review. When it says CLOSED, say so
in the assessment and prose; never call a closed signal open, uncertain, or still to be resolved.
When it says NOT DECIDED, no closure judgement was reached — never call the signal closed on that
basis. Absence of trials is not closure; a signal closes on a negative result, not on never having
been tested.

A late-stage termination for safety or benefit:risk is a closure signal even when a completed
Phase 3 also exists — the development stage names the furthest trial reached, not the fate of the
programme. When terminations are reported above, the prose and key_risk must reflect them rather
than reading the stage alone.

When the reviewed evidence reports one or more relevant trials, the hypothesis WAS studied. Do not
call it untested or abandoned. Registry query matches are search coverage, not supporting trials.
"None active" means no trial is currently recruiting — state that as the status; do not infer a \
cause (sponsor disengagement, commercial failure, abandonment) you were not given.

When the reviewed evidence reports ZERO relevant trials, that means no registered trial matched \
this pair — it does NOT mean the drug was never studied in humans for this disease. If the \
literature says "RCT-backed / controlled" or "observational", human studies exist: describe the \
registry record as empty and never write "untested in humans", "no clinical development has been \
initiated", "no clinical testing", or "entirely unvalidated". Reserve an untested verdict for \
candidates whose literature is "animal/in-vitro only".

The drug is NOT approved for THIS candidate indication unless the approved indication above \
exactly names it. If the approved indication is "none", do not claim any approval for this use.

Write four fields:
- constraint: what (if anything) holds this back (regulatory / commercial / evidence gap), or the \
current status if nothing clearly does. One short line. Do not invent a blocker; if the stage is \
"status unknown", say the status is unconfirmed rather than asserting a gap. If a Phase 3 is \
completed or active, do NOT write "no Phase 3" or "no dedicated development program".
- key_risk: the single biggest risk to the hypothesis. One short line. Phase-free.
- assessment: a short interpretive verdict tag. Choose one that fits the facts — e.g. "Live but \
bottlenecked", "Maturing, awaiting readout", "Tested, status unconfirmed", "Stalled, regulatory \
gap", "Untested at scale", "Studied outside the registry", "Closed signal". When trials exist but \
their status is unknown, prefer a neutral tag (e.g. "Tested, status unconfirmed") over a decline \
tag. When no registered trial matched but the literature reports human studies, prefer "Studied \
outside the registry" over "Untested at scale". Do NOT name a phase tier.
- prose: EXACTLY 2 sentences interpreting the state of the hypothesis, consistent with the stage \
and active programs above. If the literature direction is "contradicts", surface that the drug \
failed / was disproven. Do NOT name a phase tier that disagrees with the stage.

Respond with the JSON object and NOTHING else — no reasoning, no preamble, no prose outside \
the object: \
{{"constraint":"...","key_risk":"...","assessment":"...","prose":"..."}}"""


@dataclass(frozen=True)
class InterpretiveJudgment:
    """The isolated interpretation of a candidate's resolved facts. Field names match the
    CandidateBlurb schema (the prompt's `constraint`->`blocker`, `assessment`->`verdict`).
    """

    blocker: str
    key_risk: str
    verdict: str
    prose: str


def _parse_interpretive(text: str) -> InterpretiveJudgment | None:
    """Extract the four interpretive fields from the LLM JSON response, mapping the prompt's
    JSON keys to CandidateBlurb field names. None on parse failure (caller leaves fields empty —
    never fabricates)."""
    # The model sometimes reasons in prose before emitting the object; requiring the whole
    # response to BE the object discarded a valid judgment and left the fields empty.
    data = parse_last_json_object(text)
    if data is None:
        return None

    def _s(key: str) -> str:
        v = data.get(key)
        return v.strip() if isinstance(v, str) else ""

    return InterpretiveJudgment(
        blocker=_s("constraint"),
        key_risk=_s("key_risk"),
        verdict=_s("assessment"),
        prose=_s("prose"),
    )


# Plain-language phrasing for each approval-relationship label fed to the judge LLM. The raw
# label words ("contaminated"/"combination_only") are INTERNAL verbiage and must never reach the
# report — the judge can echo whatever it is given into prose, so translate before passing.
_RELATIONSHIP_PHRASE: dict[str, str] = {
    "contaminated": (
        "this candidate is a broader/related disease that overlaps an approved related "
        "indication, so its trial record cannot be cleanly separated from the approved use"
    ),
    "combination_only": (
        "the drug is approved for this disease only as part of a combination product, not as "
        "monotherapy"
    ),
    "none": "no notable relationship to an approved indication",
}


async def judge_interpretive(
    *,
    stage: str,
    active_programs: str,
    literature: str,
    relationship: str,
    approved_indication: str | None,
    trial_evidence: str,
    closure: str,
    terminations: str,
    cache_dir: Path,
    drug: str = "",
    indication: str = "",
) -> InterpretiveJudgment | None:
    """Return the interpretive blurb fields synthesized from the RESOLVED facts, or None on a
    parse failure (the caller then leaves the fields empty — one source of truth, no fabrication).

    Cached per the fact-tuple (stage, active_programs, literature, relationship,
    approved_indication, trial_evidence, closure, terminations) so a candidate is judged once within
    the TTL window. `closure` is the clinical-trials agent's typed live/closed verdict, rendered as a
    sentence — without it the writer judged closure itself and contradicted the trial section.
    `terminations` names any late-stage stop for safety or efficacy, which the development stage
    hides whenever a completed Phase 3 also exists.
    `relationship` is the upstream FDA label (e.g. "contaminated" / "combination_only" / "none")
    — not prose. `trial_evidence` states the relevance-reviewed count and coverage limit.
    """
    approved = approved_indication or "none"
    cache_params = {
        "drug": drug,
        "indication": indication,
        "stage": stage,
        "active_programs": active_programs,
        "literature": literature,
        "relationship": relationship or "none",
        "approved_indication": approved,
        "trial_evidence": trial_evidence,
        "closure": closure,
        "terminations": terminations,
    }
    cached = cache_get("interpretive", cache_params, cache_dir)
    if isinstance(cached, dict):
        return InterpretiveJudgment(
            blocker=cached.get("blocker", ""),
            key_risk=cached.get("key_risk", ""),
            verdict=cached.get("verdict", ""),
            prose=cached.get("prose", ""),
        )

    # Translate the internal label to plain language before the LLM sees it (the raw label word
    # must not appear in report prose). Cache key above keeps the raw label for stability.
    relationship_phrase = _RELATIONSHIP_PHRASE.get(
        relationship or "none", _RELATIONSHIP_PHRASE["none"]
    )
    prompt = _INTERP_PROMPT.format(
        stage=stage,
        active_programs=active_programs,
        literature=literature,
        relationship=relationship_phrase,
        approved_indication=approved,
        trial_evidence=trial_evidence,
        closure=closure,
        terminations=terminations,
    )
    response = await query_llm(prompt)
    judgment = _parse_interpretive(response)
    if judgment is None:
        logger.warning(
            "judge_interpretive: could not parse a valid response for %s x %s; leaving "
            "interpretive fields empty. Response was: %s",
            drug,
            indication,
            response,
        )
        return None

    cache_set(
        "interpretive",
        cache_params,
        {
            "blocker": judgment.blocker,
            "key_risk": judgment.key_risk,
            "verdict": judgment.verdict,
            "prose": judgment.prose,
        },
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return judgment
