"""LLM-authored Clinical Trials section prose for a drug x indication pair, FED the already-
resolved development stage so it cannot contradict it.

The bug this fixes: the in-loop sub-agent wrote the trial-section prose BEFORE the authoritative
dev_stage was resolved, so the prose judged the tier on its own and contradicted the card —
e.g. card "Phase 3 completed" while the prose said "no completed Phase 3 specifically for T1DM"
(a completed Phase 2/Phase 3 counts). The fix is to author the prose AFTER the stage is resolved
and FEED it that stage. Proven in scratch/ct_summary_harness.py (7/7 on Sonnet: the T1DM crux,
cocaine-shape Phase 2/3, terminated-for-safety closure, old-generic-no-approval, operational
stops).

The summary DESCRIBES the relevant trials and judges CLOSURE (live vs closed) only — it does NOT
re-judge the tier (the stage is given) and says NOTHING about approval status (the supervisor
owns all approval framing). `first_approval` is fed only for the "old drug, no-approval is not
closure" rule. Cached per the fact-tuple so a pair is summarized once within the TTL window.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from indication_scout.constants import JUDGMENT_CACHE_TTL
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import query_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_CLOSURE_VALUES = ("live", "closed", "unknown")

_CT_SUMMARY_PROMPT = """You are a clinical development analyst writing the Clinical Trials \
section for one drug x indication pair.

The development STAGE has ALREADY been determined authoritatively and is GIVEN to you below as
ground truth. Your job is to DESCRIBE the trials and judge CLOSURE (is the pair still live or
closed). You must NOT re-judge or contradict the stage.

GIVEN (ground truth — do not contradict):
- stage: {stage}
- active_programs: {active_programs}
- first_approval (year the drug was first approved anywhere; "unknown" if not known): \
{first_approval}

Hard rules on the prose:
- The stage is authoritative. NEVER write a phrase that re-judges or contradicts it — e.g. "no
  completed Phase 3", "no dedicated Phase 3 program", "no pivotal trial on record", "the program
  is exploratory / Phase 4 only" — when the given stage says otherwise.
- A completed "Phase 2/Phase 3" or "Phase 3/Phase 4" trial HAS a Phase 3 arm. If such a trial is
  on record, describe it as such; never write that "no completed Phase 3 exists".
- Describe the trials by name with phase + status + title; note completed/terminated/active
  facts. Leave the "what stage has this reached" conclusion to the given stage.
- Use plain English. Never use internal field names (by_status, total_count, etc).

CLOSURE (live vs closed) — judge from the RELEVANT trials only:
- Call the pair CLOSED only on a real negative: a relevant Phase 3 terminated for SAFETY or
  benefit:risk, or literature reporting the drug failed for this indication. Operational stops
  (low enrollment, funding, sponsor decision) are NOT closure.
- An old/off-patent drug (first_approval many years before now) with NO approval for this
  indication is NOT closure — it reflects no commercial NDA filing, not efficacy failure.
- If neither holds, the pair is live (or unknown if you genuinely cannot tell).
- Say NOTHING about approval status in the prose.

Trials (relevant set):
{trials}

Respond with ONLY a JSON object:
{{"prose": "<the trial-section prose>", "closure": "live"|"closed"|"unknown", \
"closure_reason": "<one short sentence>"}}"""


def _format_trials(trials: list[Trial]) -> str:
    lines = []
    for t in trials:
        line = (
            f"- {t.nct_id or 'unknown'}: phase={t.phase or 'unknown'}, "
            f"status={t.overall_status or 'unknown'}, title={t.title or ''}"
        )
        if t.why_stopped:
            line += f", why_stopped={t.why_stopped}"
        lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class CTSummary:
    """The CT agent's isolated trial-section output: human-report prose plus a TYPED closure
    verdict the supervisor consumes directly (no parse-from-prose)."""

    prose: str
    closure: Literal["live", "closed", "unknown"]
    closure_reason: str


def _parse_summary(text: str) -> CTSummary | None:
    """Extract {prose, closure, closure_reason} from the LLM JSON response. None on parse
    failure or an unknown closure value (caller then leaves the summary empty — never
    fabricates)."""
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
        prose = data.get("prose")
        closure = data.get("closure")
        closure_reason = data.get("closure_reason")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(prose, str) or not prose.strip():
        return None
    if closure not in _CLOSURE_VALUES:
        return None
    reason = closure_reason.strip() if isinstance(closure_reason, str) else ""
    return CTSummary(prose=prose.strip(), closure=closure, closure_reason=reason)


async def judge_ct_summary(
    relevant_trials: list[Trial],
    *,
    stage: str,
    active_programs: str,
    first_approval: int | None,
    cache_dir: Path,
    drug: str = "",
    indication: str = "",
) -> CTSummary | None:
    """Author the trial-section prose + typed closure for the RELEVANT trial set, fed the
    resolved development `stage` so the prose cannot contradict it.

    `relevant_trials` must already be relevance-filtered and contamination-excluded upstream.
    `stage` is the authoritative dev_stage phrase (services.dev_stage.dev_stage_phrase).
    Returns None when there are no trials, or on a parse failure (the caller leaves the summary
    empty — never fabricates prose). Cached per the (sorted trial facts + stage + active_programs
    + first_approval) tuple so a pair is summarized once within the TTL window.
    """
    if not relevant_trials:
        return None

    first_approval_str = (
        str(first_approval) if first_approval is not None else "unknown"
    )
    facts = sorted(
        (t.nct_id or "", t.phase or "", t.overall_status or "", t.why_stopped or "")
        for t in relevant_trials
    )
    cache_params = {
        "drug": drug,
        "indication": indication,
        "stage": stage,
        "active_programs": active_programs,
        "first_approval": first_approval_str,
        "trials": facts,
    }
    cached = cache_get("ct_summary", cache_params, cache_dir)
    if isinstance(cached, dict) and cached.get("closure") in _CLOSURE_VALUES:
        return CTSummary(
            prose=cached.get("prose", ""),
            closure=cached["closure"],
            closure_reason=cached.get("closure_reason", ""),
        )

    prompt = _CT_SUMMARY_PROMPT.format(
        stage=stage,
        active_programs=active_programs,
        first_approval=first_approval_str,
        trials=_format_trials(relevant_trials),
    )
    response = await query_llm(prompt)
    summary = _parse_summary(response)
    if summary is None:
        logger.warning(
            "judge_ct_summary: could not parse a valid summary for %s x %s; leaving summary "
            "empty. Response was: %s",
            drug,
            indication,
            response,
        )
        return None

    cache_set(
        "ct_summary",
        cache_params,
        {
            "prose": summary.prose,
            "closure": summary.closure,
            "closure_reason": summary.closure_reason,
        },
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return summary
