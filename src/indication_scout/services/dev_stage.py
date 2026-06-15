"""LLM-judged development-stage tier for a drug x indication pair.

The development STAGE (which tier the pair sits in) is a clinical judgment the LLM makes
reliably from trial phase + status — validated in scratch/dev_stage_judgment_harness.py (18/18
cases on Sonnet, including the Phase-4-ranks-above-Phase-3 trap, completed Phase 2/Phase 3,
withdrawn/unknown status, and large mixed portfolios). It replaces the deterministic
phase-rank code, which kept mis-encoding edge cases (Phase 4 inflation, etc.).

Sent data is MINIMAL and structured: for each RELEVANT trial (relevance/contamination
filtering stays deterministic upstream), only nct_id + phase + overall_status. The judgment is
cached (per the relevant trial set) so a given pair is judged once within the TTL window.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from indication_scout.constants import JUDGMENT_CACHE_TTL
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import query_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Authoritative development-stage tier -> english phrase. This module is the SINGLE home of
# both the tier vocabulary and its rendered phrase, so the supervisor blurb, the demotion
# footer, and the per-disease report section all state the tier identically (one source of
# truth). The active Phase 3 NCTs are appended at render time when present.
DEV_STAGE_PHRASE = {
    "phase3_terminated_for_cause": "Phase 3 terminated for cause (safety/efficacy stop)",
    "completed_phase3": "Phase 3 completed for this indication",
    "active_phase3": "Active Phase 3 development on record for this indication",
    "phase3_unknown_status": "Phase 3 on record, status unknown",
    "completed_phase2": "Phase 2 completed for this indication, no Phase 3",
    "exploratory_phase4_only": (
        "Phase 4 exploratory only (post-approval off-label study; no dedicated "
        "development program for this indication)"
    ),
    "early_phase": "Early-phase only, no completed pivotal readout",
    "untested": "No registry (ClinicalTrials.gov) trials on record for this indication",
}

DEV_STAGE_TIERS = tuple(DEV_STAGE_PHRASE)


def dev_stage_phrase(sig) -> str | None:
    """Render the authoritative stage phrase for a TrialSignals, or None when unavailable.

    Appends the active Phase 3 NCT ids for the active_phase3 tier so the line names the
    programs it is asserting exist.
    """
    if sig is None or not getattr(sig, "dev_stage", None):
        return None
    phrase = DEV_STAGE_PHRASE.get(sig.dev_stage)
    if phrase is None:
        return None
    if sig.dev_stage == "active_phase3" and sig.active_phase3_nct_ids:
        phrase = f"{phrase} ({', '.join(sig.active_phase3_nct_ids)})"
    return phrase


_STAGE_PROMPT = """You are a clinical development analyst. Given the trials below for one \
drug x indication pair, classify the development STAGE into exactly ONE tier.

Apply the clinical-trial conventions you already know. In particular:
- Phases rank Early Phase 1 < Phase 1 < Phase 1/2 < Phase 2 < Phase 2/3 < Phase 3 < Phase 4.
- BUT Phase 4 is POST-APPROVAL / off-label activity — it is NOT progression beyond Phase 3.
  A trial that is only Phase 4 (with no Phase 2 or Phase 3 on record) is exploratory, NOT a
  completed pivotal program.
- A "Phase 2/Phase 3" or "Phase 3/Phase 4" trial HAS a Phase 3 arm — a completed one counts \
as a completed Phase 3.
- A Phase 3 whose status is UNKNOWN is "on record, status unknown" — not completed, not active.
- A WITHDRAWN / no-longer-available Phase 3 never produced a program.
- Recruiting / Active, not recruiting / Not yet recruiting / Enrolling by invitation / \
Suspended = an active/ongoing program.
- A trial that is only RECRUITING (not completed) does NOT count as a completed phase.

Tiers (choose ONE, highest applicable wins, in this priority order):
1. phase3_terminated_for_cause — a Phase 3-band trial was TERMINATED for a safety or \
efficacy/benefit-risk reason (a closure signal), and no completed Phase 3 exists. Operational \
stops (low enrollment, funding, sponsor decision) do NOT count here.
2. completed_phase3 — a COMPLETED Phase 3 / Phase 2/Phase 3 / Phase 3/Phase 4 trial exists. \
(This wins even when active Phase 3 trials are ALSO present — they are additional.)
3. active_phase3 — an ACTIVE/ongoing Phase 3-band trial exists, but none completed.
4. phase3_unknown_status — a Phase 3-band trial on record with UNKNOWN status (not completed, \
not active, not withdrawn).
5. completed_phase2 — a completed Phase 2 (or Phase 1/Phase 2) exists, and no Phase 3 at all.
6. exploratory_phase4_only — ONLY Phase 4 trials, no Phase 2 or Phase 3.
7. early_phase — only Phase 1 / Early Phase 1 / unclassifiable phase, or trials with no \
completed phase.
8. untested — no trials.

Also summarize ACTIVE PROGRAMS — what clinical activity is still MOVING (recruiting, active,
not-yet-recruiting, enrolling, suspended). A COMPLETED or TERMINATED trial is NOT an active
program — never list one here. One short line, following these rules EXACTLY:
- If an active PIVOTAL trial exists (Phase 2/3 or Phase 3), name those with their NCT ids, e.g.
  "2 Phase 3 recruiting (NCT…, NCT…)". Prefer pivotal; you may append a brief note of any
  non-pivotal activity.
- If NO pivotal trial is active but EARLIER-phase or POST-APPROVAL trials ARE moving (Phase 1,
  Phase 1/2, Phase 2, Phase 4, Early Phase 1), do NOT write "None active" — that would be false.
  Write exactly: "No pivotal program active; <phase>-only activity (NCT…)", e.g. "No pivotal
  program active; Phase 1/Phase 4-only activity (NCT06959784)". Phase 4 is POST-APPROVAL
  off-label activity, NOT pivotal development — so it is correctly "no pivotal program active",
  but it IS still activity and must be disclosed, never hidden behind "None active".
- ONLY when NOTHING is recruiting/active at any phase, return exactly "None active".
Never write "None active" in the same line as a list of active trials — that is self-contradictory.

Trials:
{trials}

Respond with ONLY a JSON object:
{{"tier": "<one_tier>", "active_programs": "<one short line or 'None active'>", \
"reason": "<one short sentence>"}}"""


def _format_trials(trials: list[Trial]) -> str:
    return "\n".join(
        f"- {t.nct_id}: phase={t.phase or 'unknown'}, "
        f"status={t.overall_status or 'unknown'}"
        for t in trials
    )


@dataclass(frozen=True)
class StageJudgment:
    """The LLM's isolated read of a relevant trial set: the development-stage tier plus a
    one-line 'what is still moving' summary (active programs)."""

    tier: str
    active_programs: str


def _parse_judgment(text: str) -> StageJudgment | None:
    """Extract {tier, active_programs} from the LLM JSON response. None on parse failure or
    an unknown tier (so the caller can fall back to the safe floor)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Strip a ```json ... ``` fence.
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
    try:
        data = json.loads(stripped)
        tier = data.get("tier")
        active = data.get("active_programs")
    except (json.JSONDecodeError, AttributeError):
        return None
    if tier not in DEV_STAGE_TIERS:
        return None
    # active_programs is free text; default to "None active" when missing/blank.
    active_programs = (
        active.strip()
        if isinstance(active, str) and active.strip()
        else ("None active")
    )
    return StageJudgment(tier=tier, active_programs=active_programs)


async def judge_dev_stage(
    relevant_trials: list[Trial],
    cache_dir: Path,
    *,
    drug: str = "",
    indication: str = "",
) -> StageJudgment:
    """Return the development-stage tier + active-programs line for the RELEVANT trial set.

    `relevant_trials` must already be relevance-filtered and contamination-excluded upstream —
    only nct_id + phase + overall_status are sent to the model. Cached per the (sorted) trial
    facts so a pair is judged once within the TTL window. Returns tier="untested" /
    active_programs="None active" when there are no trials. On a parse failure the raw output
    is logged and the same safe floor is returned (never fabricates a higher tier).
    """
    floor = StageJudgment(tier="untested", active_programs="None active")
    if not relevant_trials:
        return floor

    # Cache key = the sorted (nct, phase, status) facts. Order-independent so two orderings of
    # the same set collapse to one entry. drug/indication included for cache readability only.
    facts = sorted(
        (t.nct_id or "", t.phase or "", t.overall_status or "") for t in relevant_trials
    )
    cache_params = {"drug": drug, "indication": indication, "trials": facts}
    cached = cache_get("dev_stage", cache_params, cache_dir)
    if isinstance(cached, dict):  # ignore stale str-format entries from the old schema
        return StageJudgment(
            tier=cached.get("tier", "untested"),
            active_programs=cached.get("active_programs", "None active"),
        )

    prompt = _STAGE_PROMPT.format(trials=_format_trials(relevant_trials))
    response = await query_llm(prompt)
    judgment = _parse_judgment(response)
    if judgment is None:
        logger.warning(
            "judge_dev_stage: could not parse a valid judgment for %s x %s; defaulting to "
            "the safe floor. Response was: %s",
            drug,
            indication,
            response,
        )
        judgment = floor

    cache_set(
        "dev_stage",
        cache_params,
        {"tier": judgment.tier, "active_programs": judgment.active_programs},
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return judgment
