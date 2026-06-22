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

from indication_scout.agents._trial_formatting import _classify_stop_reason
from indication_scout.agents._trial_signals import _is_active
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
- A "Phase 3/Phase 4" trial HAS a true Phase 3 arm — a completed one counts as a completed Phase 3.
- A "Phase 2/Phase 3" trial is a COMBINED designation that often resolves at the Phase 2 stage. A \
completed Phase 2/Phase 3 counts as a completed Phase 3 ONLY when there is NO active/ongoing pure \
Phase 3 (or Phase 3/Phase 4) trial. When an ACTIVE pure Phase 3 trial also exists, the program's \
real pivotal stage is still ONGOING — classify as active_phase3, not completed_phase3 (a completed \
Phase 2/3 must not read as "Phase 3 completed" while the actual Phase 3 is still recruiting).
- A Phase 3 whose status is UNKNOWN is "on record, status unknown" — not completed, not active.
- A WITHDRAWN / no-longer-available Phase 3 never produced a program.
- Recruiting / Active, not recruiting / Not yet recruiting / Enrolling by invitation / \
Suspended = an active/ongoing program.
- A trial that is only RECRUITING (not completed) does NOT count as a completed phase.

Tiers (choose ONE, highest applicable wins, in this priority order):
1. phase3_terminated_for_cause — a Phase 3-band trial was TERMINATED for a safety or \
efficacy/benefit-risk reason (a closure signal), and no completed Phase 3 exists. Operational \
stops (low enrollment, funding, sponsor decision) do NOT count here.
2. completed_phase3 — a COMPLETED pure Phase 3 (or Phase 3/Phase 4) trial exists; OR a completed \
Phase 2/Phase 3 exists AND there is NO active pure Phase 3. (When the only completed Phase-3-band \
trial is a Phase 2/Phase 3 and an active pure Phase 3 is ALSO present, use active_phase3 instead — \
see the Phase 2/Phase 3 rule above.) A completed pure Phase 3 wins even when active Phase 3 trials \
are also present.
3. active_phase3 — an ACTIVE/ongoing Phase 3-band trial exists and no qualifying completed Phase 3 \
(per tier 2).
4. phase3_unknown_status — a Phase 3-band trial on record with UNKNOWN status (not completed, \
not active, not withdrawn).
5. completed_phase2 — a completed Phase 2 (or Phase 1/Phase 2) exists, and no Phase 3 at all.
6. exploratory_phase4_only — ONLY Phase 4 trials, no Phase 2 or Phase 3.
7. early_phase — trials exist but none reach a completed Phase 2 or higher: only Phase 1 / \
Early Phase 1, or trials whose phase is "Not Applicable" / blank / observational / otherwise \
unclassifiable. ANY trial on record that does not fit a higher tier lands here — a "Not \
Applicable" or no-phase trial is still a registered trial, NOT untested.
8. untested — NO trials at all on record (the list above is empty). Never choose this tier when \
any trial is listed, regardless of its phase.

Trials:
{trials}

Respond with ONLY a JSON object:
{{"tier": "<one_tier>", "reason": "<one short sentence>"}}"""


def _has_completed_phase3_band(trials: list[Trial]) -> bool:
    """A COMPLETED pure Phase 3 (or Phase 3/Phase 4) trial is on record.

    Excludes the combined "Phase 2/Phase 3" designation — that resolves to completed_phase3 only
    under the active-pure-Phase-3 rule, which stays the LLM's call. This guard only fires on the
    unambiguous case: a completed trial whose phase band IS Phase 3.
    """
    for t in trials:
        phase = (t.phase or "").strip().lower()
        status = (t.overall_status or "").strip().lower()
        if status != "completed":
            continue
        if phase in ("phase 3", "phase 3/phase 4", "phase3", "phase 3/phase4"):
            return True
    return False


def _has_active_phase3_band(trials: list[Trial]) -> bool:
    """An ACTIVE/ongoing pure Phase 3 (or Phase 3/Phase 4) trial is on record.

    Active-status matching reuses _trial_signals._is_active (handles CT.gov's underscored forms
    NOT_YET_RECRUITING / ACTIVE_NOT_RECRUITING etc). Excludes the combined "Phase 2/Phase 3"
    designation (same reasoning as the completed guard). Only fires on an unambiguous active Phase 3.
    """
    for t in trials:
        if not _is_active(t):
            continue
        if (t.phase or "").strip().lower() in ("phase 3", "phase 3/phase 4", "phase3", "phase 3/phase4"):
            return True
    return False


def _has_unknown_phase3_band(trials: list[Trial]) -> bool:
    """A Phase-3-band trial with genuinely UNKNOWN status is on record.

    Phase band here is the wider {Phase 2/Phase 3, Phase 3, Phase 3/Phase 4} set: an
    unknown-status Phase 2/3 legitimately supports "Phase 3 on record, status unknown".
    UNKNOWN means CT.gov's "Unknown status" — NOT terminated, withdrawn, completed, or active
    (those are KNOWN statuses and must not route into the "status unknown" tier).
    """
    for t in trials:
        if (t.overall_status or "").strip().lower() not in ("unknown", "unknown status"):
            continue
        if (t.phase or "").strip().lower() in (
            "phase 2/phase 3",
            "phase2/phase3",
            "phase 2/phase3",
            "phase 3",
            "phase3",
            "phase 3/phase 4",
            "phase 3/phase4",
        ):
            return True
    return False


def _has_completed_phase2_band(trials: list[Trial]) -> bool:
    """A COMPLETED Phase 2 (or Phase 1/Phase 2, Phase 2/Phase 3) trial is on record."""
    for t in trials:
        if (t.overall_status or "").strip().lower() != "completed":
            continue
        if (t.phase or "").strip().lower() in (
            "phase 2",
            "phase2",
            "phase 1/phase 2",
            "phase 1/phase2",
            "phase 2/phase 3",
            "phase2/phase3",
            "phase 2/phase3",
        ):
            return True
    return False


def _terminated_for_cause_supported(trials: list[Trial]) -> bool:
    """True when a Phase-3-band trial in the set was terminated for a SAFETY or EFFICACY
    reason — the only evidence that justifies the phase3_terminated_for_cause tier.

    The dev-stage judge is fed only nct_id + phase + status (no why_stopped), so it cannot
    itself know whether a TERMINATED Phase 3 was a cause-stop or an operational one (COVID,
    enrollment, funding, sponsor/withdrawal). This deterministic check — reusing the same
    keyword classifier as the _trial_signals fact — gates the tier so a reason-blind guess
    cannot brand an operational termination as a safety/efficacy closure.
    """
    for t in trials:
        if (t.overall_status or "").strip().lower() != "terminated":
            continue
        if (t.phase or "").strip().lower() not in (
            "phase 2/phase 3",
            "phase2/phase3",
            "phase 2/phase3",
            "phase 3",
            "phase3",
        ):
            continue
        if _classify_stop_reason(t.why_stopped) in ("safety", "efficacy"):
            return True
    return False


def _enforce_tier_floor(tier: str, trials: list[Trial]) -> str:
    """Deterministic floor over the LLM tier — code owns the clinical-accuracy invariants the
    prompt must not be trusted to honor at scale:

    1. trials exist -> the tier can NEVER be 'untested' (that means zero trials on record).
    2. a completed pure Phase-3-band trial exists -> the tier is at least 'completed_phase3'.
    3. an active pure Phase-3-band trial exists -> the tier is at least 'active_phase3'
       (unless a completed Phase 3 already raised it higher — completed outranks active).
    4. phase3_terminated_for_cause is honored ONLY when a Phase-3-band trial actually has a
       safety/efficacy why_stopped. The judge is reason-blind (it sees only phase+status), so an
       unsupported guess is demoted to the highest applicable tier from the remaining evidence
       (completed P3 -> active P3 -> genuinely-unknown-status P3 -> completed P2 -> early).
       Operational/withdrawn Phase 3s do NOT route into phase3_unknown_status — "status unknown"
       must not be asserted about a trial whose status is actually known.

    Each rule only RAISES the tier toward the trial evidence; a higher LLM tier is left alone.
    Mirrors the strength-cap pattern: the prompt emits the nuanced read; code enforces the floor.
    """
    if not trials:
        return tier
    # terminated-for-cause is honored ONLY when a real safety/efficacy stop backs it; otherwise
    # the reason-blind judge guessed it, so demote to the highest tier the evidence supports.
    if tier == "phase3_terminated_for_cause":
        if _terminated_for_cause_supported(trials):
            return tier
        if _has_completed_phase3_band(trials):
            return "completed_phase3"
        if _has_active_phase3_band(trials):
            return "active_phase3"
        if _has_unknown_phase3_band(trials):
            return "phase3_unknown_status"
        if _has_completed_phase2_band(trials):
            return "completed_phase2"
        return "early_phase"
    if _has_completed_phase3_band(trials):
        # completed_phase3 outranks active_phase3, so it wins when both are present.
        return "completed_phase3"
    if _has_active_phase3_band(trials):
        # Raise to active_phase3 only when the LLM tier is BELOW it (active is below completed/
        # unknown in the ladder, above completed_phase2/exploratory/early/untested).
        if tier not in ("completed_phase3", "active_phase3", "phase3_unknown_status"):
            return "active_phase3"
    if tier == "untested":
        return "early_phase"
    return tier


_PURE_PHASE3 = ("phase 3", "phase 3/phase 4", "phase3", "phase 3/phase4")
_PHASE2_PHASE3 = ("phase 2/phase 3", "phase2/phase3", "phase 2/phase3")


def _active_trials(trials: list[Trial]) -> list[Trial]:
    """Trials with an active/ongoing status (reuses _is_active for CT.gov status forms)."""
    return [t for t in trials if _is_active(t)]


def _render_active_programs(trials: list[Trial]) -> str:
    """Deterministically render the 'active programs' line — what is still MOVING — from the
    relevant trial set. NO LLM: counting and listing NCTs is mechanical, and letting the model do
    it produced miscounts (count != listed ids) and false "None active". Rules mirror the former
    prompt:
      - active PIVOTAL trials (pure Phase 3 band, else Phase 2/3) → name them with NCT ids and a
        count that equals the listed ids.
      - else if earlier-phase / post-approval trials are active → "No pivotal program active;
        <N> non-pivotal active (NCT...)".
      - else → "None active".
    Completed/terminated trials are never listed (only active statuses qualify).
    """
    active = _active_trials(trials)
    if not active:
        return "None active"

    def _ids(predicate) -> list[str]:
        return [t.nct_id for t in active if t.nct_id and predicate(t)]

    def _phase(t: Trial) -> str:
        return (t.phase or "").strip().lower()

    pure3 = _ids(lambda t: _phase(t) in _PURE_PHASE3)
    p2p3 = _ids(lambda t: _phase(t) in _PHASE2_PHASE3)
    pivotal = pure3 + p2p3
    if pivotal:
        parts = []
        if pure3:
            parts.append(f"{len(pure3)} Phase 3 active/recruiting ({', '.join(pure3)})")
        if p2p3:
            parts.append(f"{len(p2p3)} Phase 2/Phase 3 active ({', '.join(p2p3)})")
        return "; ".join(parts)

    # No pivotal trial active, but something earlier-phase / post-approval is moving.
    non_pivotal = [t.nct_id for t in active if t.nct_id]
    if non_pivotal:
        return (
            f"No pivotal program active; {len(non_pivotal)} non-pivotal active "
            f"({', '.join(non_pivotal)})"
        )
    return "None active"


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


def _parse_tier(text: str) -> str | None:
    """Extract the `tier` from the LLM JSON response. None on parse failure or an unknown tier
    (so the caller can fall back to the safe floor). active_programs is NOT read from the LLM —
    it is rendered deterministically in judge_dev_stage (_render_active_programs)."""
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
    except (json.JSONDecodeError, AttributeError):
        return None
    if tier not in DEV_STAGE_TIERS:
        return None
    return tier


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
    # active_programs is ALWAYS deterministic (counting/listing NCTs is mechanical, not judgment).
    active_programs = _render_active_programs(relevant_trials)
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
        # Tier is cached (the LLM call); active_programs is re-rendered deterministically so a
        # prompt/logic change to the line takes effect without a cache bust.
        return StageJudgment(
            tier=cached.get("tier", "untested"),
            active_programs=active_programs,
        )

    prompt = _STAGE_PROMPT.format(trials=_format_trials(relevant_trials))
    response = await query_llm(prompt)
    tier = _parse_tier(response)
    if tier is None:
        logger.warning(
            "judge_dev_stage: could not parse a valid tier for %s x %s; defaulting to "
            "the safe floor. Response was: %s",
            drug,
            indication,
            response,
        )
        tier = "untested"

    floored_tier = _enforce_tier_floor(tier, relevant_trials)
    if floored_tier != tier:
        logger.warning(
            "judge_dev_stage: floored tier %r -> %r for %s x %s (deterministic invariant)",
            tier,
            floored_tier,
            drug,
            indication,
        )

    # Cache only the tier (the LLM judgment); active_programs is always re-rendered.
    cache_set(
        "dev_stage",
        cache_params,
        {"tier": floored_tier},
        cache_dir,
        ttl=JUDGMENT_CACHE_TTL,
    )
    return StageJudgment(tier=floored_tier, active_programs=active_programs)
