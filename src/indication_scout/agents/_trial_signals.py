"""Deterministic trial FACTS derived from a ClinicalTrialsOutput artifact.

The supervisor's blurb rules hinge on one fact the model otherwise has to extract from a
phase string buried in a text table: "what is the highest completed phase, and was a
Phase 3 terminated for cause?". That extraction is deterministic and unambiguous — code
does it perfectly. So code computes the facts and feeds them to the LLM as an
authoritative block.

This module does NOT decide whether a hypothesis is CLOSED. Closure is a JUDGMENT made by
the clinical_trials sub-agent in its focused per-pair context (fed first_approval and the
relevance split). Code only surfaces the FACTS that judgment reads — including
`phase3_terminated_for_cause`, the positive closure signal. A pure-code closure trigger
was tried and produced FALSE-CLOSES; that is why the decision is an LLM judgment fed
facts, not a code rule.

Relevance filter: when `relevant_nct_ids` is supplied, the facts are computed over
RELEVANT trials only, so a contaminating Phase 3 pulled in by the recall-first search does
not inflate the phase or trip the termination-for-cause signal.
"""

from indication_scout.agents._trial_formatting import (
    _PHASE_RANK,
    _classify_stop_reason,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
    TrialSignals,
)
from indication_scout.models.model_clinical_trials import Trial

# Rank at/above which a label counts as "Phase 3 reached" for the (benign) stage-floor
# display — "Phase 2/Phase 3" (rank 5) qualifies for describing development maturity.
_PHASE3_MIN_RANK = _PHASE_RANK["Phase 2/Phase 3"]

# Stop categories that constitute a genuine cause-termination: an explicit safety or
# efficacy stop is the ONLY evidence of cause. A blank/unknown/operational/administrative
# reason is NOT evidence of a safety/efficacy stop and must not set the flag — absence of a
# stated reason is not a negative signal.
_CAUSE_STOP_CATEGORIES = {"safety", "efficacy"}


def _is_phase3(trial: Trial) -> bool:
    """True when the phase label is Phase 2/Phase 3 or higher (stage-floor display)."""
    return _PHASE_RANK.get(trial.phase or "", -1) >= _PHASE3_MIN_RANK


def _is_pivotal_phase3(trial: Trial) -> bool:
    """True only for the pivotal Phase 3 band: "Phase 2/Phase 3" or "Phase 3".

    Used for the cause-termination signal. Excludes pure "Phase 4" (post-approval — a
    Phase 4 event must not render as a "Phase 3 terminated for cause") and "Phase 3/Phase 4"
    (ambiguous combined designation). A contaminating sub-Phase-2/3 trial is also excluded.
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 2/Phase 3"] <= rank <= _PHASE_RANK["Phase 3"]


def _highest_completed_phase(trials: list[Trial]) -> str | None:
    """Return the highest-ranked phase label among the trials, or None if empty."""
    ranked = [
        (t.phase, _PHASE_RANK.get(t.phase or "", -1))
        for t in trials
        if t.phase and _PHASE_RANK.get(t.phase or "", -1) >= 0
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda pr: pr[1])[0]


def _filter_relevant(
    trials: list[Trial], relevant_nct_ids: set[str] | None
) -> list[Trial]:
    """Keep only trials whose nct_id is in the relevant set; pass through when None."""
    if relevant_nct_ids is None:
        return trials
    return [t for t in trials if t.nct_id in relevant_nct_ids]


def derive_trial_signals(
    ct: ClinicalTrialsOutput | None,
    relevant_nct_ids: set[str] | None = None,
) -> TrialSignals:
    """Compute deterministic trial facts from a clinical-trials artifact.

    When `relevant_nct_ids` is given, only those trials contribute — facts reflect
    RELEVANT trials, not the recall-first contaminated set.
    """
    if ct is None:
        return TrialSignals()

    completed_trials = _filter_relevant(
        ct.completed.trials if ct.completed else [], relevant_nct_ids
    )
    terminated_trials = _filter_relevant(
        ct.terminated.trials if ct.terminated else [], relevant_nct_ids
    )

    completed_phase3 = [t for t in completed_trials if _is_phase3(t)]
    completed_phase3_nct_ids = [t.nct_id for t in completed_phase3 if t.nct_id]

    terminated_phase3_for_cause = [
        t
        for t in terminated_trials
        if _is_pivotal_phase3(t)
        and _classify_stop_reason(t.why_stopped) in _CAUSE_STOP_CATEGORIES
    ]
    terminated_phase3_nct_ids = [
        t.nct_id for t in terminated_phase3_for_cause if t.nct_id
    ]

    highest = _highest_completed_phase(completed_trials + terminated_phase3_for_cause)

    return TrialSignals(
        highest_completed_phase=highest,
        has_completed_phase3=bool(completed_phase3),
        completed_phase3_nct_ids=completed_phase3_nct_ids,
        phase3_terminated_for_cause=bool(terminated_phase3_for_cause),
        terminated_phase3_nct_ids=terminated_phase3_nct_ids,
    )


def format_derived_signals(sig: TrialSignals) -> str:
    """Render the relevance-filtered facts as an authoritative block for the LLM.

    Facts only — no closure verdict and no closure hint. Closure is judged by the
    clinical_trials sub-agent; the supervisor reads these facts and the sub-agent's
    verdict, never re-deriving the decision here.
    """
    lines = [
        "DERIVED SIGNALS (authoritative facts — relevant trials only; do not re-derive "
        "the phase):",
        f"  highest_completed_phase: {sig.highest_completed_phase or 'none'}",
    ]
    if sig.has_completed_phase3:
        lines.append(
            f"  completed_phase_3: yes ({', '.join(sig.completed_phase3_nct_ids)})"
        )
    else:
        lines.append("  completed_phase_3: no")
    if sig.phase3_terminated_for_cause:
        lines.append(
            "  relevant_phase3_terminated_for_cause: yes "
            f"({', '.join(sig.terminated_phase3_nct_ids)})"
        )
    else:
        lines.append("  relevant_phase3_terminated_for_cause: no")
    return "\n".join(lines)


# Resolve the forward ref on ClinicalTrialsOutput.signals now that TrialSignals exists.
ClinicalTrialsOutput.model_rebuild()
