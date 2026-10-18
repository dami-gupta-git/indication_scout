"""Deterministic trial facts derived from a ClinicalTrialsOutput artifact.

Code computes facts such as the highest completed phase and whether a Phase 3 was terminated for cause, and feeds
them to the LLM as an authoritative block. Closure of a hypothesis is an LLM judgment by the clinical_trials
sub-agent; a pure-code closure trigger produced false closes.

When `relevant_nct_ids` is supplied, facts are computed over relevant trials only, so a contaminating Phase 3
cannot inflate the phase or trip the termination signal.
"""

from indication_scout.agents._trial_formatting import (
    _PHASE_RANK,
    _classify_stop_reason,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
    TrialSignals,
)
from indication_scout.constants import (
    CLINICAL_TRIALS_ACTIVE_STATUSES,
    NON_THERAPEUTIC_INTERVENTION_TYPES,
)
from indication_scout.models.model_clinical_trials import Trial

# Only an explicit safety or efficacy stop counts as cause-termination; a blank or operational reason does not.
_CAUSE_STOP_CATEGORIES = {"safety", "efficacy"}


def is_non_therapeutic_study(trial: Trial) -> bool:
    """True when EVERY typed intervention is non-therapeutic (diagnostic test, device, tracer, procedure).

    Such a trial studies that object, not the drug. The judgment is disease-independent, so it is settled once here
    rather than per candidate. A trial with no typed interventions returns False and is left to the relevance gate.
    """
    typed = [i.intervention_type for i in trial.interventions if i.intervention_type]
    if not typed:
        return False
    return all(t in NON_THERAPEUTIC_INTERVENTION_TYPES for t in typed)


# Statuses counting as an active program. overall_status may be display-cased ("Active, not recruiting") or the API
# enum, so normalize (upper, non-alnum -> "_") and match by exact set membership (substring matching admitted
# "NOT_YET_RECRUITING" via "RECRUITING"). Planned and invitation-only programs count; "Suspended" is a pause, so it
# counts as ongoing.
_ACTIVE_STATUSES = frozenset(CLINICAL_TRIALS_ACTIVE_STATUSES)


def _normalize_status(status: str) -> str:
    """Uppercase; collapse each run of non-alphanumerics to a single underscore."""
    out: list[str] = []
    for c in (status or "").upper():
        if c.isalnum():
            out.append(c)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def _is_active(trial: Trial) -> bool:
    """True when overall_status is an active/ongoing program status (exact match)."""
    return _normalize_status(trial.overall_status) in _ACTIVE_STATUSES


def _is_pivotal_phase3(trial: Trial) -> bool:
    """True only for the pivotal band "Phase 2/Phase 3" or "Phase 3", used for the cause-termination signal.

    Excludes pure "Phase 4" (post-approval) and the ambiguous "Phase 3/Phase 4".
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 2/Phase 3"] <= rank <= _PHASE_RANK["Phase 3"]


def _is_pure_completed_phase3(trial: Trial) -> bool:
    """True for a completed pivotal Phase 3: "Phase 3" or "Phase 3/Phase 4".

    Excludes "Phase 2/Phase 3", which often resolves at the Phase 2 stage.
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 3"] <= rank <= _PHASE_RANK["Phase 3/Phase 4"]


def _is_active_pivotal_phase3(trial: Trial) -> bool:
    """True for the active-program band: "Phase 2/Phase 3", "Phase 3", or "Phase 3/Phase 4".

    Excludes pure "Phase 4" (post-approval / off-label activity).
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 2/Phase 3"] <= rank <= _PHASE_RANK["Phase 3/Phase 4"]


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
    contaminated_nct_ids: set[str] | None = None,
) -> TrialSignals:
    """Compute deterministic trial facts from a clinical-trials artifact.

    When `relevant_nct_ids` is given, only those trials contribute to the completed/terminated facts.
    The active-Phase-3 fact comes from the all-status `ct.search` set, which is not relevance-classified, so
    contamination dropping is best-effort there.
    """
    if ct is None:
        return TrialSignals()

    contaminated = contaminated_nct_ids or set()

    all_completed = ct.completed.trials if ct.completed else []
    all_terminated = ct.terminated.trials if ct.terminated else []
    completed_trials = _filter_relevant(all_completed, relevant_nct_ids)
    terminated_trials = _filter_relevant(all_terminated, relevant_nct_ids)

    # Exclusions for the active/unknown reads (the search set is not relevance-classified): contaminated ncts, and
    # ncts classified as not relevant, which would otherwise be re-admitted via the unfiltered search read.
    search_excluded = set(contaminated)
    if relevant_nct_ids is not None:
        classified = {t.nct_id for t in all_completed + all_terminated if t.nct_id}
        search_excluded |= {n for n in classified if n not in relevant_nct_ids}

    # Active Phase 3 from the search set, using the bounded pivotal band, not the >= floor (a recruiting Phase 4
    # is post-approval activity).
    search_trials = ct.search.trials if ct.search else []
    active_phase3 = [
        t
        for t in search_trials
        if _is_active(t)
        and _is_active_pivotal_phase3(t)
        and t.nct_id not in search_excluded
    ]
    active_phase3_nct_ids = [t.nct_id for t in active_phase3 if t.nct_id]

    # Use the same pivotal band as the active signals, not the >= floor (counts a completed Phase 4) and not the
    # stricter termination band (excludes Phase 3/4, which has a Phase 3 arm).
    completed_phase3 = [t for t in completed_trials if _is_active_pivotal_phase3(t)]
    completed_phase3_nct_ids = [t.nct_id for t in completed_phase3 if t.nct_id]

    # Track pure-Phase-3 completions separately: a completed Phase 2/3 is not a Phase 3 readout, and the
    # dev-stage judge must not read it as one. See _is_pure_completed_phase3.
    completed_pure_phase3 = [
        t for t in completed_phase3 if _is_pure_completed_phase3(t)
    ]
    completed_pure_phase3_nct_ids = [
        t.nct_id for t in completed_pure_phase3 if t.nct_id
    ]

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

    # dev_stage is not set here; it is an LLM judgment (judge_dev_stage) that overwrites the "untested" default.
    # The boolean signals below feed the supervisor's fact critic and report counts.
    return TrialSignals(
        highest_completed_phase=highest,
        has_completed_phase3=bool(completed_phase3),
        completed_phase3_nct_ids=completed_phase3_nct_ids,
        has_completed_pure_phase3=bool(completed_pure_phase3),
        completed_pure_phase3_nct_ids=completed_pure_phase3_nct_ids,
        has_active_phase3=bool(active_phase3),
        active_phase3_nct_ids=active_phase3_nct_ids,
        phase3_terminated_for_cause=bool(terminated_phase3_for_cause),
        terminated_phase3_nct_ids=terminated_phase3_nct_ids,
    )


def format_derived_signals(sig: TrialSignals) -> str:
    """Render the relevance-filtered facts as an authoritative block for the LLM; facts only, no closure verdict."""
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
    if sig.has_active_phase3:
        lines.append(f"  active_phase_3: yes ({', '.join(sig.active_phase3_nct_ids)})")
    else:
        lines.append("  active_phase_3: no")
    if sig.phase3_terminated_for_cause:
        lines.append(
            "  relevant_phase3_terminated_for_cause: yes "
            f"({', '.join(sig.terminated_phase3_nct_ids)})"
        )
    else:
        lines.append("  relevant_phase3_terminated_for_cause: no")
    lines.append(f"  dev_stage: {sig.dev_stage}")
    return "\n".join(lines)


# Resolve the forward ref on ClinicalTrialsOutput.signals now that TrialSignals exists.
ClinicalTrialsOutput.model_rebuild()
