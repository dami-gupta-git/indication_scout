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

# Stop categories that constitute a genuine cause-termination: an explicit safety or
# efficacy stop is the ONLY evidence of cause. A blank/unknown/operational/administrative
# reason is NOT evidence of a safety/efficacy stop and must not set the flag — absence of a
# stated reason is not a negative signal.
_CAUSE_STOP_CATEGORIES = {"safety", "efficacy"}

# Statuses that count an active/ongoing program. overall_status carries display-cased strings
# ("Recruiting", "Active, not recruiting") as well as the API enum form; normalize
# (upper, non-alnum -> "_") before compare so both shapes match. Matched by EXACT set
# membership (not substring) — substring matching wrongly admitted "NOT_YET_RECRUITING" via
# "RECRUITING". "Not yet recruiting" IS included as a real (planned) program; "Enrolling by
# invitation" is an open program too. "Suspended" is a PAUSE, not a dead end, so it counts as
# ongoing development rather than a dead/unknown trial.
_ACTIVE_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "SUSPENDED",
}

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
    """True only for the pivotal Phase 3 band: "Phase 2/Phase 3" or "Phase 3".

    Used for the cause-termination signal. Excludes pure "Phase 4" (post-approval — a
    Phase 4 event must not render as a "Phase 3 terminated for cause") and "Phase 3/Phase 4"
    (ambiguous combined designation). A contaminating sub-Phase-2/3 trial is also excluded.
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 2/Phase 3"] <= rank <= _PHASE_RANK["Phase 3"]


def _is_pure_completed_phase3(trial: Trial) -> bool:
    """True for a true completed pivotal Phase 3: "Phase 3" or "Phase 3/Phase 4".

    EXCLUDES "Phase 2/Phase 3" — a completed combined Phase 2/3 trial is not, on its own, a
    completed pivotal Phase 3 readout (it often resolves at the Phase 2 stage). Used to keep the
    dev-stage judge from calling a candidate "Phase 3 completed" off a Phase 2/3 trial alone.
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return _PHASE_RANK["Phase 3"] <= rank <= _PHASE_RANK["Phase 3/Phase 4"]


def _is_active_pivotal_phase3(trial: Trial) -> bool:
    """True for the active-program band: "Phase 2/Phase 3", "Phase 3", or "Phase 3/Phase 4".

    Distinct from both other phase predicates. Used for the has_active_phase3 signal: a
    recruiting trial in this band is genuine pivotal development. Excludes pure "Phase 4"
    (post-approval / off-label activity — the very thing a "no dedicated development program,
    Phase 4 only" verdict correctly describes, so it must NOT trip this signal). "Phase
    3/Phase 4" has a Phase-3 arm and counts as pivotal activity.
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

    When `relevant_nct_ids` is given, only those trials contribute to the
    completed/terminated facts — they reflect RELEVANT trials, not the recall-first
    contaminated set.

    The active-Phase-3 fact is computed from the all-status `ct.search` set, which is
    NOT relevance-classified (only completed+terminated trials are). Contamination drop
    is therefore best-effort: trials whose nct is in `contaminated_nct_ids` are excluded,
    but an active trial the agent never saw in the completed/terminated lists is never
    flagged and can still slip in. Acceptable for a "yes, an active Phase 3 exists"
    boolean.
    """
    if ct is None:
        return TrialSignals()

    contaminated = contaminated_nct_ids or set()

    all_completed = ct.completed.trials if ct.completed else []
    all_terminated = ct.terminated.trials if ct.terminated else []
    completed_trials = _filter_relevant(all_completed, relevant_nct_ids)
    terminated_trials = _filter_relevant(all_terminated, relevant_nct_ids)

    # Search-set exclusion set for the active/unknown reads. The search set is NOT
    # relevance-classified, so we drop two kinds of irrelevant trials:
    #   1. explicitly contaminated ncts (best-effort, as before), and
    #   2. ncts the agent DID classify (they appear in completed/terminated) but judged
    #      NOT relevant — i.e. in the completed/terminated population yet absent from
    #      relevant_nct_ids. Without this, a relevant=False Phase 3 dropped from the
    #      completed signal would be re-admitted via the unfiltered search read (E4).
    search_excluded = set(contaminated)
    if relevant_nct_ids is not None:
        classified = {t.nct_id for t in all_completed + all_terminated if t.nct_id}
        search_excluded |= {n for n in classified if n not in relevant_nct_ids}

    # Active/ongoing Phase 3 from the all-status search set. Uses the bounded
    # {Phase 2/3, Phase 3, Phase 3/4} band (_is_active_pivotal_phase3) — NOT the >= floor:
    # a recruiting Phase 4 is post-approval/off-label activity, exactly the case the "no
    # dedicated development program, Phase 4 only" verdict describes, so it must not fire
    # this signal.
    search_trials = ct.search.trials if ct.search else []
    active_phase3 = [
        t
        for t in search_trials
        if _is_active(t)
        and _is_active_pivotal_phase3(t)
        and t.nct_id not in search_excluded
    ]
    active_phase3_nct_ids = [t.nct_id for t in active_phase3 if t.nct_id]

    # Use the pivotal band {Phase 2/3, Phase 3, Phase 3/4} (_is_active_pivotal_phase3) — the SAME
    # band as the active/unknown Phase-3 signals — NOT the >= floor (which counts a completed
    # Phase 4 as "completed Phase 3", since Phase 4 ranks above Phase 3) and NOT the stricter
    # termination band (_is_pivotal_phase3, 5-6) which excludes Phase 3/4. A completed Phase 3/4
    # trial has a Phase 3 arm and IS a completed Phase 3; a completed pure Phase 4 is post-approval
    # activity and is NOT. (The termination band is deliberately stricter — Phase 3/4 is ambiguous
    # for a cause-termination — but that strictness is wrong for the completed-evidence signal.)
    completed_phase3 = [t for t in completed_trials if _is_active_pivotal_phase3(t)]
    completed_phase3_nct_ids = [t.nct_id for t in completed_phase3 if t.nct_id]

    # A completed "Phase 2/Phase 3" trial is NOT a completed pivotal Phase 3 readout — the
    # combined designation often resolves at the Phase 2 stage. Track the PURE-Phase-3 completions
    # ("Phase 3" / "Phase 3/Phase 4", excluding "Phase 2/Phase 3") separately so the dev-stage
    # judge does not read a completed Phase 2/3 as "Phase 3 completed" — especially when the only
    # real Phase 3 trials are still recruiting (active_phase3). See _is_pure_completed_phase3.
    completed_pure_phase3 = [t for t in completed_phase3 if _is_pure_completed_phase3(t)]
    completed_pure_phase3_nct_ids = [t.nct_id for t in completed_pure_phase3 if t.nct_id]

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

    # NOTE: dev_stage is NOT set here. It is an LLM judgment (services/dev_stage.judge_dev_stage)
    # made over the relevant trials in run_clinical_trials_agent, which overwrites the field's
    # "untested" default on TrialSignals. The deterministic phase-rank ladder was removed: it
    # kept mis-encoding edge cases (Phase 4 ranks above Phase 3, etc.). The boolean signals
    # below (has_completed_phase3, has_active_phase3, …) are still computed deterministically —
    # they feed the supervisor's fact critic and report counts, not the stage tier.
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
