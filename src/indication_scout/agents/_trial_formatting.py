"""Shared formatters for rendering Trial data into LLM-visible content strings.

Both the clinical_trials sub-agent's tools and the supervisor's
`analyze_clinical_trials` tool use these helpers. The point of centralizing them is
to keep the prompt-side contract consistent across both layers — see
`agent_data_contracts.md` at the project root for the spec.

Background: the clinical-trials tools use
`@tool(response_format="content_and_artifact")`. LangChain serializes only the
`content` string into the API payload; the typed Pydantic artifact stays
Python-side and is invisible to the model. So whatever the LLM is going to
reason over has to be in the content string. These helpers build that string.
"""

from indication_scout.models.model_clinical_trials import MeshTerm, Trial

# Width of the phase column in rendered rows. Phase strings range from
# "Not Applicable" (14 chars) to "Phase 1/Phase 2" (15 chars). Pad to 16 so
# the trailing pipe lines up across rows for visual scanning.
_PHASE_COL_WIDTH = 16

# Phase ordering for distribution display — higher rank shown first. Matches
# data_sources.clinical_trials.ClinicalTrialsClient._phase_rank.
_PHASE_RANK = {
    "Not Applicable": 0,
    "Early Phase 1": 1,
    "Phase 1": 2,
    "Phase 1/Phase 2": 3,
    "Phase 2": 4,
    "Phase 2/Phase 3": 5,
    "Phase 3": 6,
    "Phase 3/Phase 4": 7,
    "Phase 4": 8,
}

_MESH_CAP = 10
_WHY_STOPPED_CAP = 500


def _phase_distribution(trials: list[Trial]) -> str:
    """Format a phase-count distribution string from a list of trials.

    Returns "Phase 3=2, Phase 2/Phase 3=1, Phase 4=1" — sorted by phase rank
    descending, skipping zero counts. Empty/missing phases render as
    "Unknown phase" and sort last.
    """
    counts: dict[str, int] = {}
    for t in trials:
        phase = t.phase or "Unknown phase"
        counts[phase] = counts.get(phase, 0) + 1

    def sort_key(phase: str) -> tuple[int, str]:
        # Higher rank first; unknown phase (rank=-1) sorts last.
        rank = _PHASE_RANK.get(phase, -1)
        return (-rank, phase)

    items = sorted(counts.items(), key=lambda kv: sort_key(kv[0]))
    return ", ".join(f"{phase}={n}" for phase, n in items)


def _format_mesh_list(mesh_terms: list[MeshTerm], cap: int = _MESH_CAP) -> str:
    """Render mesh_conditions as "term1; term2; term3", capped at `cap`.

    Preserves CT.gov ordering (first-listed = primary). Empty list renders
    as "(none)" so the LLM sees the absence explicitly rather than a
    missing column. Per project rules, no fabricated values.
    """
    if not mesh_terms:
        return "(none)"
    terms = [m.term for m in mesh_terms[:cap] if m.term]
    if not terms:
        return "(none)"
    return "; ".join(terms)


def _format_date(value: str | None) -> str:
    """Render a date for the supervisor view.

    CT.gov returns full ISO dates, year-month, or year-only depending on the
    trial. Pass through whatever CT.gov gave us; render missing dates as "?"
    rather than synthesizing a value.
    """
    if not value:
        return "?"
    return value


def _format_trial_row(
    trial: Trial,
    columns: tuple[str, ...],
    classified_stop_reason: str | None = None,
) -> str:
    """Render one trial as a pipe-separated row.

    Columns supported: nct_id, phase, status, start_date, completion_date,
    stop_reason, mesh, title. The phase column is padded for visual
    alignment; other columns render their value verbatim.

    `classified_stop_reason` is passed in (rather than computed here) so this
    module stays free of the keyword-classification logic — the caller owns
    the classifier.
    """
    parts: list[str] = []
    for col in columns:
        if col == "nct_id":
            parts.append(trial.nct_id or "?")
        elif col == "phase":
            phase = trial.phase or "Unknown phase"
            parts.append(phase.ljust(_PHASE_COL_WIDTH))
        elif col == "status":
            parts.append(trial.overall_status or "?")
        elif col == "start_date":
            parts.append(f"start {_format_date(trial.start_date)}")
        elif col == "completion_date":
            parts.append(f"end {_format_date(trial.completion_date)}")
        elif col == "stop_reason":
            # When the keyword classifier produces a useful label, surface it
            # so the LLM gets a deterministic hint. When the classifier punts
            # ("other" / "unknown"), fall through to the raw why_stopped text
            # — the LLM is the better classifier of last resort. If there's
            # no text at all, render "(none)" parallel to the mesh column.
            if classified_stop_reason and classified_stop_reason not in {
                "unknown",
                "other",
            }:
                parts.append(f"stop: {classified_stop_reason}")
            else:
                raw = (trial.why_stopped or "").strip()
                if raw:
                    parts.append(f"stop (raw): {_truncate_why_stopped(raw)}")
                else:
                    parts.append("stop (raw): (none)")
        elif col == "mesh":
            parts.append(f"mesh: {_format_mesh_list(trial.mesh_conditions)}")
        elif col == "refs":
            # PMIDs from CT.gov's referencesModule — papers the registry
            # links to this trial. Used by the supervisor as a signal that a
            # completed trial has a real published readout (not just a
            # registry entry).
            pmids = [p for p in trial.references if p]
            parts.append(f"refs: {', '.join(pmids) if pmids else '(none)'}")
        elif col == "title":
            parts.append(trial.title or "?")
        else:
            raise ValueError(f"Unknown trial-row column: {col!r}")
    return " | ".join(parts)


def _truncate_why_stopped(why_stopped: str | None) -> str:
    """Trim why_stopped to the per-trial cap, preserving the original text."""
    if not why_stopped:
        return ""
    text = why_stopped.strip()
    if len(text) <= _WHY_STOPPED_CAP:
        return text
    return text[:_WHY_STOPPED_CAP].rstrip() + "…"


def _borda_rank_by_enrollment_and_recency(trials: list[Trial], k: int) -> list[Trial]:
    """Return the top k trials ranked by enrollment desc and recency desc combined.

    Borda count: rank trials by enrollment descending and by completion_date
    descending independently, then sum the two ranks per trial. Lower combined
    rank = more interesting. Trials missing enrollment or completion_date get
    the worst rank (len(trials)) in that dimension, so they aren't preferred
    over trials with values. Ties broken by enrollment descending.

    Rationale: surfaces both well-powered older trials and newer trials
    without letting either dominate the supervisor's top-k slice.
    """
    if not trials:
        return []
    n = len(trials)

    # Rank by enrollment desc. Missing enrollment → worst rank (n).
    by_enrollment = sorted(
        enumerate(trials),
        key=lambda it: (it[1].enrollment if it[1].enrollment is not None else -1),
        reverse=True,
    )
    enrollment_rank: dict[int, int] = {}
    for rank, (idx, t) in enumerate(by_enrollment):
        enrollment_rank[idx] = rank if t.enrollment is not None else n

    # Rank by completion_date desc. Missing date → worst rank (n).
    by_date = sorted(
        enumerate(trials),
        key=lambda it: (it[1].completion_date or ""),
        reverse=True,
    )
    date_rank: dict[int, int] = {}
    for rank, (idx, t) in enumerate(by_date):
        date_rank[idx] = rank if t.completion_date else n

    def combined(idx_trial: tuple[int, Trial]) -> tuple[int, int]:
        idx, t = idx_trial
        combined_rank = enrollment_rank[idx] + date_rank[idx]
        # Tie-break: enrollment desc (so missing enrollment loses ties).
        tie_break = -(t.enrollment if t.enrollment is not None else -1)
        return (combined_rank, tie_break)

    ranked = sorted(enumerate(trials), key=combined)
    return [t for _, t in ranked[:k]]


def _format_trial_table(
    trials: list[Trial],
    columns: tuple[str, ...],
    cap: int,
    include_why_stopped: bool = False,
    stop_classifier=None,
) -> str:
    """Render a list of trials as a multi-line table.

    `cap` truncates the trials list (callers pre-sort if a specific ordering
    is required — Borda for the supervisor, enrollment-desc for the sub-agent
    via the existing artifact ordering).

    `include_why_stopped` adds an indented "why_stopped: ..." line under each
    row when True. `stop_classifier`, if provided, is a callable that maps a
    `why_stopped` string to a classified category — used to populate the
    `stop_reason` column without coupling this module to the classifier.
    """
    if not trials:
        return "  (none)"
    rows: list[str] = []
    for trial in trials[:cap]:
        classified = stop_classifier(trial.why_stopped) if stop_classifier else None
        row = _format_trial_row(trial, columns, classified_stop_reason=classified)
        rows.append(f"  {row}")
        if include_why_stopped:
            excerpt = _truncate_why_stopped(trial.why_stopped)
            if excerpt:
                rows.append(f"    why_stopped: {excerpt}")
    return "\n".join(rows)
