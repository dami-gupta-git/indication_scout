"""Compare a golden SupervisorOutput snapshot to a current one.

Produces a list of `Diff` records. Pure function — no I/O, no network, no LLM.
Callable from both the regression test and the `scout diff-report` CLI.

Snapshot boundary is "final + per-agent": the SupervisorOutput already nests
LiteratureOutput, ClinicalTrialsOutput, and MechanismOutput under
disease_findings / mechanism, so one Pydantic dump captures everything.
"""

from __future__ import annotations

from typing import Any

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.regression.constants import (
    BLURB_PROSE_MAX_LEN,
    BLURB_PROSE_MIN_LEN,
    CANDIDATE_SET_JACCARD_MIN,
    EVIDENCE_COUNT_TOLERANCE,
    SUMMARY_MAX_LEN,
    SUMMARY_MIN_LEN,
    TOP_DISEASES_JACCARD_MIN,
)
from indication_scout.regression.diff import Diff, jaccard


def compare_reports(golden: SupervisorOutput, current: SupervisorOutput) -> list[Diff]:
    """Return all structural and semantic-overlap diffs between two reports."""
    diffs: list[Diff] = []
    diffs.extend(_compare_top_level(golden, current))
    diffs.extend(_compare_candidate_sets(golden, current))
    diffs.extend(_compare_mechanism(golden, current))
    diffs.extend(_compare_per_disease(golden, current))
    diffs.extend(_compare_summary(golden, current))
    return diffs


def _compare_top_level(
    golden: SupervisorOutput, current: SupervisorOutput
) -> list[Diff]:
    diffs: list[Diff] = []
    if golden.drug_name != current.drug_name:
        diffs.append(
            Diff(
                path="drug_name",
                kind="value_changed",
                severity="error",
                detail="drug_name differs between golden and current",
                golden=golden.drug_name,
                current=current.drug_name,
            )
        )
    if not current.candidate_diseases:
        diffs.append(
            Diff(
                path="candidate_diseases",
                kind="empty",
                severity="error",
                detail="current run produced no candidate diseases",
                golden=len(golden.candidate_diseases),
                current=0,
            )
        )
    return diffs


def _compare_candidate_sets(
    golden: SupervisorOutput, current: SupervisorOutput
) -> list[Diff]:
    diffs: list[Diff] = []
    g_cands = set(golden.candidate_diseases)
    c_cands = set(current.candidate_diseases)
    cand_overlap = jaccard(g_cands, c_cands)
    if cand_overlap < CANDIDATE_SET_JACCARD_MIN:
        diffs.append(
            Diff(
                path="candidate_diseases",
                kind="set_divergence",
                severity="error",
                detail=(
                    f"Jaccard {cand_overlap:.2f} < min {CANDIDATE_SET_JACCARD_MIN:.2f}; "
                    f"only_golden={sorted(g_cands - c_cands)}; "
                    f"only_current={sorted(c_cands - g_cands)}"
                ),
                golden=sorted(g_cands),
                current=sorted(c_cands),
            )
        )

    g_top = set(golden.top_diseases)
    c_top = set(current.top_diseases)
    top_overlap = jaccard(g_top, c_top)
    if top_overlap < TOP_DISEASES_JACCARD_MIN:
        diffs.append(
            Diff(
                path="top_diseases",
                kind="set_divergence",
                severity="error",
                detail=(
                    f"Jaccard {top_overlap:.2f} < min {TOP_DISEASES_JACCARD_MIN:.2f}; "
                    f"only_golden={sorted(g_top - c_top)}; "
                    f"only_current={sorted(c_top - g_top)}"
                ),
                golden=sorted(g_top),
                current=sorted(c_top),
            )
        )

    # top_diseases must remain a strict subset of disease_findings (invariant
    # enforced upstream; a violation means the supervisor's contract changed).
    findings_diseases = {f.disease for f in current.disease_findings}
    leaked = c_top - findings_diseases
    if leaked:
        diffs.append(
            Diff(
                path="top_diseases",
                kind="invariant_violation",
                severity="error",
                detail=f"top_diseases not a subset of disease_findings: {sorted(leaked)}",
                golden=None,
                current=sorted(leaked),
            )
        )
    return diffs


def _compare_mechanism(
    golden: SupervisorOutput, current: SupervisorOutput
) -> list[Diff]:
    diffs: list[Diff] = []
    if golden.mechanism is None and current.mechanism is None:
        return diffs
    if golden.mechanism is None or current.mechanism is None:
        diffs.append(
            Diff(
                path="mechanism",
                kind="presence_changed",
                severity="error",
                detail="mechanism populated in one run but not the other",
                golden=golden.mechanism is not None,
                current=current.mechanism is not None,
            )
        )
        return diffs

    g_targets = set(golden.mechanism.drug_targets.keys())
    c_targets = set(current.mechanism.drug_targets.keys())
    if g_targets != c_targets:
        # Drug targets come from OpenTargets, not the LLM — any drift here is
        # an upstream-data or caching issue worth investigating.
        diffs.append(
            Diff(
                path="mechanism.drug_targets",
                kind="set_divergence",
                severity="error",
                detail=(
                    f"drug_targets diverged; only_golden={sorted(g_targets - c_targets)}; "
                    f"only_current={sorted(c_targets - g_targets)}"
                ),
                golden=sorted(g_targets),
                current=sorted(c_targets),
            )
        )

    g_cand_diseases = {c.disease_name for c in golden.mechanism.candidates}
    c_cand_diseases = {c.disease_name for c in current.mechanism.candidates}
    mech_overlap = jaccard(g_cand_diseases, c_cand_diseases)
    if mech_overlap < CANDIDATE_SET_JACCARD_MIN:
        diffs.append(
            Diff(
                path="mechanism.candidates",
                kind="set_divergence",
                severity="error",
                detail=(
                    f"mechanism candidate diseases Jaccard {mech_overlap:.2f} "
                    f"< min {CANDIDATE_SET_JACCARD_MIN:.2f}"
                ),
                golden=sorted(g_cand_diseases),
                current=sorted(c_cand_diseases),
            )
        )
    return diffs


def _compare_per_disease(
    golden: SupervisorOutput, current: SupervisorOutput
) -> list[Diff]:
    diffs: list[Diff] = []
    g_by_disease = {f.disease: f for f in golden.disease_findings}
    c_by_disease = {f.disease: f for f in current.disease_findings}
    shared = sorted(set(g_by_disease) & set(c_by_disease))
    for disease in shared:
        diffs.extend(_compare_one_finding(g_by_disease[disease], c_by_disease[disease]))
    return diffs


def _compare_one_finding(
    golden: CandidateFindings, current: CandidateFindings
) -> list[Diff]:
    diffs: list[Diff] = []
    path_prefix = f"disease_findings[{current.disease!r}]"

    if golden.literature is not None and current.literature is None:
        diffs.append(
            Diff(
                path=f"{path_prefix}.literature",
                kind="presence_changed",
                severity="error",
                detail="literature populated in golden but not current",
                golden=True,
                current=False,
            )
        )
    elif golden.literature is not None and current.literature is not None:
        g_lit, c_lit = golden.literature, current.literature
        g_n, c_n = len(g_lit.pmids), len(c_lit.pmids)
        if abs(g_n - c_n) > EVIDENCE_COUNT_TOLERANCE:
            diffs.append(
                Diff(
                    path=f"{path_prefix}.literature.pmids",
                    kind="count_drift",
                    severity="warn",
                    detail=(
                        f"PMID count drifted by {abs(g_n - c_n)} "
                        f"(>{EVIDENCE_COUNT_TOLERANCE} tolerance)"
                    ),
                    golden=g_n,
                    current=c_n,
                )
            )
        if g_lit.evidence_summary is not None and c_lit.evidence_summary is None:
            diffs.append(
                Diff(
                    path=f"{path_prefix}.literature.evidence_summary",
                    kind="presence_changed",
                    severity="error",
                    detail="evidence_summary missing in current",
                )
            )

    if golden.clinical_trials is not None and current.clinical_trials is None:
        diffs.append(
            Diff(
                path=f"{path_prefix}.clinical_trials",
                kind="presence_changed",
                severity="error",
                detail="clinical_trials populated in golden but not current",
            )
        )
    elif golden.clinical_trials is not None and current.clinical_trials is not None:
        diffs.extend(
            _compare_trial_counts(
                golden.clinical_trials, current.clinical_trials, path_prefix
            )
        )

    if golden.blurb is not None and current.blurb is not None:
        prose_len = len(current.blurb.prose)
        if prose_len < BLURB_PROSE_MIN_LEN or prose_len > BLURB_PROSE_MAX_LEN:
            diffs.append(
                Diff(
                    path=f"{path_prefix}.blurb.prose",
                    kind="length_out_of_bounds",
                    severity="warn",
                    detail=(
                        f"prose length {prose_len} outside "
                        f"[{BLURB_PROSE_MIN_LEN}, {BLURB_PROSE_MAX_LEN}]"
                    ),
                    golden=len(golden.blurb.prose),
                    current=prose_len,
                )
            )
    return diffs


def _compare_trial_counts(golden: Any, current: Any, path_prefix: str) -> list[Diff]:
    diffs: list[Diff] = []
    for section in ("search", "completed", "terminated"):
        g_sec = getattr(golden, section, None)
        c_sec = getattr(current, section, None)
        if g_sec is None or c_sec is None:
            continue
        g_total = getattr(g_sec, "total_count", None)
        c_total = getattr(c_sec, "total_count", None)
        if g_total is None or c_total is None:
            continue
        if abs(g_total - c_total) > EVIDENCE_COUNT_TOLERANCE:
            diffs.append(
                Diff(
                    path=f"{path_prefix}.clinical_trials.{section}.total_count",
                    kind="count_drift",
                    severity="warn",
                    detail=(
                        f"trial total_count drifted by {abs(g_total - c_total)} "
                        f"(>{EVIDENCE_COUNT_TOLERANCE} tolerance)"
                    ),
                    golden=g_total,
                    current=c_total,
                )
            )
    return diffs


def _compare_summary(golden: SupervisorOutput, current: SupervisorOutput) -> list[Diff]:
    diffs: list[Diff] = []
    n = len(current.summary)
    if n < SUMMARY_MIN_LEN:
        diffs.append(
            Diff(
                path="summary",
                kind="length_out_of_bounds",
                severity="error",
                detail=f"summary length {n} < min {SUMMARY_MIN_LEN}",
                golden=len(golden.summary),
                current=n,
            )
        )
    elif n > SUMMARY_MAX_LEN:
        diffs.append(
            Diff(
                path="summary",
                kind="length_out_of_bounds",
                severity="warn",
                detail=f"summary length {n} > max {SUMMARY_MAX_LEN}",
                golden=len(golden.summary),
                current=n,
            )
        )
    return diffs


def has_errors(diffs: list[Diff]) -> bool:
    return any(d.severity == "error" for d in diffs)


def render_diffs(diffs: list[Diff]) -> str:
    """Compact human-readable table for the CLI and pytest failure output."""
    if not diffs:
        return "no diffs"
    lines = [f"{'SEV':<6} {'PATH':<60} KIND / DETAIL"]
    for d in diffs:
        lines.append(f"{d.severity:<6} {d.path:<60} {d.kind}: {d.detail}")
    return "\n".join(lines)
