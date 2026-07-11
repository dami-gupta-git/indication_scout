"""Per-assertion functions that run a `DrugSpec` against a `SupervisorOutput`.

Each function returns a list of `BucketedDiff` records — empty if the
assertion passed, one or more diffs if it failed. The Layer 2 test collects
all diffs across all assertions, then fails if any are error-severity.

Indications are compared case-insensitively after stripping whitespace, since
the supervisor and the spec author may capitalize differently.
"""

from __future__ import annotations

from typing import Iterable

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)

from tests.regression.common.failure_buckets import BucketedDiff
from tests.regression.layer2_structural.spec import (
    CandidateSetContains,
    DrugSpec,
    ForbiddenInRanked,
    ForbiddenPhrase,
    RankedOrder,
    RequiredInRanked,
    RequiredNCTs,
    RequiredPMIDs,
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _find_finding(report: SupervisorOutput, indication: str) -> CandidateFindings | None:
    target = _norm(indication)
    for f in report.disease_findings:
        if _norm(f.disease) == target:
            return f
    return None


def _ncts_in_section(finding: CandidateFindings, section: str) -> set[str]:
    ct = finding.clinical_trials
    if ct is None:
        return set()
    # The curated set the report surfaces to the user.
    if section == "relevant":
        return set(ct.relevant_nct_ids or [])
    out: set[str] = set()
    pools: Iterable = ()
    if section in ("completed", "any") and ct.completed is not None:
        pools = (*pools, ct.completed.trials or [])
    if section in ("terminated", "any") and ct.terminated is not None:
        pools = (*pools, ct.terminated.trials or [])
    if section in ("search", "any") and ct.search is not None:
        pools = (*pools, ct.search.trials or [])
    for pool in pools:
        for t in pool:
            nct = getattr(t, "nct_id", None)
            if nct:
                out.add(nct)
    return out


def check_required_ncts(report: SupervisorOutput, a: RequiredNCTs) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication)
    if finding is None:
        return [
            BucketedDiff(
                bucket=a.bucket,
                path=f"disease_findings[{a.indication!r}]",
                severity="error",
                detail=f"indication not present in disease_findings",
                spec_ref="required_ncts_surfaced",
            )
        ]
    found = _ncts_in_section(finding, a.section)
    missing = [n for n in a.ncts if n not in found]
    if not missing:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path=f"disease_findings[{a.indication!r}].clinical_trials.{a.section}",
            severity="error",
            detail=f"missing NCTs: {missing}",
            spec_ref="required_ncts_surfaced",
        )
    ]


def _cited_pmids(finding: CandidateFindings) -> set[str]:
    """Supporting + contradicting PMIDs from the curated evidence summary."""
    lit = finding.literature
    if lit is None or lit.evidence_summary is None:
        return set()
    es = lit.evidence_summary
    return set(es.supporting_pmids or []) | set(es.contradicting_pmids or [])


def check_required_pmids(report: SupervisorOutput, a: RequiredPMIDs) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication)
    if finding is None or finding.literature is None:
        return [
            BucketedDiff(
                bucket=a.bucket,
                path=f"disease_findings[{a.indication!r}].literature",
                severity="error",
                detail="literature block missing for indication",
                spec_ref="required_pmids_cited",
            )
        ]
    if a.mode == "cited":
        found = _cited_pmids(finding)
        path = f"disease_findings[{a.indication!r}].literature.evidence_summary"
    else:
        found = set(finding.literature.pmids or [])
        path = f"disease_findings[{a.indication!r}].literature.pmids"
    missing = [p for p in a.pmids if p not in found]
    if not missing:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path=path,
            severity="error",
            detail=f"missing PMIDs ({a.mode}): {missing}",
            spec_ref="required_pmids_cited",
        )
    ]


def check_required_in_ranked(
    report: SupervisorOutput, a: RequiredInRanked
) -> list[BucketedDiff]:
    ranked = {_norm(d) for d in report.top_diseases}
    if _norm(a.indication) in ranked:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path="top_diseases",
            severity="error",
            detail=f"required indication {a.indication!r} not in top_diseases",
            spec_ref="required_in_ranked",
        )
    ]


def check_ranked_order(report: SupervisorOutput, a: RankedOrder) -> list[BucketedDiff]:
    ranked = [_norm(d) for d in report.top_diseases]
    wanted = [_norm(i) for i in a.indications]
    missing = [orig for orig, n in zip(a.indications, wanted) if n not in ranked]
    if missing:
        return [
            BucketedDiff(
                bucket=a.bucket,
                path="top_diseases",
                severity="error",
                detail=f"ranked-order indications not in top_diseases: {missing}",
                spec_ref="ranked_order",
            )
        ]
    positions = [ranked.index(n) for n in wanted]
    if positions == sorted(positions):
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path="top_diseases",
            severity="error",
            detail=(
                f"ranked order violated: expected {a.indications}, "
                f"got {report.top_diseases}"
            ),
            spec_ref="ranked_order",
        )
    ]


def check_forbidden_in_ranked(
    report: SupervisorOutput, a: ForbiddenInRanked
) -> list[BucketedDiff]:
    ranked = {_norm(d) for d in report.top_diseases}
    if _norm(a.indication) not in ranked:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path="top_diseases",
            severity="error",
            detail=f"forbidden indication {a.indication!r} appeared in top_diseases",
            spec_ref="forbidden_in_ranked",
        )
    ]


def check_forbidden_phrase(
    report: SupervisorOutput, rendered_md: str, a: ForbiddenPhrase
) -> list[BucketedDiff]:
    needle = a.phrase.lower()
    if a.scope == "summary":
        haystack = (report.summary or "").lower()
    elif a.scope == "blurb":
        haystack = "\n".join(
            (f.blurb.prose or "")
            for f in report.disease_findings
            if f.blurb is not None
        ).lower()
    else:
        haystack = rendered_md.lower()
    if needle not in haystack:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path=f"report[{a.scope}]",
            severity="error",
            detail=f"forbidden phrase {a.phrase!r} appeared",
            spec_ref="forbidden_phrases",
        )
    ]


def check_candidate_set_contains(
    report: SupervisorOutput, a: CandidateSetContains
) -> list[BucketedDiff]:
    found = {_norm(d) for d in report.candidate_diseases}
    missing = [i for i in a.indications if _norm(i) not in found]
    if not missing:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path="candidate_diseases",
            severity="error",
            detail=f"missing required candidates: {missing}",
            spec_ref="candidate_set_contains",
        )
    ]


def run_spec(
    spec: DrugSpec,
    report: SupervisorOutput,
    rendered_md: str = "",
) -> list[BucketedDiff]:
    """Run every assertion in `spec` against `report`. Returns all diffs."""
    diffs: list[BucketedDiff] = []
    for a in spec.required_ncts_surfaced:
        diffs.extend(check_required_ncts(report, a))
    for a in spec.required_pmids_cited:
        diffs.extend(check_required_pmids(report, a))
    for a in spec.required_in_ranked:
        diffs.extend(check_required_in_ranked(report, a))
    if spec.ranked_order is not None:
        diffs.extend(check_ranked_order(report, spec.ranked_order))
    for a in spec.forbidden_in_ranked:
        diffs.extend(check_forbidden_in_ranked(report, a))
    for a in spec.forbidden_phrases:
        diffs.extend(check_forbidden_phrase(report, rendered_md, a))
    if spec.candidate_set_contains is not None:
        diffs.extend(check_candidate_set_contains(report, spec.candidate_set_contains))
    return diffs
