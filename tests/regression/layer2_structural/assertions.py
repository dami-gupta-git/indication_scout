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
    DrugSafety,
    DrugSpec,
    ForbiddenInRanked,
    ForbiddenPhrase,
    IndicationHarm,
    RankedOrder,
    RequiredInRanked,
    RequiredNCTs,
    RequiredPMIDs,
    SafetySeverity,
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _alias_set(aliases: dict[str, list[str]], indication: str) -> set[str]:
    """All normalized names that count as `indication`: itself plus its aliases.

    Matches on either side — an alias key or any of its values resolves to the
    full group — so the spec author can write whichever name is canonical to
    them and still match a run that emitted a variant.
    """
    target = _norm(indication)
    for key, variants in aliases.items():
        group = {_norm(key), *(_norm(v) for v in variants)}
        if target in group:
            return group
    return {target}


def _find_finding(
    report: SupervisorOutput,
    indication: str,
    aliases: dict[str, list[str]] | None = None,
) -> CandidateFindings | None:
    names = _alias_set(aliases or {}, indication)
    for f in report.disease_findings:
        if _norm(f.disease) in names:
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


def check_required_ncts(
    report: SupervisorOutput,
    a: RequiredNCTs,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication, aliases)
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


def check_required_pmids(
    report: SupervisorOutput,
    a: RequiredPMIDs,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication, aliases)
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
    report: SupervisorOutput,
    a: RequiredInRanked,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    ranked = {_norm(d) for d in report.top_diseases}
    if _alias_set(aliases or {}, a.indication) & ranked:
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


def check_ranked_order(
    report: SupervisorOutput,
    a: RankedOrder,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    ranked = [_norm(d) for d in report.top_diseases]

    def _pos(indication: str) -> int | None:
        names = _alias_set(aliases or {}, indication)
        for i, name in enumerate(ranked):
            if name in names:
                return i
        return None

    positions_opt = [(orig, _pos(orig)) for orig in a.indications]
    missing = [orig for orig, p in positions_opt if p is None]
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
    positions = [p for _, p in positions_opt]
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
    report: SupervisorOutput,
    a: ForbiddenInRanked,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    ranked = {_norm(d) for d in report.top_diseases}
    if not (_alias_set(aliases or {}, a.indication) & ranked):
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
    report: SupervisorOutput,
    a: CandidateSetContains,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    found = {_norm(d) for d in report.candidate_diseases}
    missing = [i for i in a.indications if not (_alias_set(aliases or {}, i) & found)]
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


def _evidence_summary(finding: CandidateFindings):
    lit = finding.literature
    return lit.evidence_summary if lit is not None else None


def check_safety_severity(
    report: SupervisorOutput,
    a: SafetySeverity,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication, aliases)
    if finding is None:
        return [
            BucketedDiff(
                bucket=a.bucket,
                path=f"disease_findings[{a.indication!r}]",
                severity="error",
                detail="indication not present in disease_findings",
                spec_ref="safety_severity",
            )
        ]
    es = _evidence_summary(finding)
    got = es.safety_severity if es is not None else None
    allowed = {_norm(v) for v in a.allowed}
    if got is not None and _norm(got) in allowed:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path=f"disease_findings[{a.indication!r}].literature.evidence_summary.safety_severity",
            severity="error",
            detail=f"safety_severity {got!r} not in allowed {a.allowed}",
            spec_ref="safety_severity",
        )
    ]


def check_indication_harm(
    report: SupervisorOutput,
    a: IndicationHarm,
    aliases: dict[str, list[str]] | None = None,
) -> list[BucketedDiff]:
    finding = _find_finding(report, a.indication, aliases)
    if finding is None:
        return [
            BucketedDiff(
                bucket=a.bucket,
                path=f"disease_findings[{a.indication!r}]",
                severity="error",
                detail="indication not present in disease_findings",
                spec_ref="indication_harm",
            )
        ]
    es = _evidence_summary(finding)
    got = es.indication_harm if es is not None else None
    if got == a.expected:
        return []
    return [
        BucketedDiff(
            bucket=a.bucket,
            path=f"disease_findings[{a.indication!r}].literature.evidence_summary.indication_harm",
            severity="error",
            detail=f"indication_harm {got!r}, expected {a.expected!r}",
            spec_ref="indication_harm",
        )
    ]


def check_drug_safety(report: SupervisorOutput, a: DrugSafety) -> list[BucketedDiff]:
    diffs: list[BucketedDiff] = []
    present = bool((report.drug_safety_summary or "").strip())
    if a.summary_present and not present:
        diffs.append(
            BucketedDiff(
                bucket=a.bucket,
                path="drug_safety_summary",
                severity="error",
                detail="drug_safety_summary is empty; expected a collapsed drug-wide signal",
                spec_ref="drug_safety",
            )
        )
    elif not a.summary_present and present:
        diffs.append(
            BucketedDiff(
                bucket=a.bucket,
                path="drug_safety_summary",
                severity="error",
                detail="drug_safety_summary is non-empty; expected none",
                spec_ref="drug_safety",
            )
        )
    found = set(report.drug_safety_pmids or [])
    missing = [p for p in a.required_pmids if p not in found]
    if missing:
        diffs.append(
            BucketedDiff(
                bucket=a.bucket,
                path="drug_safety_pmids",
                severity="error",
                detail=f"missing drug-level safety PMIDs: {missing}",
                spec_ref="drug_safety",
            )
        )
    return diffs


def run_spec(
    spec: DrugSpec,
    report: SupervisorOutput,
    rendered_md: str = "",
) -> list[BucketedDiff]:
    """Run every assertion in `spec` against `report`. Returns all diffs."""
    aliases = spec.aliases
    diffs: list[BucketedDiff] = []
    for a in spec.required_ncts_surfaced:
        diffs.extend(check_required_ncts(report, a, aliases))
    for a in spec.required_pmids_cited:
        diffs.extend(check_required_pmids(report, a, aliases))
    for a in spec.required_in_ranked:
        diffs.extend(check_required_in_ranked(report, a, aliases))
    if spec.ranked_order is not None:
        diffs.extend(check_ranked_order(report, spec.ranked_order, aliases))
    for a in spec.forbidden_in_ranked:
        diffs.extend(check_forbidden_in_ranked(report, a, aliases))
    for a in spec.forbidden_phrases:
        diffs.extend(check_forbidden_phrase(report, rendered_md, a))
    if spec.candidate_set_contains is not None:
        diffs.extend(check_candidate_set_contains(report, spec.candidate_set_contains, aliases))
    for a in spec.safety_severity:
        diffs.extend(check_safety_severity(report, a, aliases))
    for a in spec.indication_harm:
        diffs.extend(check_indication_harm(report, a, aliases))
    if spec.drug_safety is not None:
        diffs.extend(check_drug_safety(report, spec.drug_safety))
    return diffs
