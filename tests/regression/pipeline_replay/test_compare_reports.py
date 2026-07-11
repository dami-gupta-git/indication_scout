"""Unit tests for compare_reports. No LLM, no network."""

from __future__ import annotations

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import (
    MechanismCandidate,
    MechanismOutput,
)
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.models.model_clinical_trials import SearchTrialsResult
from indication_scout.regression.diff import jaccard
from indication_scout.regression.harness import compare_reports, has_errors


def _make_report(
    *,
    drug: str = "metformin",
    candidates: list[str] | None = None,
    top: list[str] | None = None,
    summary: str = "A" * 200,
    mechanism_targets: dict[str, str] | None = None,
    mechanism_candidates: list[str] | None = None,
    with_findings_for: list[str] | None = None,
    pmids_per_disease: int = 10,
    trial_total: int = 50,
) -> SupervisorOutput:
    candidates = candidates if candidates is not None else ["dm2", "pcos", "cancer"]
    top = top if top is not None else candidates[:2]
    with_findings_for = with_findings_for if with_findings_for is not None else candidates

    findings = [
        CandidateFindings(
            disease=d,
            literature=LiteratureOutput(pmids=[f"PMID{i}" for i in range(pmids_per_disease)]),
            clinical_trials=ClinicalTrialsOutput(
                search=SearchTrialsResult(total_count=trial_total)
            ),
            blurb=CandidateBlurb(prose="A" * 100) if d in top else None,
        )
        for d in with_findings_for
    ]

    mechanism = None
    if mechanism_targets is not None:
        mechanism = MechanismOutput(
            drug_targets=mechanism_targets,
            candidates=[
                MechanismCandidate(disease_name=name) for name in (mechanism_candidates or [])
            ],
        )

    return SupervisorOutput(
        drug_name=drug,
        candidate_diseases=candidates,
        top_diseases=top,
        disease_findings=findings,
        mechanism=mechanism,
        summary=summary,
    )


class TestJaccard:
    def test_two_empty_sets_match(self):
        assert jaccard(set(), set()) == 1.0

    def test_identical_sets(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


class TestCompareReports:
    def test_identical_reports_have_no_errors(self):
        g = _make_report()
        c = _make_report()
        diffs = compare_reports(g, c)
        assert not has_errors(diffs), [d.detail for d in diffs if d.severity == "error"]

    def test_drug_name_mismatch_is_error(self):
        g = _make_report(drug="metformin")
        c = _make_report(drug="aspirin")
        diffs = compare_reports(g, c)
        assert has_errors(diffs)
        assert any(d.path == "drug_name" for d in diffs)

    def test_empty_candidates_is_error(self):
        g = _make_report(candidates=["dm2", "pcos"])
        c = _make_report(
            candidates=[], top=[], with_findings_for=[], mechanism_targets=None
        )
        diffs = compare_reports(g, c)
        assert has_errors(diffs)
        assert any(d.path == "candidate_diseases" and d.kind == "empty" for d in diffs)

    def test_candidate_set_divergence_below_threshold_is_error(self):
        g = _make_report(candidates=["a", "b", "c", "d"], top=["a", "b"])
        c = _make_report(candidates=["w", "x", "y", "z"], top=["w", "x"])
        diffs = compare_reports(g, c)
        assert has_errors(diffs)
        assert any(
            d.path == "candidate_diseases" and d.kind == "set_divergence" for d in diffs
        )

    def test_top_diseases_must_be_subset_of_findings(self):
        # Top contains a disease not in disease_findings — invariant violation.
        g = _make_report()
        c = _make_report(
            candidates=["dm2", "pcos"],
            top=["dm2", "ghost"],
            with_findings_for=["dm2", "pcos"],
        )
        diffs = compare_reports(g, c)
        assert any(
            d.path == "top_diseases" and d.kind == "invariant_violation" for d in diffs
        )

    def test_summary_too_short_is_error(self):
        g = _make_report(summary="A" * 200)
        c = _make_report(summary="short")
        diffs = compare_reports(g, c)
        assert any(
            d.path == "summary" and d.kind == "length_out_of_bounds" and d.severity == "error"
            for d in diffs
        )

    def test_mechanism_target_drift_is_error(self):
        g = _make_report(mechanism_targets={"TARGET_A": "ENS1"})
        c = _make_report(mechanism_targets={"TARGET_B": "ENS2"})
        diffs = compare_reports(g, c)
        assert any(
            d.path == "mechanism.drug_targets" and d.kind == "set_divergence" for d in diffs
        )

    def test_mechanism_presence_change_is_error(self):
        g = _make_report(mechanism_targets={"TARGET_A": "ENS1"})
        c = _make_report(mechanism_targets=None)
        diffs = compare_reports(g, c)
        assert any(d.path == "mechanism" and d.kind == "presence_changed" for d in diffs)

    def test_pmid_count_drift_within_tolerance_is_not_error(self):
        g = _make_report(pmids_per_disease=10)
        c = _make_report(pmids_per_disease=12)  # diff = 2, tolerance = 5
        diffs = compare_reports(g, c)
        assert not has_errors(diffs)

    def test_pmid_count_drift_beyond_tolerance_is_warn(self):
        g = _make_report(pmids_per_disease=10)
        c = _make_report(pmids_per_disease=30)  # diff = 20
        diffs = compare_reports(g, c)
        # Drift is a warning, not an error — we don't want exact-count
        # assertions blowing up the suite on every PubMed result reshuffle.
        assert not has_errors(diffs)
        assert any(d.kind == "count_drift" and d.severity == "warn" for d in diffs)

    def test_trial_total_drift_beyond_tolerance_is_warn(self):
        g = _make_report(trial_total=100)
        c = _make_report(trial_total=200)
        diffs = compare_reports(g, c)
        assert any(
            d.kind == "count_drift"
            and "clinical_trials" in d.path
            and d.severity == "warn"
            for d in diffs
        )
