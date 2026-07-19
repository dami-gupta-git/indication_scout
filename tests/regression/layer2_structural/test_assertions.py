"""Unit tests for the Layer 2 assertion functions. No LLM, no network."""

from __future__ import annotations

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary

from tests.regression.common.failure_buckets import Bucket, has_errors
from tests.regression.layer2_structural.assertions import (
    check_candidate_set_contains,
    check_drug_safety,
    check_forbidden_in_ranked,
    check_forbidden_phrase,
    check_indication_harm,
    check_ranked_order,
    check_required_in_ranked,
    check_required_ncts,
    check_required_pmids,
    check_safety_severity,
    run_spec,
)
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


def _trial(nct: str) -> Trial:
    return Trial(nct_id=nct)


def _finding(
    *,
    disease: str,
    completed: list[str] | None = None,
    terminated: list[str] | None = None,
    relevant: list[str] | None = None,
    supporting_pmids: list[str] | None = None,
    contradicting_pmids: list[str] | None = None,
    pool_pmids: list[str] | None = None,
    safety_severity: str = "none",
    indication_harm: bool = False,
    safety_pmids: list[str] | None = None,
) -> CandidateFindings:
    """Build a finding.

    `relevant` populates clinical_trials.relevant_nct_ids (the curated set the
    "relevant" NCT section reads). `supporting_pmids`/`contradicting_pmids`
    populate evidence_summary (the "cited" PMID mode). `pool_pmids` populates
    literature.pmids (the raw-pool "pool" mode). `completed`/`terminated`
    populate the trial lists used by the "completed"/"terminated" sections.
    """
    completed = completed or []
    terminated = terminated or []
    relevant = relevant or []
    supporting_pmids = supporting_pmids or []
    contradicting_pmids = contradicting_pmids or []
    pool_pmids = pool_pmids or []
    safety_pmids = safety_pmids or []
    return CandidateFindings(
        disease=disease,
        literature=LiteratureOutput(
            pmids=pool_pmids,
            evidence_summary=EvidenceSummary(
                supporting_pmids=supporting_pmids,
                contradicting_pmids=contradicting_pmids,
                safety_severity=safety_severity,
                indication_harm=indication_harm,
                safety_pmids=safety_pmids,
            ),
        ),
        clinical_trials=ClinicalTrialsOutput(
            search=SearchTrialsResult(total_count=len(completed) + len(terminated)),
            completed=CompletedTrialsResult(
                total_count=len(completed),
                trials=[_trial(n) for n in completed],
            ),
            terminated=TerminatedTrialsResult(
                total_count=len(terminated),
                trials=[_trial(n) for n in terminated],
            ),
            relevant_nct_ids=relevant,
        ),
    )


def _report(
    *,
    drug: str = "bupropion",
    findings: list[CandidateFindings] | None = None,
    top: list[str] | None = None,
    candidates: list[str] | None = None,
    summary: str = "A" * 200,
    drug_safety_summary: str = "",
    drug_safety_pmids: list[str] | None = None,
) -> SupervisorOutput:
    findings = findings or []
    if top is None:
        top = [f.disease for f in findings[:2]]
    if candidates is None:
        candidates = [f.disease for f in findings]
    return SupervisorOutput(
        drug_name=drug,
        candidate_diseases=candidates,
        top_diseases=top,
        disease_findings=findings,
        summary=summary,
        drug_safety_summary=drug_safety_summary,
        drug_safety_pmids=drug_safety_pmids or [],
    )


class TestRequiredNCTs:
    def test_passes_when_relevant_ncts_present(self):
        # Default section is "relevant" — reads clinical_trials.relevant_nct_ids.
        f = _finding(disease="adhd", relevant=["NCT00048360", "NCT00061087"])
        r = _report(findings=[f])
        a = RequiredNCTs(indication="adhd", ncts=["NCT00048360"])
        assert check_required_ncts(r, a) == []

    def test_relevant_ignores_completed_pool(self):
        # An NCT that's in completed.trials but NOT relevant_nct_ids must fail
        # the default "relevant" section — the report didn't surface it.
        f = _finding(disease="adhd", completed=["NCT00048360"], relevant=[])
        r = _report(findings=[f])
        a = RequiredNCTs(indication="adhd", ncts=["NCT00048360"])
        diffs = check_required_ncts(r, a)
        assert len(diffs) == 1
        assert "NCT00048360" in diffs[0].detail

    def test_passes_when_completed_ncts_present(self):
        f = _finding(disease="adhd", completed=["NCT00048360", "NCT00061087"])
        r = _report(findings=[f])
        a = RequiredNCTs(indication="adhd", section="completed", ncts=["NCT00048360"])
        assert check_required_ncts(r, a) == []

    def test_fails_when_indication_missing(self):
        r = _report(findings=[_finding(disease="other")])
        a = RequiredNCTs(indication="adhd", ncts=["NCT00048360"])
        diffs = check_required_ncts(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.LITERATURE_COVERAGE
        assert "not present" in diffs[0].detail

    def test_fails_when_nct_missing(self):
        f = _finding(disease="adhd", relevant=["NCT00048360"])
        r = _report(findings=[f])
        a = RequiredNCTs(
            indication="adhd",
            ncts=["NCT00048360", "NCT99999999"],
        )
        diffs = check_required_ncts(r, a)
        assert len(diffs) == 1
        assert "NCT99999999" in diffs[0].detail

    def test_case_insensitive_indication_match(self):
        f = _finding(disease="ADHD", relevant=["NCT00048360"])
        r = _report(findings=[f])
        a = RequiredNCTs(indication="adhd", ncts=["NCT00048360"])
        assert check_required_ncts(r, a) == []

    def test_section_any_searches_both_lists(self):
        f = _finding(
            disease="cocaine",
            completed=["NCT01077024"],
            terminated=["NCT00000276"],
        )
        r = _report(findings=[f])
        a = RequiredNCTs(
            indication="cocaine", section="any", ncts=["NCT01077024", "NCT00000276"]
        )
        assert check_required_ncts(r, a) == []


class TestRequiredPMIDs:
    def test_cited_passes_for_supporting_and_contradicting(self):
        # Default mode is "cited" — reads evidence_summary supporting + contradicting.
        f = _finding(
            disease="cocaine",
            supporting_pmids=["16461866"],
            contradicting_pmids=["18551884"],
        )
        r = _report(findings=[f])
        a = RequiredPMIDs(indication="cocaine", pmids=["16461866", "18551884"])
        assert check_required_pmids(r, a) == []

    def test_cited_ignores_raw_pool(self):
        # A PMID in the raw literature.pmids pool but NOT cited must fail the
        # default "cited" mode.
        f = _finding(disease="schizophrenia", pool_pmids=["24201231"])
        r = _report(findings=[f])
        a = RequiredPMIDs(indication="schizophrenia", pmids=["24201231"])
        diffs = check_required_pmids(r, a)
        assert len(diffs) == 1
        assert "24201231" in diffs[0].detail
        assert "cited" in diffs[0].detail

    def test_pool_mode_reads_raw_pool(self):
        f = _finding(disease="schizophrenia", pool_pmids=["24201231", "1234567"])
        r = _report(findings=[f])
        a = RequiredPMIDs(indication="schizophrenia", mode="pool", pmids=["24201231"])
        assert check_required_pmids(r, a) == []

    def test_fails_when_pmid_missing(self):
        f = _finding(disease="schizophrenia", supporting_pmids=["999"])
        r = _report(findings=[f])
        a = RequiredPMIDs(indication="schizophrenia", pmids=["24201231"])
        diffs = check_required_pmids(r, a)
        assert len(diffs) == 1
        assert "24201231" in diffs[0].detail


class TestRankedOrder:
    def test_passes_when_order_matches(self):
        r = _report(
            findings=[_finding(disease=d) for d in ("a", "b", "c")],
            top=["a", "b", "c"],
        )
        a = RankedOrder(indications=["a", "b", "c"])
        assert check_ranked_order(r, a) == []

    def test_passes_ignoring_interleaved_others(self):
        r = _report(
            findings=[_finding(disease=d) for d in ("a", "x", "b")],
            top=["a", "x", "b"],
        )
        a = RankedOrder(indications=["a", "b"])
        assert check_ranked_order(r, a) == []

    def test_fails_when_order_reversed(self):
        r = _report(
            findings=[_finding(disease=d) for d in ("a", "b")],
            top=["b", "a"],
        )
        a = RankedOrder(indications=["a", "b"])
        diffs = check_ranked_order(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.RANKING

    def test_fails_when_indication_absent(self):
        r = _report(findings=[_finding(disease="a")], top=["a"])
        a = RankedOrder(indications=["a", "b"])
        diffs = check_ranked_order(r, a)
        assert len(diffs) == 1
        assert "b" in diffs[0].detail


class TestRequiredInRanked:
    def test_passes_when_present(self):
        r = _report(findings=[_finding(disease="adhd")], top=["adhd"])
        a = RequiredInRanked(indication="adhd")
        assert check_required_in_ranked(r, a) == []

    def test_fails_when_absent(self):
        r = _report(findings=[_finding(disease="other")], top=["other"])
        a = RequiredInRanked(indication="adhd")
        diffs = check_required_in_ranked(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.RANKING


class TestForbiddenInRanked:
    def test_fails_when_demotion_didnt_fire(self):
        # Obesity should have been demoted; it's still in top_diseases.
        r = _report(
            findings=[_finding(disease="obesity")],
            top=["obesity"],
        )
        a = ForbiddenInRanked(indication="obesity")
        diffs = check_forbidden_in_ranked(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.DEMOTION_LOGIC

    def test_passes_when_demotion_fired(self):
        r = _report(findings=[_finding(disease="obesity")], top=[])
        a = ForbiddenInRanked(indication="obesity")
        assert check_forbidden_in_ranked(r, a) == []


class TestForbiddenPhrase:
    def test_fails_when_phrase_in_summary(self):
        r = _report(
            findings=[_finding(disease="obesity")],
            summary="bupropion is approved for obesity " + "x" * 200,
        )
        a = ForbiddenPhrase(phrase="approved for obesity", scope="summary")
        diffs = check_forbidden_phrase(r, "", a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.FACTUAL_ACCURACY

    def test_passes_when_phrase_absent(self):
        r = _report(findings=[_finding(disease="obesity")], summary="A" * 200)
        a = ForbiddenPhrase(phrase="approved for obesity", scope="summary")
        assert check_forbidden_phrase(r, "", a) == []

    def test_case_insensitive(self):
        r = _report(
            findings=[_finding(disease="obesity")],
            summary="Bupropion is APPROVED FOR OBESITY " + "x" * 200,
        )
        a = ForbiddenPhrase(phrase="approved for obesity", scope="summary")
        assert len(check_forbidden_phrase(r, "", a)) == 1


class TestCandidateSetContains:
    def test_passes_when_all_present(self):
        r = _report(
            findings=[_finding(disease="adhd"), _finding(disease="obesity")],
            candidates=["adhd", "obesity", "other"],
        )
        a = CandidateSetContains(indications=["adhd", "obesity"])
        assert check_candidate_set_contains(r, a) == []

    def test_fails_with_missing(self):
        r = _report(
            findings=[_finding(disease="adhd")],
            candidates=["adhd"],
        )
        a = CandidateSetContains(indications=["adhd", "schizophrenia"])
        diffs = check_candidate_set_contains(r, a)
        assert len(diffs) == 1
        assert "schizophrenia" in diffs[0].detail


class TestRunSpec:
    def test_clean_report_against_realistic_spec(self):
        # End-to-end happy path: a spec with every assertion type, a report
        # that satisfies all of them. Confirms run_spec aggregates correctly.
        r = _report(
            findings=[
                _finding(
                    disease="adhd",
                    relevant=["NCT00048360"],
                    supporting_pmids=["24201231"],
                ),
                _finding(disease="obesity"),
            ],
            top=["adhd"],
            candidates=["adhd", "obesity"],
            summary="Clean summary " + "x" * 200,
        )
        spec = DrugSpec(
            drug="bupropion",
            required_ncts_surfaced=[
                RequiredNCTs(indication="adhd", ncts=["NCT00048360"]),
            ],
            required_pmids_cited=[
                RequiredPMIDs(indication="adhd", pmids=["24201231"]),
            ],
            required_in_ranked=[RequiredInRanked(indication="adhd")],
            ranked_order=RankedOrder(indications=["adhd"]),
            forbidden_in_ranked=[ForbiddenInRanked(indication="obesity")],
            forbidden_phrases=[
                ForbiddenPhrase(phrase="approved for obesity", scope="summary"),
            ],
            candidate_set_contains=CandidateSetContains(indications=["adhd", "obesity"]),
        )
        assert run_spec(spec, r) == []

    def test_multiple_failures_aggregate(self):
        # Empty report — every assertion fires. Confirms run_spec collects
        # rather than short-circuiting.
        r = _report(findings=[], top=[], candidates=[])
        spec = DrugSpec(
            drug="bupropion",
            required_in_ranked=[
                RequiredInRanked(indication="adhd"),
                RequiredInRanked(indication="schizophrenia"),
            ],
            candidate_set_contains=CandidateSetContains(
                indications=["adhd", "obesity"]
            ),
        )
        diffs = run_spec(spec, r)
        assert has_errors(diffs)
        assert len(diffs) == 3  # 2 ranking failures + 1 candidate-set failure


class TestSafetySeverity:
    def test_passes_when_in_allowed_set(self):
        f = _finding(disease="pain", safety_severity="withdrawn")
        r = _report(findings=[f])
        a = SafetySeverity(indication="pain", allowed=["withdrawn", "serious"])
        assert check_safety_severity(r, a) == []

    def test_fails_when_not_in_allowed_set(self):
        f = _finding(disease="pain", safety_severity="moderate")
        r = _report(findings=[f])
        a = SafetySeverity(indication="pain", allowed=["withdrawn", "serious"])
        diffs = check_safety_severity(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.FACTUAL_ACCURACY
        assert "moderate" in diffs[0].detail

    def test_fails_when_indication_missing(self):
        r = _report(findings=[_finding(disease="other")])
        a = SafetySeverity(indication="pain", allowed=["withdrawn"])
        diffs = check_safety_severity(r, a)
        assert len(diffs) == 1
        assert "not present" in diffs[0].detail


class TestIndicationHarm:
    def test_passes_when_matches(self):
        f = _finding(disease="pain", indication_harm=True)
        r = _report(findings=[f])
        a = IndicationHarm(indication="pain", expected=True)
        assert check_indication_harm(r, a) == []

    def test_fails_when_mismatch(self):
        f = _finding(disease="pain", indication_harm=False)
        r = _report(findings=[f])
        a = IndicationHarm(indication="pain", expected=True)
        diffs = check_indication_harm(r, a)
        assert len(diffs) == 1
        assert diffs[0].bucket == Bucket.FACTUAL_ACCURACY


class TestDrugSafety:
    def test_passes_when_summary_present(self):
        r = _report(
            findings=[_finding(disease="pain")],
            drug_safety_summary="Withdrawn in 2004 for cardiovascular harm.",
            drug_safety_pmids=["15713943"],
        )
        a = DrugSafety(summary_present=True, required_pmids=["15713943"])
        assert check_drug_safety(r, a) == []

    def test_fails_when_summary_expected_but_empty(self):
        r = _report(findings=[_finding(disease="pain")], drug_safety_summary="")
        a = DrugSafety(summary_present=True)
        diffs = check_drug_safety(r, a)
        assert len(diffs) == 1
        assert "empty" in diffs[0].detail

    def test_fails_when_required_pmid_missing(self):
        r = _report(
            findings=[_finding(disease="pain")],
            drug_safety_summary="Withdrawn.",
            drug_safety_pmids=["111"],
        )
        a = DrugSafety(summary_present=True, required_pmids=["15713943"])
        diffs = check_drug_safety(r, a)
        assert len(diffs) == 1
        assert "15713943" in diffs[0].detail


class TestAliasing:
    def test_finding_resolves_alias(self):
        # Report emits "cocaine dependence"; spec written as canonical
        # "cocaine use disorder" must still match via the alias table.
        f = _finding(disease="cocaine dependence", relevant=["NCT02111798"])
        r = _report(findings=[f])
        a = RequiredNCTs(indication="cocaine use disorder", ncts=["NCT02111798"])
        aliases = {"cocaine use disorder": ["cocaine dependence"]}
        assert check_required_ncts(r, a, aliases) == []

    def test_ranked_resolves_alias(self):
        r = _report(
            findings=[_finding(disease="coronary artery disorder")],
            top=["coronary artery disorder"],
        )
        a = RequiredInRanked(indication="cardiovascular disease")
        aliases = {"cardiovascular disease": ["coronary artery disorder"]}
        assert check_required_in_ranked(r, a, aliases) == []

    def test_ranked_order_resolves_alias(self):
        r = _report(
            findings=[_finding(disease=d) for d in ("masld", "cad")],
            top=["masld", "cad"],
        )
        a = RankedOrder(indications=["hepatic steatosis", "cardiovascular disease"])
        aliases = {
            "hepatic steatosis": ["masld"],
            "cardiovascular disease": ["cad"],
        }
        assert check_ranked_order(r, a, aliases) == []

    def test_candidate_set_resolves_alias(self):
        r = _report(
            findings=[_finding(disease="masld")],
            candidates=["masld"],
        )
        a = CandidateSetContains(indications=["hepatic steatosis"])
        aliases = {"hepatic steatosis": ["masld"]}
        assert check_candidate_set_contains(r, a, aliases) == []

    def test_forbidden_resolves_alias(self):
        # Forbidden indication appears under an alias — must still fire.
        r = _report(
            findings=[_finding(disease="obesity disorder")],
            top=["obesity disorder"],
        )
        a = ForbiddenInRanked(indication="obesity")
        aliases = {"obesity": ["obesity disorder"]}
        diffs = check_forbidden_in_ranked(r, a, aliases)
        assert len(diffs) == 1

    def test_run_spec_threads_aliases(self):
        # End-to-end: spec canonical names differ from report names; aliases
        # bridge them so run_spec reports no diffs.
        r = _report(
            findings=[
                _finding(
                    disease="cocaine dependence",
                    safety_severity="black_box",
                    indication_harm=True,
                ),
            ],
            top=["cocaine dependence"],
            candidates=["cocaine dependence"],
            drug_safety_summary="Black-box warnings.",
        )
        spec = DrugSpec(
            drug="bupropion",
            aliases={"cocaine use disorder": ["cocaine dependence"]},
            required_in_ranked=[RequiredInRanked(indication="cocaine use disorder")],
            safety_severity=[
                SafetySeverity(
                    indication="cocaine use disorder", allowed=["black_box"]
                )
            ],
            indication_harm=[
                IndicationHarm(indication="cocaine use disorder", expected=True)
            ],
            drug_safety=DrugSafety(summary_present=True),
        )
        assert run_spec(spec, r) == []
