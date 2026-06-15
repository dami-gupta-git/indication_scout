"""Unit tests for the markdown report formatter.

Covers the clinical-trials section rewrite (search / completed / terminated /
landscape / approval) and the top-level format_report assembly.
"""

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.report.format_report import (
    _fmt_clinical_trials,
    _fmt_literature,
    format_report,
)


def test_fmt_literature_renders_direction_and_both_pmid_lists():
    lit = LiteratureOutput(
        evidence_summary=EvidenceSummary(
            summary="Gefitinib showed no benefit (PMID: 16062074).",
            study_count=4,
            strength="strong",
            direction="contradicts",
            key_findings=["No PFS benefit (PMID: 16062074)"],
            supporting_pmids=["21220480"],
            contradicting_pmids=["16062074", "18667394"],
        )
    )
    out = _fmt_literature(lit)
    assert "**Evidence strength:** strong, contradicts" in out
    assert "**Relevant studies:** 4" in out
    assert "**Supporting PMIDs:** [21220480]" in out
    assert "**Contradicting PMIDs:** [16062074]" in out
    assert "18667394" in out


def test_fmt_literature_omits_direction_none_and_empty_contradicting():
    lit = LiteratureOutput(
        evidence_summary=EvidenceSummary(
            summary="Supports repurposing (PMID: 12345678).",
            study_count=3,
            strength="moderate",
            direction="none",
            supporting_pmids=["12345678"],
        )
    )
    out = _fmt_literature(lit)
    assert "**Evidence strength:** moderate" in out
    assert "contradicts" not in out
    assert "Contradicting PMIDs" not in out


def test_fmt_literature_class_level_basis_renders_honest_strength_line():
    """class_level evidence_basis must NOT render a bare strength/direction (the Parkinson bug);
    the line says 'class-level signal (no direct evidence for this drug)' instead."""
    lit = LiteratureOutput(
        evidence_summary=EvidenceSummary(
            summary="GLP-1 class RCTs in Parkinson's; no direct semaglutide evidence.",
            study_count=4,
            strength="none",
            direction="none",
            evidence_basis="class_level",
            supporting_pmids=["38598572"],
        )
    )
    out = _fmt_literature(lit)
    assert (
        "**Evidence strength:** class-level signal (no direct evidence for this drug)"
        in out
    )
    # the prose still renders, but the line must not read as direct drug strength
    assert "**Evidence strength:** strong" not in out
    assert "**Evidence strength:** none" not in out


def test_fmt_clinical_trials_empty_returns_placeholder():
    out = ClinicalTrialsOutput()
    rendered = _fmt_clinical_trials(out)
    assert rendered == "_No clinical trials data available._"


def test_fmt_clinical_trials_summary_only():
    out = ClinicalTrialsOutput(summary="Drug already approved for this indication.")
    rendered = _fmt_clinical_trials(out)
    assert rendered == "Drug already approved for this indication."


def test_fmt_clinical_trials_approval_is_approved():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=True,
            label_found=True,
            matched_indication="Type 2 Diabetes Mellitus",
            drug_names_checked=["semaglutide", "ozempic"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**FDA approval:** Approved (Type 2 Diabetes Mellitus)" in rendered


def test_fmt_clinical_trials_approval_label_found_not_approved():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=False,
            label_found=True,
            matched_indication=None,
            drug_names_checked=["semaglutide"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**FDA approval:** Not found on FDA label for this indication" in rendered


def test_fmt_clinical_trials_approval_no_label():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=False,
            label_found=False,
            matched_indication=None,
            drug_names_checked=["aducanumab", "aduhelm"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "No FDA label found for aducanumab, aduhelm" in rendered
    assert "status undetermined" in rendered


def test_fmt_clinical_trials_search_renders_total():
    out = ClinicalTrialsOutput(
        search=SearchTrialsResult(
            total_count=12,
            by_status={"RECRUITING": 5, "ACTIVE_NOT_RECRUITING": 4, "WITHDRAWN": 3},
            trials=[],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Trial activity:** 12 total trial(s) for this pair" in rendered
    assert "recruiting" not in rendered.lower()
    assert "withdrawn" not in rendered.lower()


def test_fmt_clinical_trials_search_whitespace():
    out = ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=0, by_status={}, trials=[])
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Trial activity:** 0 total trial(s) for this pair" in rendered
    assert "Whitespace: no trials found for this drug × indication pair." in rendered


def test_fmt_clinical_trials_completed_renders_count_and_top_trials():
    trial = Trial(
        nct_id="NCT04567890",
        title="Semaglutide in NASH",
        phase="Phase 3",
        overall_status="Completed",
    )
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(
            total_count=7,
            trials=[trial],
        )
    )
    rendered = _fmt_clinical_trials(out)
    # 7 on record but only 1 fetched, none hidden → reconciling slice clause, no false subtract.
    assert "**Completed trials (7 total on record):**" in rendered
    assert "showing 1 of the first 1 fetched" in rendered
    assert (
        "[NCT04567890](https://clinicaltrials.gov/study/NCT04567890) — Semaglutide in NASH (Phase 3, Completed)"
        in rendered
    )


def test_fmt_clinical_trials_completed_caps_at_ten():
    trials = [
        Trial(
            nct_id=f"NCT{i:08d}",
            title=f"Trial {i}",
            phase="Phase 2",
            overall_status="Completed",
        )
        for i in range(15)
    ]
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=15, trials=trials)
    )
    rendered = _fmt_clinical_trials(out)
    assert "NCT00000009" in rendered
    assert "NCT00000010" not in rendered


def test_fmt_clinical_trials_completed_skips_contaminated_examples():
    # The CT agent flagged the two PAH trials as contamination (different indication).
    # The rendered examples must skip them, but the total_count header stays verbatim.
    pah_a = Trial(
        nct_id="NCT00303459",
        title="Bosentan + Sildenafil in PAH",
        phase="Phase 4",
        overall_status="Completed",
    )
    pah_b = Trial(
        nct_id="NCT00644605",
        title="Sildenafil in PAH",
        phase="Phase 3",
        overall_status="Completed",
    )
    systemic = Trial(
        nct_id="NCT00150358",
        title="Sildenafil in Hypertensive Men",
        phase="Phase 4",
        overall_status="Completed",
    )
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(
            total_count=64,
            trials=[pah_a, pah_b, systemic],
        ),
        contaminated_nct_ids=["NCT00303459", "NCT00644605"],
    )
    rendered = _fmt_clinical_trials(out)
    # Header states the authoritative total on record; the clause reconciles the FETCHED slice
    # (3 fetched, 2 hidden → 1 shown) WITHOUT implying 64 - 2 = 62 visible.
    assert "**Completed trials (64 total on record):**" in rendered
    assert (
        "showing 1 relevant of the first 3 fetched "
        "(2 of those fetched hidden as a different indication)." in rendered
    )
    # Contaminated PAH trials are still named in the "excluded" line at the top, but NOT
    # rendered as completed-trial examples. Scope the check to the trial-table region.
    table = rendered.split("**Completed trials")[1]
    assert "NCT00303459" not in table
    assert "NCT00644605" not in table
    # The genuinely relevant systemic-hypertension trial surfaces.
    assert (
        "[NCT00150358](https://clinicaltrials.gov/study/NCT00150358) — "
        "Sildenafil in Hypertensive Men (Phase 4, Completed)" in table
    )


def test_fmt_clinical_trials_terminated_skips_contaminated_examples():
    pah = Trial(
        nct_id="NCT02060487",
        title="Oral Sildenafil on Mortality in PAH",
        phase="Phase 4",
        why_stopped="Primary objective met at interim analysis",
    )
    systemic = Trial(
        nct_id="NCT01392638",
        title="Sildenafil in Resistant Hypertension",
        phase="Phase 2",
        why_stopped="Slow enrollment",
    )
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=18, trials=[pah, systemic]),
        contaminated_nct_ids=["NCT02060487"],
    )
    rendered = _fmt_clinical_trials(out)
    # 18 on record; 2 fetched, 1 hidden → 1 shown, reconciled against the fetched slice.
    assert "**Terminated trials (18 total on record):**" in rendered
    assert (
        "showing 1 relevant of the first 2 fetched "
        "(1 of those fetched hidden as a different indication)." in rendered
    )
    table = rendered.split("**Terminated trials")[1]
    assert "NCT02060487" not in table
    assert "NCT01392638" in table


def test_fmt_clinical_trials_broader_distinct_suppresses_trial_tables():
    # Hypertension demoted broader_distinct (PAH is the approved subtype). The artifact is
    # dominated by PAH trials that can't be cleanly filtered, so example tables are suppressed
    # while the verbatim total_count is still reported.
    pah = Trial(
        nct_id="NCT00323297",
        title="Sildenafil + Bosentan in PAH",
        phase="Phase 4",
        overall_status="Completed",
    )
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=64, trials=[pah]),
        terminated=TerminatedTrialsResult(
            total_count=18,
            trials=[
                Trial(nct_id="NCT00586794", title="PAH in Eisenmenger", phase="Phase 3")
            ],
        ),
    )
    rendered = _fmt_clinical_trials(
        out, "hypertension", approval_relationship="broader_distinct"
    )
    # Verbatim totals are reported.
    assert "**Completed trials (64 total):**" in rendered
    assert "**Terminated trials (18):**" in rendered
    # But no example trials are listed.
    assert "NCT00323297" not in rendered
    assert "NCT00586794" not in rendered
    assert "contaminated by approved subtype" in rendered


def test_fmt_clinical_trials_related_family_still_lists_trials():
    # A non-contaminated relationship renders the example tables normally.
    trial = Trial(
        nct_id="NCT04567890",
        title="Drug in Crohn's",
        phase="Phase 3",
        overall_status="Completed",
    )
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=3, trials=[trial])
    )
    rendered = _fmt_clinical_trials(
        out, "crohn's disease", approval_relationship="related_family"
    )
    assert "NCT04567890" in rendered


def test_fmt_clinical_trials_terminated_with_why_stopped():
    trial = Trial(
        nct_id="NCT01112233",
        title="Cardio Trial",
        phase="Phase 2",
        why_stopped="Sponsor decision due to slow enrollment",
    )
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=1, trials=[trial])
    )
    rendered = _fmt_clinical_trials(out)
    # 1 on record, 1 fetched, none hidden → all fetched, no reconciling clause needed.
    assert "**Terminated trials (1 total on record):**" in rendered
    assert (
        "[NCT01112233](https://clinicaltrials.gov/study/NCT01112233) Cardio Trial (Phase 2)"
        " [enrollment] — *Sponsor decision due to slow enrollment*" in rendered
    )


def test_fmt_clinical_trials_terminated_unknown_when_no_why_stopped():
    trial = Trial(nct_id="NCT09998888", title="Mystery Stop", phase="Phase 1")
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=1, trials=[trial])
    )
    rendered = _fmt_clinical_trials(out)
    assert (
        "[NCT09998888](https://clinicaltrials.gov/study/NCT09998888) Mystery Stop (Phase 1) [unknown]"
        in rendered
    )


def test_fmt_clinical_trials_terminated_zero_count_renders_nothing_for_section():
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=0, trials=[])
    )
    rendered = _fmt_clinical_trials(out)
    assert "Terminated trials" not in rendered


@pytest.mark.skip(
    reason="Competitive landscape rendering in the report is deferred; the data is still "
    "surfaced in the UI (Clinical Trials tab). See PLAN_react.md."
)
def test_fmt_clinical_trials_landscape_renders_competitors():
    out = ClinicalTrialsOutput(
        landscape=IndicationLandscape(
            total_trial_count=42,
            competitors=[
                CompetitorEntry(
                    sponsor="Novo Nordisk",
                    drug_name="Semaglutide",
                    max_phase="Phase 3",
                    trial_count=8,
                ),
                CompetitorEntry(
                    sponsor="Eli Lilly",
                    drug_name="Tirzepatide",
                    max_phase="Phase 3",
                    trial_count=5,
                ),
            ],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Competitive landscape (2 competitors):**" in rendered
    assert "Semaglutide (Novo Nordisk) — Phase 3, 8 trial(s)" in rendered
    assert "Tirzepatide (Eli Lilly) — Phase 3, 5 trial(s)" in rendered


def test_format_report_full_assembly():
    """End-to-end: confirms all sections render, ordering is correct, and the
    Summary's ranked entries pull blurbs from disease_findings."""
    output = SupervisorOutput(
        drug_name="semaglutide",
        candidate_diseases=["NASH", "Alzheimer's disease"],
        summary=(
            "1. NASH — literature: moderate, 4 PMIDs; trials: 5 total, 2 completed, "
            "0 terminated\n"
        ),
        top_diseases=["NASH"],
        disease_findings=[
            CandidateFindings(
                disease="NASH",
                source="both",
                literature=LiteratureOutput(
                    evidence_summary=EvidenceSummary(
                        summary="Multiple Phase 2 trials show histological improvement.",
                        study_count=4,
                        strength="moderate",
                        key_findings=["MASH resolution in ~60% of patients"],
                        supporting_pmids=["12345678"],
                    )
                ),
                clinical_trials=ClinicalTrialsOutput(
                    summary="Active development pipeline.",
                    search=SearchTrialsResult(
                        total_count=5,
                        by_status={"RECRUITING": 2, "ACTIVE_NOT_RECRUITING": 3},
                        trials=[],
                    ),
                ),
                blurb=CandidateBlurb(
                    stage="Phase 3",
                    literature="Moderate, 4 trials",
                    verdict="Live and progressing",
                    prose="NASH is the lead repurposing target. Phase 3 readout near.",
                ),
            ),
        ],
    )

    rendered = format_report(output)

    assert "# IndicationScout Report: Semaglutide" in rendered
    assert "## Summary" in rendered
    assert "1. NASH" in rendered
    assert "**Stage**" in rendered
    assert "Phase 3" in rendered
    assert "NASH is the lead repurposing target." in rendered
    assert "## Diseases Considered" in rendered
    assert "- NASH" in rendered
    assert "- Alzheimer's Disease" in rendered
    assert "## Findings by Disease" in rendered
    assert "## NASH _(source: both)_" in rendered
    assert "### Literature" in rendered
    assert "**Evidence strength:** moderate" in rendered
    assert "[12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)" in rendered
    assert "### Clinical Trials" in rendered
    assert "**Trial activity:** 5 total trial(s) for this pair" in rendered


def test_format_report_summary_ranks_top_diseases_in_order():
    """Summary numbers entries 1.., 2.., ... in top_diseases order."""
    output = SupervisorOutput(
        drug_name="testdrug",
        candidate_diseases=["Disease A", "Disease B", "Disease C"],
        top_diseases=["Disease B", "Disease A"],
        disease_findings=[
            CandidateFindings(
                disease="Disease B",
                source="competitor",
                blurb=CandidateBlurb(verdict="Promising", prose="B prose."),
            ),
            CandidateFindings(
                disease="Disease A",
                source="mechanism",
                blurb=CandidateBlurb(verdict="Speculative", prose="A prose."),
            ),
        ],
    )

    rendered = format_report(output)

    assert "1. Disease B" in rendered
    assert "2. Disease A" in rendered
    assert rendered.index("1. Disease B") < rendered.index("2. Disease A")
    assert "B prose." in rendered
    assert "A prose." in rendered


def test_format_report_findings_extras_render_after_top():
    """A finding that isn't in top_diseases still appears in Candidate Findings."""
    output = SupervisorOutput(
        drug_name="testdrug",
        candidate_diseases=["Top Disease", "Extra Disease"],
        top_diseases=["Top Disease"],
        disease_findings=[
            CandidateFindings(
                disease="Top Disease",
                source="competitor",
                blurb=CandidateBlurb(prose="Top prose."),
            ),
            CandidateFindings(disease="Extra Disease", source="mechanism"),
        ],
    )

    rendered = format_report(output)

    # Top disease is the only one ranked in the Summary.
    assert "1. Top Disease" in rendered
    assert "1. Extra Disease" not in rendered
    assert "2. Extra Disease" not in rendered
    # Both findings appear under Candidate Findings.
    assert "## Top Disease _(source: competitor)_" in rendered
    assert "## Extra Disease _(source: mechanism)_" in rendered


def test_format_report_no_candidates_or_findings():
    output = SupervisorOutput(drug_name="metformin")
    rendered = format_report(output)
    assert "# IndicationScout Report: Metformin" in rendered
    assert "_No summary produced._" in rendered
    assert "_No candidates surfaced._" in rendered
    assert "_No candidate findings produced._" in rendered


def test_format_report_unknown_drug_name_default():
    output = SupervisorOutput()
    rendered = format_report(output)
    assert "# IndicationScout Report: Unknown Drug" in rendered


def test_fmt_clinical_trials_truncates_with_disclosure():
    """When the relevant completed list exceeds the render cap (10), the body lists the first
    10 and discloses the remainder so the count can't read as the full list (gefitinib ×
    breast cancer shape: 18 relevant completed, only 10 shown)."""
    trials = [
        Trial(
            nct_id=f"NCT{i:08d}",
            title=f"Study {i}",
            phase="Phase 2",
            overall_status="COMPLETED",
        )
        for i in range(18)
    ]
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=18, trials=trials),
    )
    rendered = _fmt_clinical_trials(out)
    # First 10 listed, 11th not.
    assert "NCT00000009" in rendered
    assert "NCT00000010" not in rendered
    # Disclosure line present with correct remainder and totals.
    assert "and 8 more relevant completed trial(s) not listed" in rendered
    assert "showing first 10 of 18" in rendered


def test_fmt_clinical_trials_no_truncation_note_at_or_below_cap():
    """Exactly 10 relevant trials → all shown, no truncation disclosure."""
    trials = [
        Trial(
            nct_id=f"NCT{i:08d}",
            title=f"Study {i}",
            phase="Phase 2",
            overall_status="COMPLETED",
        )
        for i in range(10)
    ]
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=10, trials=trials),
    )
    rendered = _fmt_clinical_trials(out)
    assert "more relevant completed trial(s) not listed" not in rendered


def test_fmt_clinical_trials_renders_authoritative_dev_stage_line():
    """The per-disease CT section renders the authoritative development-stage phrase from
    ct.signals.dev_stage — the single source of the phase-tier judgment (the CT prose must not
    judge the tier, so this is the only place it is stated)."""
    from indication_scout.agents.clinical_trials.clinical_trials_output import (
        TrialSignals,
    )

    out = ClinicalTrialsOutput(
        summary="NCT05205928 (Phase 2/Phase 3, COMPLETED) studied semaglutide in T1DM.",
        signals=TrialSignals(dev_stage="completed_phase3"),
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Development stage:** Phase 3 completed for this indication" in rendered
    # the prose still renders below it
    assert "NCT05205928" in rendered


def test_fmt_clinical_trials_no_dev_stage_line_when_no_signals():
    """No signals → no development-stage line (e.g. sub-agent didn't classify relevance)."""
    out = ClinicalTrialsOutput(summary="some prose")
    rendered = _fmt_clinical_trials(out)
    assert "**Development stage:**" not in rendered
