"""Integration tests for the supervisor agent.

Hits real Anthropic, Open Targets, PubMed, ClinicalTrials.gov, and ChEMBL APIs.
Uses the test database (scout_test) via db_session_truncating.

Expected values verified by a live run on 2026-04-08 with drug=metformin.
"""

import logging
from datetime import date

import pytest

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_agent import (
    run_supervisor_agent,
)
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.services.analysis_runner import build_agent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Metformin, no date cutoff
#
# Expected values verified by a live run on 2026-04-08.
# ------------------------------------------------------------------

# Stable subset of candidate diseases surfaced by find_candidates via Open Targets.
# Re-baselined 2026-06-06 after integration tests began loading .env.constants.integration
# (larger SUPERVISOR_CANDIDATE_CAP / OPEN_TARGETS prefetch). At integration-scale caps the
# wider competitor pool feeds the hierarchical dedup pass, which collapses specific subtypes
# (insulin resistance, metabolic syndrome, prostate cancer) into broader parents
# (metabolic disease, diabetes mellitus, cardiovascular disease). PCOS and gestational
# diabetes are the top-ranked RCT-level signals and survive dedup run-to-run.
# (full list from 2026-06-06 run: cardiovascular disease, hepatic steatosis,
#  polycystic ovary syndrome, metabolic disease, gestational diabetes, diabetes mellitus)
_EXPECTED_CANDIDATES = {
    "polycystic ovary syndrome",
    "gestational diabetes",
}

# Mechanism target symbols that must appear (mirrors test_mechanism_agent.py)
_EXPECTED_TARGET_SYMBOLS = {"NDUFS2", "NDUFS1", "NDUFV1", "MT-ND1", "GPD2"}


@pytest.fixture
def supervisor_agent(db_session_truncating, test_cache_dir):
    return build_agent(db_session_truncating, cache_dir=test_cache_dir)


async def test_metformin_supervisor_agent(supervisor_agent):
    """End-to-end: supervisor agent produces correct SupervisorOutput for metformin.

    Verifies:
    - output is a SupervisorOutput with correct drug_name
    - candidates list is non-empty (find_candidates was called and parsed)
    - mechanism is present with known targets (analyze_mechanism was called)
    - findings list is non-empty with valid disease names
    - each finding has at least one sub-agent result (literature or clinical_trials)
    - summary is non-empty and matches the structured fact-list shape (finalize_supervisor was
      called)
    """
    agent, get_merged_allowlist, get_auto_findings, get_approval_labels = supervisor_agent
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "metformin",
        get_auto_findings=get_auto_findings,
        get_approval_labels=get_approval_labels,
    )

    assert isinstance(output, SupervisorOutput)

    # --- drug_name ---
    assert output.drug_name == "metformin"

    # --- candidate_diseases: find_candidates was called and results parsed ---
    assert len(output.candidate_diseases) >= 3
    assert all(isinstance(c, str) and len(c) > 0 for c in output.candidate_diseases)
    assert _EXPECTED_CANDIDATES.issubset(set(output.candidate_diseases))

    # --- mechanism: analyze_mechanism was called ---
    assert isinstance(output.mechanism, MechanismOutput)
    assert len(output.mechanism.drug_targets) >= 10
    assert _EXPECTED_TARGET_SYMBOLS.issubset(set(output.mechanism.drug_targets.keys()))

    # --- disease_findings: at least one candidate was investigated ---
    assert len(output.disease_findings) >= 1
    assert all(isinstance(f, CandidateFindings) for f in output.disease_findings)

    # Every finding must name a disease and have at least one sub-agent result
    for finding in output.disease_findings:
        assert isinstance(finding.disease, str) and len(finding.disease) > 0
        assert (
            finding.literature is not None or finding.clinical_trials is not None
        ), f"Finding for {finding.disease!r} has no sub-agent results"

    # All finding disease names must come from the candidate_diseases list
    # finding_diseases = {f.disease for f in output.disease_findings}
    # assert finding_diseases.issubset(
    #     set(output.candidate_diseases)
    # ), f"Findings reference diseases not in candidate_diseases: {finding_diseases - set(output.candidate_diseases)}"

    # --- top_diseases: invariants ---
    # Strict subset of disease_findings (enforced in run_supervisor_agent).
    finding_names = {f.disease for f in output.disease_findings}
    assert set(output.top_diseases).issubset(
        finding_names
    ), f"top_diseases not a subset of disease_findings: {set(output.top_diseases) - finding_names}"
    # Hard cap at 5.
    assert len(output.top_diseases) <= 5
    # disease_findings ordering: top_diseases entries appear first in rank order.
    findings_order = [f.disease for f in output.disease_findings]
    assert (
        findings_order[: len(output.top_diseases)] == output.top_diseases
    ), "disease_findings must lead with top_diseases in rank order"

    # --- summary: finalize_supervisor was called ---
    # Per supervisor.txt "WRITING THE SUMMARY", the summary is a structured ranked
    # fact list headed "Ranked repurposing signals for <drug>:", one line per
    # investigated pair with literature-maturity and development-status phrases
    # (not field labels), not a narrative.
    assert len(output.summary) > 100
    summary_lower = output.summary.lower()
    assert "ranked repurposing signals for metformin:" in summary_lower
    # Must not look like raw JSON or a tool schema
    assert not output.summary.strip().startswith("{")


# ------------------------------------------------------------------
# Imatinib, holdout cutoff 2006-01-01
#
# Holdout run: candidate surfacing must include imatinib's known/emerging
# 2006-era indications. Exact disease strings (lowercased canonical forms as
# emitted by the pipeline).
# ------------------------------------------------------------------

_EXPECTED_IMATINIB_DISEASES = {
    "chronic myelogenous leukemia",
    "acute lymphoblastic leukemia",
    "gastrointestinal stromal tumor",
    "hypereosinophilic syndrome",
    "dermatofibrosarcoma protuberans",
    "systemic mastocytosis",
}


async def test_imatinib_holdout_2006(db_session_truncating, test_cache_dir):
    """End-to-end holdout: supervisor agent run for imatinib with date_before
    2006-01-01 must surface imatinib's known/emerging 2006-era indications in
    candidate_diseases (exact names)."""
    agent, get_merged_allowlist, get_auto_findings, get_approval_labels = build_agent(
        db_session_truncating,
        date_before=date(2006, 1, 1),
        cache_dir=test_cache_dir,
    )
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "imatinib",
        get_auto_findings=get_auto_findings,
        get_approval_labels=get_approval_labels,
    )

    assert isinstance(output, SupervisorOutput)
    assert output.drug_name == "imatinib"
    assert _EXPECTED_IMATINIB_DISEASES.issubset(set(output.candidate_diseases))


# ------------------------------------------------------------------
# Approval-relationship labeling + search-pool relevance gate
#
# Two behavior-defining regressions (verified by live runs on 2026-06-20):
#   - semaglutide: a SIBLING of an approved indication (T1DM vs approved T2DM) must NOT be
#     dropped/demoted (label "none", kept); a broader parent of an approved subtype (NAFLD over
#     approved MASH) is kept but label "contaminated".
#   - sildenafil: systemic "hypertension" (approved indication is PAH) is kept with label
#     "contaminated", and the PAH trials that a hypertension query pulls in are tagged
#     contaminated — including SEARCH-pool active trials, which previously bypassed the gate and
#     leaked into the "Phase 3 active" dev-stage signal.
# Rankings are LLM-driven and may shift run-to-run, so we assert the typed labels and the
# contamination tagging (the deterministic behavior), not exact rank positions.
# ------------------------------------------------------------------


def _find_finding(output: SupervisorOutput, substr: str) -> CandidateFindings | None:
    """Return the first finding whose disease name contains `substr` (lowercased), or None."""
    for f in output.disease_findings:
        if substr in f.disease.lower():
            return f
    return None


async def test_semaglutide_sibling_kept_and_contaminated_labels(supervisor_agent):
    """Semaglutide: T1DM sibling kept (label 'none'); NAFLD broader-parent kept ('contaminated').

    Regression for the approval-relationship upstream labeling. Before the fix, T1DM (a sibling of
    approved T2DM) was demoted into a footer and the report surfaced a weaker candidate; NAFLD was
    mislabeled. Now both are KEPT with label-grounded relationships. Verified 2026-06-20.
    """
    agent, get_merged_allowlist, get_auto_findings, get_approval_labels = supervisor_agent
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "semaglutide",
        get_auto_findings=get_auto_findings,
        get_approval_labels=get_approval_labels,
    )

    assert output.drug_name == "semaglutide"

    # T1DM — sibling of approved T2DM — kept and labeled "none" (NOT dropped, NOT demoted).
    t1dm = _find_finding(output, "type 1 diabetes")
    assert t1dm is not None, "Type 1 Diabetes must be kept as a candidate, not dropped"
    assert t1dm.approval_relationship == "none"

    # NAFLD — broader parent of approved MASH — kept and labeled "contaminated".
    nafld = _find_finding(output, "non-alcoholic fatty liver")
    assert nafld is not None, "NAFLD must be kept as a candidate, not dropped"
    assert nafld.approval_relationship == "contaminated"

    # Approved indications (T2DM, obesity) must NOT appear as kept findings — dropped upstream.
    assert _find_finding(output, "type 2 diabetes") is None
    assert _find_finding(output, "obesity") is None


async def test_sildenafil_hypertension_contaminated_and_pah_trials_tagged(supervisor_agent):
    """Sildenafil: systemic hypertension kept ('contaminated'); PAH search trials tagged contaminated.

    Two regressions in one pair:
      1. Approval labeling: hypertension (approved = PAH) is KEPT with label 'contaminated', not
         demoted.
      2. Search-pool relevance gate: PAH trials a hypertension query pulls into the SEARCH scope
         (where active/recruiting trials live) are now classified and tagged contaminated, so they
         cannot inflate the "Phase 3 active" dev-stage signal. NCT07462260 / NCT06317805 are PAH
         active trials that previously leaked through as relevant. Verified 2026-06-20.
    """
    agent, get_merged_allowlist, get_auto_findings, get_approval_labels = supervisor_agent
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "sildenafil",
        get_auto_findings=get_auto_findings,
        get_approval_labels=get_approval_labels,
    )

    assert output.drug_name == "sildenafil"

    hyp = _find_finding(output, "hypertension")
    assert hyp is not None, "systemic hypertension must be kept as a candidate, not dropped"
    # Distinguish systemic hypertension from the approved 'pulmonary (arterial) hypertension'.
    assert "pulmonary" not in hyp.disease.lower()
    assert hyp.approval_relationship == "contaminated"

    # The approved indication (pulmonary/PAH) must NOT be a kept ranked finding.
    assert "pulmonary" not in " ".join(output.top_diseases).lower()

    # PAH trials pulled into the hypertension SEARCH scope must be tagged contaminated.
    assert hyp.clinical_trials is not None
    contaminated = set(hyp.clinical_trials.contaminated_nct_ids)
    assert {"NCT07462260", "NCT06317805"}.issubset(contaminated), (
        "PAH search-pool trials must be tagged contaminated (search relevance gate). "
        f"Got contaminated_nct_ids={sorted(contaminated)}"
    )
