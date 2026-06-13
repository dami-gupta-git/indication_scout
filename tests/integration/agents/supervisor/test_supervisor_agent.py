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
    agent, get_merged_allowlist, get_auto_findings = supervisor_agent
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "metformin",
        get_auto_findings=get_auto_findings,
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
    agent, get_merged_allowlist, get_auto_findings = build_agent(
        db_session_truncating,
        date_before=date(2006, 1, 1),
        cache_dir=test_cache_dir,
    )
    output = await run_supervisor_agent(
        agent,
        get_merged_allowlist,
        "imatinib",
        get_auto_findings=get_auto_findings,
    )

    assert isinstance(output, SupervisorOutput)
    assert output.drug_name == "imatinib"
    assert _EXPECTED_IMATINIB_DISEASES.issubset(set(output.candidate_diseases))
