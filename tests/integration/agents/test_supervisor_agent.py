"""Integration tests for the supervisor agent.

Hits real Anthropic, Open Targets, PubMed, ClinicalTrials.gov, and ChEMBL APIs.
Uses the test database (scout_test) via db_session_truncating.

Expected values verified by a live run on 2026-04-08 with drug=metformin.
"""

import logging

import pytest
from langchain_anthropic import ChatAnthropic

from indication_scout.agents.supervisor.supervisor_agent import (
    build_supervisor_agent,
    run_supervisor_agent,
)
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Metformin, no date cutoff
#
# Expected values verified by a live run on 2026-04-08.
# ------------------------------------------------------------------

# Stable subset of candidate diseases surfaced by find_candidates via Open Targets
# (full list from live run: cardiovascular disease, polycystic ovary syndrome,
#  metabolic disease, gestational diabetes, lipid metabolism disorder,
#  metabolic syndrome, type 1 diabetes, insulin resistance, prostate cancer,
#  muscular dystrophy, brain cancer, bipolar disorder, migraine,
#  colorectal cancer, psoriasis)
_EXPECTED_CANDIDATES = {
    "polycystic ovary syndrome",
    "gestational diabetes",
    "metabolic syndrome",
    "insulin resistance",
    "prostate cancer",
}

# Mechanism target symbols that must appear (mirrors test_mechanism_agent.py)
_EXPECTED_TARGET_SYMBOLS = {"NDUFS2", "NDUFS1", "NDUFV1", "MT-ND1", "GPD2"}


@pytest.fixture
def supervisor_agent(db_session_truncating, test_cache_dir):
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    svc = RetrievalService(test_cache_dir)
    return build_supervisor_agent(llm, svc=svc, db=db_session_truncating)


async def test_metformin_supervisor_agent(supervisor_agent):
    """End-to-end: supervisor agent produces correct SupervisorOutput for metformin.

    Verifies:
    - output is a SupervisorOutput with correct drug_name
    - candidates list is non-empty (find_candidates was called and parsed)
    - mechanism is present with known targets (analyze_mechanism was called)
    - findings list is non-empty with valid disease names
    - each finding has at least one sub-agent result (literature or clinical_trials)
    - summary is non-empty and mentions metformin (finalize_supervisor was called)
    """
    output = await run_supervisor_agent(supervisor_agent, "metformin")

    assert isinstance(output, SupervisorOutput)

    # --- drug_name ---
    assert output.drug_name == "metformin"

    # --- candidates: find_candidates was called and results parsed ---
    assert len(output.candidates) >= 3
    assert all(isinstance(c, str) and len(c) > 0 for c in output.candidates)
    assert _EXPECTED_CANDIDATES.issubset(set(output.candidates))

    # --- mechanism: analyze_mechanism was called ---
    assert isinstance(output.mechanism, MechanismOutput)
    assert len(output.mechanism.drug_targets) >= 10
    assert _EXPECTED_TARGET_SYMBOLS.issubset(set(output.mechanism.drug_targets.keys()))

    # --- findings: at least one candidate was investigated ---
    assert len(output.findings) >= 1
    assert all(isinstance(f, CandidateFindings) for f in output.findings)

    # Every finding must name a disease and have at least one sub-agent result
    for finding in output.findings:
        assert isinstance(finding.disease, str) and len(finding.disease) > 0
        assert (
            finding.literature is not None or finding.clinical_trials is not None
        ), f"Finding for {finding.disease!r} has no sub-agent results"

    # All finding disease names must come from the candidates list
    # finding_diseases = {f.disease for f in output.findings}
    # assert finding_diseases.issubset(
    #     set(output.candidates)
    # ), f"Findings reference diseases not in candidates: {finding_diseases - set(output.candidates)}"

    # --- summary: finalize_supervisor was called ---
    assert len(output.summary) > 100
    assert "metformin" in output.summary.lower()
    # Must not look like raw JSON or a tool schema
    assert not output.summary.strip().startswith("{")
