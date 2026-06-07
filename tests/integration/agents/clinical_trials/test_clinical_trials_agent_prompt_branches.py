"""End-to-end regression tests for the three clinical_trials agent prompt branches.

Each test exercises a different INFERENCE branch and asserts both on the
structural artifacts (deterministic — produced by tools) and on the summary
content (LLM-generated — asserted with robust substring/regex checks that
reflect prompt rules rather than specific phrasings).

Hits real ClinicalTrials.gov, openFDA, ChEMBL, NCBI, and Anthropic APIs.
"""

import logging
import re
from datetime import date

import pytest
from langchain_anthropic import ChatAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)

# Phrases the prompt's banned-phrasings rule forbids in summaries.
# If any appear, a REPORTING rule is being ignored.
_BANNED_HEDGE_PHRASES = [
    "moderate evidence",
    "inconsistent with a positive",
    "favorable signal",
    "sustained clinical interest",
]


@pytest.fixture
def clinical_trials_agent():
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    return build_clinical_trials_agent(llm, date_before=_CUTOFF)


@pytest.fixture
def clinical_trials_agent_live():
    """Agent with no holdout cutoff (date_before=None).

    The no-label short-circuit branch (label_found=False) is only reachable on
    the live openFDA path — the holdout path hardcodes label_found=True and a
    single drug name. Tests that exercise that branch need this fixture.
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    return build_clinical_trials_agent(llm, date_before=None)


async def test_approved_pair_semaglutide_t2dm(clinical_trials_agent):
    """Approved pair → ApprovalCheck.is_approved=True, summary states approval.

    The prompt no longer short-circuits on an approved pair: it mandates a full
    analysis (call all tools, write the full summary) and lets the supervisor
    reconcile subtype/superset relationships downstream. This test verifies the
    deterministic ApprovalCheck artifact plus that the summary states the
    approval, without asserting the retired single-sentence behavior.

    Uses semaglutide × type 2 diabetes mellitus (approved 2017-12-05) so the
    pair is approved as of the holdout cutoff (_CUTOFF = 2025-01-01). With
    date_before set, check_fda_approval takes the hardcoded-table holdout path,
    which records drug_names_checked as the single queried name.
    """
    output = await run_clinical_trials_agent(
        clinical_trials_agent, "semaglutide", "type 2 diabetes mellitus"
    )

    assert isinstance(output, ClinicalTrialsOutput)

    # --- ApprovalCheck artifact (deterministic) ---
    assert output.approval is not None
    assert output.approval.is_approved is True
    assert output.approval.label_found is True
    assert output.approval.matched_indication == "type 2 diabetes mellitus"
    assert set(output.approval.drug_names_checked) == {"semaglutide"}

    # --- Summary (LLM-generated; assertions reflect prompt rules) ---
    summary_lower = output.summary.lower()
    assert "fda-approved" in summary_lower or "fda approved" in summary_lower
    assert "diabetes" in summary_lower

    # Banned hedge phrasings — prompt rule: "Do not hedge."
    for phrase in _BANNED_HEDGE_PHRASES:
        assert phrase not in summary_lower, (
            f"Banned hedge phrase {phrase!r} appeared in summary: "
            f"{output.summary!r}"
        )


async def test_no_label_atabecestat_alzheimer(clinical_trials_agent_live):
    """No FDA label found → ApprovalCheck.label_found=False, summary notes it.

    The prompt no longer short-circuits to a single sentence when no label is
    found: it mandates a full analysis. This test verifies the deterministic
    ApprovalCheck artifact (label_found=False, the signal that the no-label
    path ran across all aliases) plus that the summary states the label was
    not found.

    Uses the live fixture (date_before=None): label_found=False is only
    reachable on the live openFDA path, since the holdout path hardcodes
    label_found=True. Atabecestat is an investigational BACE1 inhibitor that
    never reached approval, so openFDA returns no label across all aliases.
    """
    output = await run_clinical_trials_agent(
        clinical_trials_agent_live, "atabecestat", "Alzheimer Disease"
    )

    assert isinstance(output, ClinicalTrialsOutput)

    # --- ApprovalCheck artifact (deterministic) ---
    assert output.approval is not None
    assert output.approval.is_approved is False
    assert output.approval.label_found is False
    assert output.approval.matched_indication is None
    assert set(output.approval.drug_names_checked) == {
        "atabecestat",
        "jnj-54861911",
        "jnj-54861911-aaa",
        "rsc- 385896",
        "rsc-385896",
    }

    # --- Summary (LLM-generated; assertions reflect prompt rules) ---
    summary_lower = output.summary.lower()

    # The summary should reference the FDA label (its absence for this drug).
    assert "label" in summary_lower


async def test_confirmed_failure_count_scaled_atorvastatin_alzheimer(
    clinical_trials_agent,
):
    """≥2 completed Phase 3s + label exists but indication not approved →
    strong-evidence phrasing, no hedging.

    Verifies the count-scaled INFERENCE branch for pair_completed Phase 3.
    Atorvastatin has an active FDA label (for hyperlipidemia) but is not
    approved for Alzheimer's disease, and two Phase 3 AD trials are on
    record as completed well before the 2025-01-01 cutoff:
      - NCT00151502 (Phase 3, enrollment 600)
      - NCT02913664 (Phase 2/Phase 3, enrollment 513, completed 2021-11-30)

    This is the only branch that requires label_found=True + is_approved=False,
    and only drugs with an active SPL and a failed side-indication Phase 3
    history can exercise it. Many historically interesting candidates
    (rosiglitazone, solanezumab, gantenerumab) do not have current SPLs in
    openFDA and therefore hit the prerequisite short-circuit instead.
    """
    output = await run_clinical_trials_agent(
        clinical_trials_agent, "atorvastatin", "Alzheimer Disease"
    )

    assert isinstance(output, ClinicalTrialsOutput)

    # --- ApprovalCheck artifact (deterministic) ---
    assert output.approval is not None
    assert output.approval.is_approved is False
    assert output.approval.label_found is True
    assert output.approval.matched_indication is None

    # --- completed Phase-3-level count (derived from shown trials list) ---
    # Phase is read off each trial in the returned list. This is a floor when
    # total_count exceeds the shown 50, but for atorvastatin × AD the full
    # programme fits comfortably within the cap. Count Phase-3-level trials,
    # which includes "Phase 2/Phase 3" (e.g. NCT02913664) alongside strict
    # "Phase 3" (NCT00151502) — both are pivotal-tier readouts.
    assert output.completed is not None
    phase3 = sum(1 for t in output.completed.trials if t.phase and "Phase 3" in t.phase)
    assert phase3 >= 2, (
        f"Expected at least 2 completed Phase-3-level trials for atorvastatin × "
        f"AD to exercise the ≥2 branch; got {phase3}."
    )

    # --- Summary (LLM-generated) ---
    summary_lower = output.summary.lower()
    assert "atorvastatin" in summary_lower
    assert "alzheimer" in summary_lower

    # Prompt rule: "State explicitly: the pivotal trials did not lead to
    # approval." Accept close paraphrases, including an intervening qualifier
    # ("regulatory"/"FDA") and verbs like advance ("did not advance to
    # approval", "did not lead to regulatory approval").
    did_not_lead_pattern = re.compile(
        r"(did not|have not|has not|no[t]?)\s+"
        r"(lead|led|result(ed)?|advance(d)?|progress(ed)?)\s+"
        r"(to|in)\s+(\w+\s+){0,2}approval",
        re.IGNORECASE,
    )
    assert did_not_lead_pattern.search(output.summary), (
        f"Expected 'did not lead to approval' phrasing per INFERENCE rule; "
        f"summary: {output.summary!r}"
    )

    # Banned hedge phrasings — prompt rule: "Do not hedge."
    for phrase in _BANNED_HEDGE_PHRASES:
        assert phrase not in summary_lower, (
            f"Banned hedge phrase {phrase!r} appeared in summary: "
            f"{output.summary!r}"
        )
