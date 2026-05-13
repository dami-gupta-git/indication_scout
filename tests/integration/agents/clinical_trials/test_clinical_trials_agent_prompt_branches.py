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

# Tokens that must NOT appear in a short-circuit summary.
# The prompt forbids reporting trials/landscape/competitors/terminations
# in either the approved or no-label short-circuit.
_SHORT_CIRCUIT_FORBIDDEN_TOKENS = [
    "trial",
    "termination",
    "landscape",
    "competitor",
]


@pytest.fixture
def clinical_trials_agent():
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    return build_clinical_trials_agent(llm, date_before=_CUTOFF)


async def test_approved_short_circuit_semaglutide_nash(clinical_trials_agent):
    """Approved pair → ApprovalCheck.is_approved=True → one-sentence summary.

    Verifies the affirmative short-circuit INFERENCE rule: when the drug is
    FDA-approved for the indication, the summary must be a single sentence
    stating that and nothing else.
    """
    output = await run_clinical_trials_agent(
        clinical_trials_agent, "semaglutide", "NASH"
    )

    assert isinstance(output, ClinicalTrialsOutput)

    # --- ApprovalCheck artifact (deterministic) ---
    assert output.approval is not None
    assert output.approval.is_approved is True
    assert output.approval.label_found is True
    assert output.approval.matched_indication == "NASH"
    assert set(output.approval.drug_names_checked) == {
        "semaglutide",
        "nn-9535",
        "nn9535",
        "nnc 0113-0217",
        "nnc-0113-0217",
        "ozempic",
        "rybelsus",
        "semaglutida",
        "wegovy",
    }

    # --- Summary (LLM-generated; assertions reflect prompt rules) ---
    summary_lower = output.summary.lower()
    assert "fda-approved" in summary_lower or "fda approved" in summary_lower
    assert "nash" in summary_lower

    # Short-circuit length bound — prompt says "single sentence"
    assert len(output.summary) < 300, (
        f"Approved short-circuit summary too long ({len(output.summary)} chars). "
        f"Prompt requires a single sentence. Summary: {output.summary!r}"
    )

    # Forbidden tokens — prompt says "Do not report trial counts, landscape,
    # terminations, or competitors."
    for token in _SHORT_CIRCUIT_FORBIDDEN_TOKENS:
        assert token not in summary_lower, (
            f"Forbidden token {token!r} appeared in approved short-circuit "
            f"summary: {output.summary!r}"
        )


async def test_no_label_short_circuit_atabecestat_alzheimer(clinical_trials_agent):
    """No FDA label found → ApprovalCheck.label_found=False → one-sentence summary.

    Verifies the prerequisite short-circuit INFERENCE rule: when no FDA label
    is found for the drug (withdrawn, never approved, or approved outside the
    US), approval status is UNKNOWN and the summary must be a single sentence
    saying so.
    """
    output = await run_clinical_trials_agent(
        clinical_trials_agent, "atabecestat", "Alzheimer Disease"
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

    # Prompt rule: "our tools did not find an FDA label for this drug, and
    # that approval status cannot be determined from available data."
    assert "label" in summary_lower
    assert (
        "cannot be determined" in summary_lower
        or "could not be determined" in summary_lower
        or "unknown" in summary_lower
    )

    # Short-circuit length bound
    assert len(output.summary) < 400, (
        f"No-label short-circuit summary too long ({len(output.summary)} chars). "
        f"Prompt requires a single sentence. Summary: {output.summary!r}"
    )

    # Forbidden tokens
    for token in _SHORT_CIRCUIT_FORBIDDEN_TOKENS:
        assert token not in summary_lower, (
            f"Forbidden token {token!r} appeared in no-label short-circuit "
            f"summary: {output.summary!r}"
        )


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

    # --- completed Phase 3 count (derived from shown trials list) ---
    # Phase 3 is read off each trial in the returned list. This is a floor
    # when total_count exceeds the shown 50, but for atorvastatin × AD the
    # full programme fits comfortably within the cap.
    assert output.completed is not None
    phase3 = sum(1 for t in output.completed.trials if t.phase == "Phase 3")
    assert phase3 >= 2, (
        f"Expected at least 2 completed Phase 3 trials for atorvastatin × AD "
        f"to exercise the ≥2 branch; got {phase3}."
    )

    # --- Summary (LLM-generated) ---
    summary_lower = output.summary.lower()
    assert "atorvastatin" in summary_lower
    assert "alzheimer" in summary_lower

    # Prompt rule: "State explicitly: the pivotal trials did not lead to
    # approval." Accept close paraphrases.
    did_not_lead_pattern = re.compile(
        r"(did not|have not|has not|no[t]?)\s+(lead|led|result(ed)?)\s+(to|in)\s+(fda )?approval",
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
