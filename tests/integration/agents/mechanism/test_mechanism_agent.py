"""Integration tests for the mechanism agent.

Hits real Anthropic and Open Targets APIs.
"""

import logging

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput

# Targets that must appear in drug_targets (imatinib's known kinase targets)
_EXPECTED_TARGET_SYMBOLS = {"ABL1", "BCR", "KIT", "PDGFRB"}


async def test_metformin_mechanism_agent():
    """End-to-end: mechanism agent produces correct MechanismOutput for imatinib.

    Verifies:
    - known kinase targets (ABL1, BCR, KIT, PDGFRB) are in drug_targets
    - Ensembl IDs are populated for each target
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_mechanism_agent(llm)

    output = await run_mechanism_agent(agent, "imatinib")

    assert isinstance(output, MechanismOutput)

    # --- drug_targets ---
    assert len(output.drug_targets) >= 4
    assert _EXPECTED_TARGET_SYMBOLS.issubset(set(output.drug_targets.keys()))
    for symbol, ensembl_id in output.drug_targets.items():
        assert ensembl_id.startswith(
            "ENSG"
        ), f"Expected Ensembl ID for {symbol}, got {ensembl_id!r}"
