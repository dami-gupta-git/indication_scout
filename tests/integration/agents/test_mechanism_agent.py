"""Integration tests for the mechanism agent.

Hits real Anthropic and Open Targets APIs.
Expected values verified by a live run on 2026-04-07.
"""

import logging

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


# ------------------------------------------------------------------
# Metformin
#
# Expected values verified by a live run on 2026-04-07.
# ------------------------------------------------------------------

# Targets that must appear in drug_targets (stable complex I / GPD2 targets)
_EXPECTED_TARGET_SYMBOLS = {"NDUFS2", "NDUFS1", "NDUFV1", "MT-ND1", "GPD2"}

# Symbols for which associations must be fetched
_EXPECTED_ASSOCIATION_SYMBOLS = {"NDUFS2", "NDUFS1", "NDUFV1", "MT-ND1", "GPD2"}

# Top diseases expected per key target
_EXPECTED_TOP_DISEASES = {
    "GPD2": "type 2 diabetes mellitus",
    "MT-ND1": "Leber hereditary optic neuropathy",
}

# Top-level pathway terms that must appear across all targets
_EXPECTED_TOP_LEVEL_PATHWAYS = {"Metabolism"}


async def test_metformin_mechanism_agent():
    """End-to-end: mechanism agent produces correct MechanismOutput for metformin.

    Verifies:
    - known complex I and GPD2 targets are in drug_targets
    - associations are fetched per target (dict keyed by symbol)
    - top disease associations match expected values for key targets
    - pathways are fetched per target (dict keyed by symbol)
    - Metabolism appears as a top-level pathway term across targets
    - narrative summary is non-empty and mentions relevant biology
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_mechanism_agent(llm)

    output = await run_mechanism_agent(agent, "imatinib")

    assert isinstance(output, MechanismOutput)

    # --- drug_targets ---
    assert len(output.drug_targets) >= 10
    assert _EXPECTED_TARGET_SYMBOLS.issubset(set(output.drug_targets.keys()))
    for symbol, ensembl_id in output.drug_targets.items():
        assert ensembl_id.startswith("ENSG"), f"Expected Ensembl ID for {symbol}, got {ensembl_id!r}"

    # --- associations ---
    assert len(output.associations) >= 5
    assert _EXPECTED_ASSOCIATION_SYMBOLS.issubset(set(output.associations.keys()))
    for symbol, assocs in output.associations.items():
        assert len(assocs) >= 1, f"Expected associations for {symbol}"
    for symbol, expected_disease in _EXPECTED_TOP_DISEASES.items():
        top = output.associations[symbol][0].disease_name
        assert top == expected_disease, f"Expected top disease for {symbol} to be {expected_disease!r}, got {top!r}"

    # --- pathways ---
    assert len(output.pathways) >= 5
    all_top_level = {p.top_level_pathway for paths in output.pathways.values() for p in paths}
    assert _EXPECTED_TOP_LEVEL_PATHWAYS.issubset(all_top_level)

    # --- summary ---
    assert len(output.summary) > 200
    summary_lower = output.summary.lower()
    assert any(kw in summary_lower for kw in ("complex i", "mitochondrial", "ampk", "gdp2", "gpd2"))
