"""Prompt-contract tests for the supervisor system prompt.

These guard instructions that encode a behavioral decision and would silently regress if the
prompt text drifted. They assert presence of key phrases, not full wording.
"""

from indication_scout.agents.supervisor.supervisor_agent import SYSTEM_PROMPT


def test_registry_not_trial_universe_rule_present():
    """The registry≠trial-universe rule must be in the production prompt so 0 registry trials
    + moderate/strong literature is not erased as 'Untested'."""
    assert "REGISTRY ≠ TRIAL UNIVERSE" in SYSTEM_PROMPT
    # Gated on moderate/strong only — weak/none with 0 trials stays untested.
    assert "MODERATE or STRONG" in SYSTEM_PROMPT
    assert "off-registry clinical evidence" in SYSTEM_PROMPT


def test_zero_evidence_rule_is_direction_gated():
    """The Untested/Rationale-only trigger must require literature direction "none", not fire
    on 0 registry trials alone — a contradicts body is a disproven negative, not untested."""
    assert 'direction is "none"' in SYSTEM_PROMPT
    # The no-trials label still keys off strength none/weak.
    assert 'strength is "none"/"weak"' in SYSTEM_PROMPT


def test_evidence_direction_rule_present():
    """DIRECTION must be modeled separately from strength so a robustly-disproven hypothesis
    (strong + contradicts) is surfaced as a bottom-ranked negative, not erased as 'none'."""
    assert "EVIDENCE DIRECTION" in SYSTEM_PROMPT
    assert "contradicts" in SYSTEM_PROMPT
    # The contradicts pair must be a disproven negative, not erased as "no evidence".
    assert "robustly DISPROVEN hypothesis" in SYSTEM_PROMPT


def test_evidence_gate_keys_off_direction():
    """The hard-rule evidence gate must exclude on direction=none, not strength=none, so a
    zero-trial contradicts pair survives the gate."""
    assert "direction=none" in SYSTEM_PROMPT
