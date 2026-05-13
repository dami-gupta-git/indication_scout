"""Unit tests for MechanismOutput defaults and null-coercion.

Focuses on the `candidates` field added for the mechanism-agent candidate
pipeline — it defaults to [] and must survive null inputs from upstream
code that might hand the model a None.
"""

from indication_scout.agents.mechanism.mechanism_output import (
    MechanismCandidate,
    MechanismOutput,
)


def test_mechanism_output_candidates_defaults_to_empty_list():
    out = MechanismOutput()
    assert out.candidates == []


def test_mechanism_output_coerces_null_candidates_to_empty_list():
    """A null list (from e.g. a missing field upstream) coerces to []
    via the model_validator, rather than breaking downstream iteration."""
    out = MechanismOutput(candidates=None)
    assert out.candidates == []


def test_mechanism_output_candidates_populates():
    candidate = MechanismCandidate(
        target_symbol="GLP1R",
        action_type="AGONIST",
        disease_name="obesity",
        disease_description="Excess body fat.",
        target_function="GLP-1 receptor.",
    )
    out = MechanismOutput(candidates=[candidate])
    assert len(out.candidates) == 1
    assert out.candidates[0].target_symbol == "GLP1R"
