"""Unit tests for the new relevance-classification columns in _trial_formatting."""

from indication_scout.agents._trial_formatting import (
    _format_interventions,
    _format_trial_row,
    _truncate_brief_summary,
)
from indication_scout.models.model_clinical_trials import Intervention, Trial


def _trial(**overrides) -> Trial:
    base = dict(
        nct_id="NCT00000001",
        title="A study of sildenafil",
        phase="Phase 3",
        overall_status="COMPLETED",
        brief_summary="Sildenafil for systemic hypertension in adults.",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Sildenafil"),
            Intervention(intervention_type="Drug", intervention_name="Placebo"),
        ],
        sponsor="S",
    )
    base.update(overrides)
    return Trial(**base)


def test_format_trial_row_renders_interventions_and_brief_summary():
    """The classification columns render drug names and a truncated summary."""
    row = _format_trial_row(
        _trial(),
        columns=("nct_id", "phase", "interventions", "title", "brief_summary"),
    )
    assert "NCT00000001" in row
    assert "drugs: Sildenafil; Placebo" in row
    assert "A study of sildenafil" in row
    assert "summary: Sildenafil for systemic hypertension in adults." in row


def test_format_interventions_empty_renders_none():
    """No interventions → '(none)', never a fabricated value."""
    assert _format_interventions([]) == "(none)"
    # Interventions present but all names blank also yields '(none)'.
    blanks = [Intervention(intervention_type="Drug", intervention_name="")]
    assert _format_interventions(blanks) == "(none)"


def test_format_interventions_caps_at_five():
    """At most five drug names are rendered."""
    many = [
        Intervention(intervention_type="Drug", intervention_name=f"Drug{i}")
        for i in range(8)
    ]
    out = _format_interventions(many)
    assert out == "Drug0; Drug1; Drug2; Drug3; Drug4"


def test_truncate_brief_summary_none_and_empty():
    """Missing/whitespace summary → '(none)'."""
    assert _truncate_brief_summary(None) == "(none)"
    assert _truncate_brief_summary("   ") == "(none)"


def test_truncate_brief_summary_truncates_long_text():
    """Text over the cap is trimmed with an ellipsis; short text passes through."""
    short = "A short summary."
    assert _truncate_brief_summary(short) == short
    long_text = "x" * 200
    out = _truncate_brief_summary(long_text)
    assert out == "x" * 160 + "…"
    assert len(out) == 161
