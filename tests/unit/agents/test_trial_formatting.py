"""Unit tests for the new relevance-classification columns in _trial_formatting."""

from indication_scout.agents._trial_formatting import (
    _BRIEF_SUMMARY_CAP,
    _format_interventions,
    _format_trial_row,
    _truncate_brief_summary,
)
from indication_scout.models.model_clinical_trials import (
    ArmGroup,
    Intervention,
    Trial,
)


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
    assert "interventions: Drug: Sildenafil; Drug: Placebo" in row
    assert "A study of sildenafil" in row
    assert "summary: Sildenafil for systemic hypertension in adults." in row


def test_format_trial_row_renders_separate_active_drug_arms():
    trial = _trial(
        arm_groups=[
            ArmGroup(
                label="Tadalafil",
                arm_type="Experimental",
                intervention_names=["Drug: Tadalafil"],
            ),
            ArmGroup(
                label="Sildenafil",
                arm_type="Experimental",
                intervention_names=["Drug: Sildenafil"],
            ),
        ]
    )

    row = _format_trial_row(trial, columns=("nct_id", "interventions", "arms"))

    assert row == (
        "NCT00000001 | interventions: Drug: Sildenafil; Drug: Placebo | "
        "arms: Tadalafil [Experimental]: Drug: Tadalafil; "
        "Sildenafil [Experimental]: Drug: Sildenafil"
    )


def test_format_trial_row_renders_fixed_combination_and_background_arms():
    trial = _trial(
        arm_groups=[
            ArmGroup(
                label="Combination",
                arm_type="Experimental",
                intervention_names=["Drug: Sildenafil", "Drug: Tadalafil"],
            ),
            ArmGroup(
                label="Standard care",
                arm_type="Active Comparator",
                intervention_names=["Drug: Background therapy"],
            ),
        ]
    )

    row = _format_trial_row(trial, columns=("arms",))

    assert row == (
        "arms: Combination [Experimental]: Drug: Sildenafil, Drug: Tadalafil; "
        "Standard care [Active Comparator]: Drug: Background therapy"
    )


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
    assert out == "Drug: Drug0; Drug: Drug1; Drug: Drug2; Drug: Drug3; Drug: Drug4"


def test_format_interventions_preserves_diagnostic_type():
    """Diagnostic interventions remain distinguishable from studied drugs."""
    interventions = [
        Intervention(
            intervention_type="Diagnostic Test",
            intervention_name="Corus CAD (ASGES)",
        )
    ]

    assert _format_interventions(interventions) == "Diagnostic Test: Corus CAD (ASGES)"


def test_truncate_brief_summary_none_and_empty():
    """Missing/whitespace summary → '(none)'."""
    assert _truncate_brief_summary(None) == "(none)"
    assert _truncate_brief_summary("   ") == "(none)"


def test_truncate_brief_summary_truncates_long_text():
    """Text over the cap is trimmed with an ellipsis; short text passes through."""
    short = "A short summary."
    assert _truncate_brief_summary(short) == short
    long_text = "x" * (_BRIEF_SUMMARY_CAP + 40)
    out = _truncate_brief_summary(long_text)
    assert out == "x" * _BRIEF_SUMMARY_CAP + "…"
    assert len(out) == _BRIEF_SUMMARY_CAP + 1
