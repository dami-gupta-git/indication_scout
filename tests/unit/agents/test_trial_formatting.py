"""Unit tests for the new relevance-classification columns in _trial_formatting."""

from indication_scout.agents._trial_formatting import (
    _format_arm_roles,
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
    assert "drugs: Sildenafil; Placebo" in row
    assert "A study of sildenafil" in row
    assert "summary: Sildenafil for systemic hypertension in adults." in row


def test_format_trial_row_appends_arm_roles_when_present():
    """When arm_groups carry roles, the interventions column appends an [arms: ...] clause so the
    relevance gate can see EXPERIMENTAL vs comparator without inferring from the title.
    """
    row = _format_trial_row(
        _trial(
            arm_groups=[
                ArmGroup(label="Nebivolol", arm_type="EXPERIMENTAL"),
                ArmGroup(label="Sildenafil", arm_type="ACTIVE_COMPARATOR"),
            ]
        ),
        columns=("nct_id", "interventions"),
    )
    assert "drugs: Sildenafil; Placebo" in row
    assert "[arms: Nebivolol=EXPERIMENTAL; Sildenafil=ACTIVE_COMPARATOR]" in row


def test_format_trial_row_omits_arm_clause_when_no_arm_groups():
    """No arm_groups → no [arms: ...] clause; the drugs column renders bare."""
    row = _format_trial_row(_trial(arm_groups=[]), columns=("nct_id", "interventions"))
    assert "drugs: Sildenafil; Placebo" in row
    assert "[arms:" not in row


def test_format_arm_roles_renders_label_type_pairs():
    arms = [
        ArmGroup(label="Nebivolol 5 mg", arm_type="EXPERIMENTAL"),
        ArmGroup(label="Sildenafil 25 mg", arm_type="ACTIVE_COMPARATOR"),
    ]
    assert (
        _format_arm_roles(arms)
        == "Nebivolol 5 mg=EXPERIMENTAL; Sildenafil 25 mg=ACTIVE_COMPARATOR"
    )


def test_format_arm_roles_empty_and_blank_render_empty_string():
    """No arm_groups → ''; arms missing label or type are skipped."""
    assert _format_arm_roles([]) == ""
    assert _format_arm_roles([ArmGroup(label="", arm_type="EXPERIMENTAL")]) == ""
    assert _format_arm_roles([ArmGroup(label="Arm A", arm_type="")]) == ""


def test_format_arm_roles_caps_at_five():
    arms = [ArmGroup(label=f"Arm{i}", arm_type="EXPERIMENTAL") for i in range(8)]
    out = _format_arm_roles(arms)
    assert out == "; ".join(f"Arm{i}=EXPERIMENTAL" for i in range(5))


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
