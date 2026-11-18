"""Unit tests for the arm-attribution fact: can a trial measure THIS drug?

A drug named in a trial record is not necessarily a drug the trial evaluates. In a fixed
multi-drug regimen, or a menu where the investigator picks one agent from a list, the arm's
outcome belongs to the regimen or the strategy, not to any single agent. These tests pin the
shapes with real registry structures: INTENSIFY (NCT05973786), where bupropion is one of four
interchangeable antidepressants in the experimental arm, versus ordinary single-agent and
two-drug combination trials, which stay attributable.
"""

import pytest

from indication_scout.agents._trial_signals import (
    derive_trial_signals,
    is_drug_isolated_in_arms,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.models.model_clinical_trials import (
    ArmGroup,
    CompletedTrialsResult,
    Intervention,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.services.dev_stage import (
    _render_active_programs,
    dev_stage_phrase,
)

_EIPT_ARM = "Bipolar Depression EIPT: Switch to one of the following combinations:"
_TAU_ARM = "Bipolar Depression TAU: Switch to quetiapine plus lithium or valproate acid"

# NCT05973786 (INTENSIFY BD), as the registry reports it: bupropion is one of four
# interchangeable antidepressants in the experimental arm, each paired with two of three mood
# stabilizers. The comparator arm carries no bupropion.
_INTENSIFY = Trial(
    nct_id="NCT05973786",
    title="Six Week Intensified Pharmacological Treatment for Bipolar Depression",
    phase="Phase 3",
    overall_status="RECRUITING",
    interventions=[
        Intervention(
            intervention_type="Drug",
            intervention_name=name,
            arm_group_labels=[_EIPT_ARM],
        )
        for name in ("Escitalopram", "Sertraline", "Venlafaxine", "Bupropion")
    ]
    + [
        Intervention(
            intervention_type="Drug",
            intervention_name=name,
            arm_group_labels=[_EIPT_ARM, _TAU_ARM],
        )
        for name in ("Lithium", "Valproate acid", "Quetiapine")
    ],
    arm_groups=[
        ArmGroup(label=_EIPT_ARM, arm_type="Experimental", intervention_names=[]),
        ArmGroup(label=_TAU_ARM, arm_type="Active Comparator", intervention_names=[]),
    ],
)

# NCT00943618: varenicline + bupropion, a genuine two-drug combination trial. The drug shares
# its arm, but the arm is small enough that the trial still tests bupropion's contribution.
# This is the shape the threshold must NOT flag.
_COMBINATION = Trial(
    nct_id="NCT00943618",
    phase="Phase 3",
    overall_status="COMPLETED",
    interventions=[
        Intervention(
            intervention_type="Drug", intervention_name=name, arm_group_labels=["Arm 1"]
        )
        for name in ("Varenicline", "Bupropion")
    ],
    arm_groups=[ArmGroup(label="Arm 1", arm_type="Experimental")],
)

# NCT03374956 (Individualized Obesity Pharmacotherapy), the trial that sits exactly at the
# threshold: three agents in every arm, the investigator picks one. Both arms carry the drug, so
# the gate reads the minimum crowding across them.
_BOUNDARY_MENU = Trial(
    nct_id="NCT03374956",
    phase="Phase 3",
    overall_status="COMPLETED",
    interventions=[
        Intervention(
            intervention_type="Drug",
            intervention_name=name,
            arm_group_labels=["Intervention group", "Control Group"],
        )
        for name in ("Phentermine-Topiramate", "Liraglutide", "Naltrexone/bupropion")
    ],
    arm_groups=[
        ArmGroup(label="Intervention group", arm_type="Experimental"),
        ArmGroup(label="Control Group", arm_type="Active Comparator"),
    ],
)

# NCT01477931: single-agent Wellbutrin XL trial — the unambiguous attributable case.
_SINGLE_AGENT = Trial(
    nct_id="NCT01477931",
    phase="Phase 4",
    overall_status="COMPLETED",
    interventions=[
        Intervention(
            intervention_type="Drug",
            intervention_name="Bupropion XL",
            arm_group_labels=["Treatment"],
        )
    ],
    arm_groups=[ArmGroup(label="Treatment", arm_type="Experimental")],
)

# Older records populate armGroups.interventionNames (type-prefixed) and leave
# interventions[].armGroupLabels empty. The fallback path must read these.
_LEGACY_MENU = Trial(
    nct_id="NCT05603104",
    phase="Phase 3",
    overall_status="RECRUITING",
    interventions=[
        Intervention(intervention_type="Drug", intervention_name=name)
        for name in (
            "Clozapine",
            "Escitalopram",
            "Sertraline",
            "Venlafaxine",
            "Lithium",
            "Bupropion",
        )
    ],
    arm_groups=[
        ArmGroup(
            label="EIPT",
            arm_type="Experimental",
            intervention_names=[
                "Drug: Clozapine",
                "Drug: Escitalopram",
                "Drug: Sertraline",
                "Drug: Venlafaxine",
                "Drug: Lithium",
                "Drug: Bupropion",
            ],
        )
    ],
)


@pytest.mark.parametrize(
    "trial,expected",
    [
        (_INTENSIFY, False),
        (_LEGACY_MENU, False),
        (_BOUNDARY_MENU, False),
        (_COMBINATION, True),
        (_SINGLE_AGENT, True),
    ],
)
def test_arm_attribution_by_trial_shape(trial, expected):
    """Menu and regimen arms are non-attributable; combination and single-agent arms are not.

    _BOUNDARY_MENU sits exactly at DRUG_ARM_CROWDING_THRESHOLD and _COMBINATION one below it —
    together they pin the cut, which is the calibration claim the constant documents.
    """
    assert is_drug_isolated_in_arms(trial, "bupropion") is expected


def test_factorial_designs_stay_attributable_despite_crowded_arms():
    """NCT06894784 (semaglutide x T1D): a 2x2 factorial whose arms each list 4 interventions.

    A factorial trial administers several agents precisely so each one's individual effect can be
    estimated, so its crowding is the opposite of a menu. Without the guard the crowding rule
    alone suppresses a genuine development claim — this trial is cited in the semaglutide golden.
    """
    factorial = Trial(
        nct_id="NCT06894784",
        phase="Phase 3",
        overall_status="RECRUITING",
        intervention_model="Factorial",
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name=name,
                arm_group_labels=["Semaglutide + AID", "Placebo + AID"],
            )
            for name in (
                "Period 1: Semaglutide + Empagliflozin",
                "Period 2: Semaglutide + Empagliflozin Placebo",
                "Period 3: Semaglutide Placebo + Empagliflozin",
                "Period 4: Semaglutide Placebo + Empagliflozin Placebo",
            )
        ],
        arm_groups=[
            ArmGroup(label="Semaglutide + AID", arm_type="Experimental"),
            ArmGroup(label="Placebo + AID", arm_type="Active Comparator"),
        ],
    )

    assert is_drug_isolated_in_arms(factorial, "semaglutide") is True
    # The same record without the factorial declaration is suppressed on crowding alone, which is
    # what makes the guard load-bearing rather than incidental.
    assert (
        is_drug_isolated_in_arms(
            factorial.model_copy(update={"intervention_model": "Parallel"}),
            "semaglutide",
        )
        is False
    )


@pytest.mark.parametrize(
    "trial,drug",
    [
        (Trial(nct_id="NCT00012558", phase="Not Applicable"), "bupropion"),
        (_SINGLE_AGENT, "lithium"),
    ],
)
def test_unknowable_cases_return_none(trial, drug):
    """No arm structure, or the drug absent entirely, is unknown — not non-attributable.

    None must stay distinct from False: a naturalistic study that reports no arms is not
    evidence that the drug shares an arm, and must not suppress the development claim.
    """
    assert is_drug_isolated_in_arms(trial, drug) is None


def test_signals_flag_non_attributable_active_phase3():
    """The INTENSIFY shape: active Phase 3 on record, but no drug-specific readout possible."""
    ct = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=0, trials=[]),
        terminated=TerminatedTrialsResult(total_count=0, trials=[]),
        search=SearchTrialsResult(total_count=1, trials=[_INTENSIFY]),
    )
    sig = derive_trial_signals(ct, drug="bupropion")

    assert sig.has_active_phase3 is True
    assert sig.active_phase3_nct_ids == ["NCT05973786"]
    assert sig.non_attributable_active_phase3_nct_ids == ["NCT05973786"]
    assert sig.active_phase3_all_non_attributable is True


def test_signals_leave_attributable_active_phase3_unflagged():
    """A single-agent active Phase 3 keeps the plain active-development reading."""
    active_single = _SINGLE_AGENT.model_copy(
        update={"phase": "Phase 3", "overall_status": "RECRUITING"}
    )
    ct = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=0, trials=[]),
        terminated=TerminatedTrialsResult(total_count=0, trials=[]),
        search=SearchTrialsResult(total_count=1, trials=[active_single]),
    )
    sig = derive_trial_signals(ct, drug="bupropion")

    assert sig.has_active_phase3 is True
    assert sig.non_attributable_active_phase3_nct_ids == []
    assert sig.active_phase3_all_non_attributable is False


def test_stage_phrase_drops_the_development_claim_when_nothing_is_attributable():
    """The reported headline must not promise a readout that will never report on this drug."""
    ct = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=0, trials=[]),
        terminated=TerminatedTrialsResult(total_count=0, trials=[]),
        search=SearchTrialsResult(total_count=1, trials=[_INTENSIFY]),
    )
    sig = derive_trial_signals(ct, drug="bupropion")
    sig.dev_stage = "active_phase3"

    phrase = dev_stage_phrase(sig)
    assert phrase == (
        "No drug-specific trial programme for this indication; the drug appears as one of "
        "several agents within an active pivotal trial (NCT05973786)"
    )


def test_active_programs_line_qualifies_the_count():
    """The count stays accurate; the qualifier stops it reading as a development program."""
    line = _render_active_programs([_INTENSIFY], drug="bupropion")
    assert line == (
        "1 Phase 3 active (NCT05973786) — none drug-specific: the drug shares its arm with "
        "several other agents in every active trial, so no drug-specific readout is possible"
    )


def test_active_programs_line_unchanged_without_a_drug_name():
    """Omitting the drug leaves every existing caller's output byte-identical."""
    assert _render_active_programs([_INTENSIFY]) == "1 Phase 3 active (NCT05973786)"
