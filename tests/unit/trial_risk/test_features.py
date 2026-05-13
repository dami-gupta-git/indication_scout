"""Unit tests for trial_risk.features."""

import math

import pytest

from indication_scout.models.model_clinical_trials import (
    Intervention,
    MeshTerm,
    PrimaryOutcome,
    Trial,
)
from indication_scout.ml_models.trial_risk.features import (
    PHASES,
    build_features,
    parse_year,
    sponsor_class,
    vectorize,
)
from indication_scout.ml_models.trial_risk.literature import LiteratureSignals


@pytest.mark.parametrize(
    "sponsor, expected",
    [
        ("Pfizer Inc", "industry"),
        ("Stanford University", "academic"),
        ("National Cancer Institute", "nih"),
        ("Mayo Clinic", "academic"),
        ("", "unknown"),
    ],
)
def test_sponsor_class(sponsor, expected):
    assert sponsor_class(sponsor) == expected


@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("2020-05-12", 2020),
        ("2018-03", 2018),
        ("", None),
        (None, None),
        ("not-a-date", None),
    ],
)
def test_parse_year(date_str, expected):
    assert parse_year(date_str) == expected


def test_build_features_full_trial():
    trial = Trial(
        nct_id="NCT00000001",
        title="t",
        phase="Phase 2",
        overall_status="Completed",
        sponsor="Pfizer Inc",
        enrollment=200,
        start_date="2018-06-15",
        completion_date="2021-03-01",
        mesh_conditions=[MeshTerm(id="D003", term="Diabetes")],
        mesh_ancestors=[MeshTerm(id="D001", term="A"), MeshTerm(id="D002", term="B")],
        interventions=[Intervention(intervention_type="Drug", intervention_name="X")],
        primary_outcomes=[
            PrimaryOutcome(measure="HbA1c", time_frame="24w"),
            PrimaryOutcome(measure="Weight", time_frame="24w"),
        ],
    )
    lit = LiteratureSignals(
        failure_signal=0.42,
        safety_signal=0.55,
        efficacy_signal=0.71,
        available=True,
    )

    row = build_features(trial, lit)
    f = row.features

    assert row.nct_id == "NCT00000001"
    assert f["phase__Phase 2"] == 1.0
    assert f["phase__Phase 3"] == 0.0
    assert sum(f[f"phase__{p}"] for p in PHASES) == 1.0
    assert f["sponsor__industry"] == 1.0
    assert f["sponsor__academic"] == 0.0
    assert f["has_enrollment"] == 1.0
    assert f["log_enrollment"] == pytest.approx(math.log1p(200))
    assert f["n_mesh_ancestors"] == 2.0
    assert f["n_interventions"] == 1.0
    assert f["n_primary_outcomes"] == 2.0
    assert f["has_start_date"] == 1.0
    assert f["start_year"] == 2018.0
    assert f["lit_failure_signal"] == 0.42
    assert f["lit_safety_signal"] == 0.55
    assert f["lit_efficacy_signal"] == 0.71
    assert f["lit_signal_available"] == 1.0


def test_build_features_missing_optional_fields():
    trial = Trial(
        nct_id="NCT00000002",
        phase="Phase 1",
        overall_status="Terminated",
        sponsor="",
        enrollment=None,
        start_date=None,
    )
    lit = LiteratureSignals()

    row = build_features(trial, lit)
    f = row.features

    assert f["has_enrollment"] == 0.0
    assert f["log_enrollment"] == 0.0
    assert f["has_start_date"] == 0.0
    assert f["start_year"] == 0.0
    assert f["sponsor__unknown"] == 1.0
    assert f["lit_signal_available"] == 0.0
    assert f["lit_failure_signal"] == 0.0


def test_vectorize_aligns_columns_and_fills_missing():
    trial_a = Trial(nct_id="A", phase="Phase 1", sponsor="Pfizer Inc", enrollment=100)
    trial_b = Trial(nct_id="B", phase="Phase 3", sponsor="Stanford University")
    rows = [
        build_features(trial_a, LiteratureSignals()),
        build_features(trial_b, LiteratureSignals()),
    ]
    columns, matrix = vectorize(rows)

    assert columns == sorted(columns)
    assert len(matrix) == 2
    assert len(matrix[0]) == len(columns)
    assert len(matrix[1]) == len(columns)

    p1_idx = columns.index("phase__Phase 1")
    p3_idx = columns.index("phase__Phase 3")
    assert matrix[0][p1_idx] == 1.0
    assert matrix[0][p3_idx] == 0.0
    assert matrix[1][p1_idx] == 0.0
    assert matrix[1][p3_idx] == 1.0
