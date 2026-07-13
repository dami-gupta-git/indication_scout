"""Unit tests for helpers/drug_helpers."""

import pytest

from indication_scout.helpers.drug_helpers import DrugIntake, normalize_drug_name


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("Dextromethorphan Hydrobromide", "dextromethorphan"),
        ("Sertraline Hydrochloride", "sertraline"),
        ("Morphine Sulfate", "morphine"),
        ("Imatinib Mesylate", "imatinib"),
        ("Metoprolol Tartrate", "metoprolol"),
    ],
)
def test_strips_salt_suffix_group1(input_name, expected):
    assert normalize_drug_name(input_name) == expected


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("Enalapril Maleate", "enalapril"),
        ("Codeine Phosphate", "codeine"),
        ("Metoprolol Succinate", "metoprolol"),
        ("Semaglutide", "semaglutide"),
        ("ibuprofen", "ibuprofen"),
    ],
)
def test_strips_salt_suffix_group2(input_name, expected):
    assert normalize_drug_name(input_name) == expected


@pytest.mark.parametrize(
    "input_name, expected",
    [
        # "sulfate" only stripped at end — "sulfated" should not match
        ("Heparin Sulfated", "heparin sulfated"),
        # "acetate" is not in SALT_SUFFIXES — second word should be kept
        ("Medroxyprogesterone Acetate", "medroxyprogesterone acetate"),
    ],
)
def test_does_not_strip_non_salt_suffix(input_name, expected):
    assert normalize_drug_name(input_name) == expected


def test_drug_intake_populated_fields():
    intake = DrugIntake(
        chembl_id="CHEMBL1431",
        aliases=["metformin", "glucophage"],
        first_approval=1995,
        approved_indications=["type 2 diabetes mellitus"],
    )
    assert intake.chembl_id == "CHEMBL1431"
    assert intake.aliases == ["metformin", "glucophage"]
    assert intake.first_approval == 1995
    assert intake.approved_indications == ["type 2 diabetes mellitus"]


def test_drug_intake_defaults_when_empty():
    intake = DrugIntake(chembl_id="CHEMBL25")
    assert intake.chembl_id == "CHEMBL25"
    assert intake.aliases == []
    assert intake.first_approval is None
    assert intake.approved_indications == []


def test_drug_intake_coerces_none_to_defaults():
    # coerce_nones: None for a field whose default is non-None becomes that default. first_approval's
    # default IS None (genuinely optional), so a None stays None — not fabricated.
    intake = DrugIntake(
        chembl_id="CHEMBL1431",
        aliases=None,
        first_approval=None,
        approved_indications=None,
    )
    assert intake.chembl_id == "CHEMBL1431"
    assert intake.aliases == []
    assert intake.first_approval is None
    assert intake.approved_indications == []
