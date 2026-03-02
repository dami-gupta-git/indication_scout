"""Unit tests for helpers/drug_helpers."""

import pytest

from indication_scout.helpers.drug_helpers import normalize_drug_name


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
