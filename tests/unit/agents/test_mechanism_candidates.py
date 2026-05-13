"""Unit tests for mechanism_candidates pure helpers."""

import pytest

from indication_scout.agents.mechanism.mechanism_candidates import (
    aggregate_directions,
    classify_positive,
    clean_function_description,
    select_top_candidates,
)
from indication_scout.models.model_open_targets import EvidenceRecord


def _ev(dir_target: str | None = None, dir_trait: str | None = None) -> EvidenceRecord:
    """Tiny factory for EvidenceRecord; only direction fields matter here."""
    return EvidenceRecord(
        disease_id="EFO_0000001",
        datatype_id="genetic_association",
        direction_on_target=dir_target,
        direction_on_trait=dir_trait,
    )


# --- classify_positive --------------------------------------------------------


@pytest.mark.parametrize(
    "action_types,dir_targets,dir_traits,expected,reason",
    [
        # Aligned pairs — candidate
        (
            {"AGONIST"},
            {"GoF"},
            {"protect"},
            True,
            "agonist + GoF protects (LoF-driven disease)",
        ),
        (
            {"AGONIST"},
            {"LoF"},
            {"risk"},
            True,
            "agonist + LoF risks (LoF-driven disease)",
        ),
        (
            {"INHIBITOR"},
            {"GoF"},
            {"risk"},
            True,
            "inhibitor + GoF risks (GoF-driven disease)",
        ),
        (
            {"INHIBITOR"},
            {"LoF"},
            {"protect"},
            True,
            "inhibitor + LoF protects (GoF-driven disease)",
        ),
        ({"ANTAGONIST"}, {"GoF"}, {"risk"}, True, "antagonist is LoF-class"),
        ({"ACTIVATOR"}, {"LoF"}, {"risk"}, True, "activator is GoF-class"),
        # Opposed pairs — contraindication, NOT a candidate
        (
            {"INHIBITOR"},
            {"LoF"},
            {"risk"},
            False,
            "inhibitor would worsen LoF-driven disease",
        ),
        (
            {"AGONIST"},
            {"GoF"},
            {"risk"},
            False,
            "agonist would worsen GoF-driven disease",
        ),
        # Unknown drug action — not a candidate
        (set(), {"LoF"}, {"risk"}, False, "no action_types"),
        ({"BINDING AGENT"}, {"LoF"}, {"risk"}, False, "unrecognised action type"),
        # Mixed action — genuinely ambiguous, not a candidate
        (
            {"INHIBITOR", "AGONIST"},
            {"GoF"},
            {"protect"},
            False,
            "drug both inhibits and agonizes",
        ),
        # Unknown disease direction — not a candidate
        ({"INHIBITOR"}, set(), set(), False, "no direction evidence"),
        ({"INHIBITOR"}, {"GoF"}, set(), False, "directionOnTrait empty"),
        ({"INHIBITOR"}, set(), {"risk"}, False, "directionOnTarget empty"),
        # Inconclusive disease direction — not a candidate
        (
            {"INHIBITOR"},
            {"GoF", "LoF"},
            {"risk"},
            False,
            "contradictory directionOnTarget",
        ),
        (
            {"INHIBITOR"},
            {"GoF"},
            {"risk", "protect"},
            False,
            "contradictory directionOnTrait",
        ),
    ],
)
def test_classify_positive(action_types, dir_targets, dir_traits, expected, reason):
    assert classify_positive(action_types, dir_targets, dir_traits) is expected, reason


# --- clean_function_description ----------------------------------------------


def test_clean_function_description_strips_pubmed_and_eco():
    raw = (
        "G-protein coupled receptor for glucagon-like peptide 1 "
        "(PubMed:19861722, PubMed:26308095). Plays a role in regulating "
        "insulin secretion (By similarity). "
        "{ECO:0000250|UniProtKB:O35659, ECO:0000269|PubMed:19861722}."
    )
    cleaned = clean_function_description(raw)
    assert "PubMed" not in cleaned
    assert "ECO" not in cleaned
    assert "By similarity" not in cleaned
    assert cleaned.startswith("G-protein coupled receptor")
    # No stranded whitespace before punctuation
    assert " ." not in cleaned
    assert " ," not in cleaned


def test_clean_function_description_empty_and_noise_only():
    assert clean_function_description("") == ""
    assert clean_function_description(None) == ""
    assert clean_function_description("{ECO:0000269|PubMed:1}").strip() == ""


# --- aggregate_directions ----------------------------------------------------


def test_aggregate_directions_unanimous():
    """All records agree → dominant pair is returned as singletons."""
    records = [_ev("GoF", "protect")] * 5
    dt, dtr = aggregate_directions(records)
    assert dt == {"GoF"} and dtr == {"protect"}


def test_aggregate_directions_majority_above_threshold():
    """One outlier in a 146:1 split → majority wins (99% >= 80%)."""
    records = [_ev("GoF", "protect")] * 146 + [_ev("LoF", "risk")]
    dt, dtr = aggregate_directions(records)
    assert dt == {"GoF"} and dtr == {"protect"}


def test_aggregate_directions_below_threshold_returns_empty():
    """Split 3:2 → neither side reaches 80% → inconclusive."""
    records = [_ev("GoF", "protect")] * 3 + [_ev("LoF", "risk")] * 2
    dt, dtr = aggregate_directions(records)
    assert dt == set() and dtr == set()


def test_aggregate_directions_ignores_records_missing_either_field():
    """Records with only one direction field populated are not counted."""
    records = [
        _ev("GoF", "protect"),
        _ev("GoF", "protect"),
        _ev("GoF", None),  # ignored
        _ev(None, "risk"),  # ignored
        _ev(None, None),  # ignored
    ]
    dt, dtr = aggregate_directions(records)
    assert dt == {"GoF"} and dtr == {"protect"}


def test_aggregate_directions_no_direction_labeled_records():
    """No records carry both fields → returns empty sets."""
    records = [_ev(None, None), _ev("GoF", None)]
    dt, dtr = aggregate_directions(records)
    assert dt == set() and dtr == set()


def test_aggregate_directions_empty_input():
    assert aggregate_directions([]) == (set(), set())


def test_aggregate_directions_custom_threshold():
    """Passing a lower threshold allows a weaker majority to pass."""
    records = [_ev("GoF", "protect")] * 3 + [_ev("LoF", "risk")] * 2
    # At 60% threshold, 3/5 = 60% passes.
    dt, dtr = aggregate_directions(records, min_fraction=0.6)
    assert dt == {"GoF"} and dtr == {"protect"}


# --- select_top_candidates ----------------------------------------------------


def _row(
    target_symbol="TGT",
    action_types=None,
    disease_name="disease x",
    overall_score=0.5,
    evidences=None,
    disease_description="",
    target_function="",
):
    return {
        "target_symbol": target_symbol,
        "action_types": action_types if action_types is not None else {"INHIBITOR"},
        "disease_name": disease_name,
        "overall_score": overall_score,
        "evidences": evidences or [],
        "disease_description": disease_description,
        "target_function": target_function,
    }


def test_select_top_candidates_keeps_only_positive_rows():
    rows = [
        _row(
            disease_name="pos-a", overall_score=0.9, evidences=[_ev("GoF", "risk")] * 5
        ),
        # Opposed: inhibitor + LoF-driven disease
        _row(
            disease_name="contra-b",
            overall_score=0.8,
            evidences=[_ev("LoF", "risk")] * 5,
        ),
        # Unknown direction — no records with both fields
        _row(disease_name="unk-c", overall_score=0.85, evidences=[]),
    ]
    out = select_top_candidates(rows, approved_diseases=set(), limit=5)
    names = [c.disease_name for c in out]
    assert names == ["pos-a"]


def test_select_top_candidates_majority_vote_survives_outlier():
    """A single outlier evidence record does not flip the verdict to False."""
    rows = [
        _row(
            disease_name="pos-a",
            overall_score=0.9,
            evidences=[_ev("GoF", "risk")] * 100 + [_ev("LoF", "protect")],
        ),
    ]
    out = select_top_candidates(rows, approved_diseases=set(), limit=5)
    assert [c.disease_name for c in out] == ["pos-a"]


def test_select_top_candidates_excludes_approved_diseases_exact_match():
    """Approved disease names must match the OT disease_name exactly
    (case-insensitive). Synonym / substring resolution is the caller's
    problem (see services.approval_check.get_fda_approved_disease_mapping)."""
    rows = [
        _row(
            disease_name="type 2 diabetes mellitus",
            overall_score=0.9,
            evidences=[_ev("GoF", "risk")] * 5,
        ),
        _row(
            disease_name="Parkinson disease",
            overall_score=0.7,
            evidences=[_ev("GoF", "risk")] * 5,
        ),
    ]
    # Casing differs → still matches (lowercase comparison).
    out = select_top_candidates(
        rows,
        approved_diseases={"Type 2 Diabetes Mellitus"},
        limit=5,
    )
    names = [c.disease_name for c in out]
    assert names == ["Parkinson disease"]


def test_select_top_candidates_does_not_match_approved_substring():
    """A parent term like 'diabetes mellitus' does NOT filter out the more
    specific 'type 2 diabetes mellitus' — exact match only. If the caller
    wants both filtered, both must be in the approved set."""
    rows = [
        _row(
            disease_name="type 2 diabetes mellitus",
            overall_score=0.9,
            evidences=[_ev("GoF", "risk")] * 5,
        ),
    ]
    out = select_top_candidates(
        rows,
        approved_diseases={"diabetes mellitus"},
        limit=5,
    )
    assert [c.disease_name for c in out] == ["type 2 diabetes mellitus"]


def test_select_top_candidates_sorts_by_score_desc_and_applies_limit():
    rows = [
        _row(disease_name=f"d{i}", overall_score=s, evidences=[_ev("GoF", "risk")] * 5)
        for i, s in enumerate([0.4, 0.9, 0.6, 0.8, 0.5, 0.7])
    ]
    out = select_top_candidates(rows, approved_diseases=set(), limit=3)
    assert [c.disease_name for c in out] == ["d1", "d3", "d5"]  # 0.9, 0.8, 0.7


def test_select_top_candidates_builds_candidate_fields():
    raw_function = (
        "Kinase activity (PubMed:123). Binds ATP (By similarity). "
        "{ECO:0000269|PubMed:123}."
    )
    rows = [
        _row(
            target_symbol="BRAF",
            action_types={"INHIBITOR"},
            disease_name="melanoma",
            overall_score=0.9,
            evidences=[_ev("GoF", "risk")] * 5,
            disease_description="Malignant neoplasm of melanocytes.",
            target_function=raw_function,
        ),
    ]
    [c] = select_top_candidates(rows, approved_diseases=set(), limit=5)
    assert c.target_symbol == "BRAF"
    assert c.action_type == "INHIBITOR"
    assert c.disease_name == "melanoma"
    assert c.disease_description == "Malignant neoplasm of melanocytes."
    assert "PubMed" not in c.target_function
    assert "ECO" not in c.target_function
    assert c.target_function.startswith("Kinase activity")


def test_select_top_candidates_empty_rows():
    assert select_top_candidates([], approved_diseases=set(), limit=5) == []
