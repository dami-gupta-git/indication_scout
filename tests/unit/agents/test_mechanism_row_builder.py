"""Unit tests for mechanism_row_builder.build_candidate_rows.

Mocks OpenTargetsClient at the method level so we can verify the row
contract without hitting OT. Covers: basic shape, top_n truncation,
score-descending sort, empty function_descriptions fallback, missing
evidence buckets, and the empty-associations edge case.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from indication_scout.agents.mechanism.mechanism_row_builder import (
    build_candidate_rows,
)
from indication_scout.models.model_open_targets import (
    Association,
    EvidenceRecord,
    TargetData,
)


def _assoc(
    disease_id: str, disease_name: str, score: float, desc: str = ""
) -> Association:
    return Association(
        disease_id=disease_id,
        disease_name=disease_name,
        disease_description=desc,
        overall_score=score,
        datatype_scores={},
        therapeutic_areas=[],
    )


def _ev(disease_id: str, dir_t: str = "GoF", dir_trait: str = "risk") -> EvidenceRecord:
    return EvidenceRecord(
        disease_id=disease_id,
        datatype_id="genetic_association",
        direction_on_target=dir_t,
        direction_on_trait=dir_trait,
    )


def _mock_client(
    target: TargetData, evidences_by_disease: dict[str, list[EvidenceRecord]]
) -> MagicMock:
    client = MagicMock()
    client.get_target_data = AsyncMock(return_value=target)
    client.get_target_evidences = AsyncMock(return_value=evidences_by_disease)
    return client


async def test_build_candidate_rows_basic_shape():
    """Each row carries the exact keys select_top_candidates expects, with
    values pulled from target data + evidence map."""
    target = TargetData(
        target_id="ENSG0001",
        symbol="GLP1R",
        function_descriptions=["GLP-1 receptor description."],
        associations=[_assoc("EFO_001", "T2D", 0.9, desc="Sugar disease.")],
    )
    evidences = {"EFO_001": [_ev("EFO_001")]}
    client = _mock_client(target, evidences)

    [row] = await build_candidate_rows(client, "ENSG0001", {"AGONIST"}, top_n=5)

    assert row == {
        "target_symbol": "GLP1R",
        "action_types": {"AGONIST"},
        "disease_name": "T2D",
        "disease_id": "EFO_001",
        "overall_score": 0.9,
        "evidences": evidences["EFO_001"],
        "disease_description": "Sugar disease.",
        "target_function": "GLP-1 receptor description.",
    }


async def test_build_candidate_rows_sorts_by_score_desc_and_trims_top_n():
    target = TargetData(
        target_id="ENSG0001",
        symbol="TGT",
        function_descriptions=["fn"],
        associations=[
            _assoc("EFO_LOW", "low", 0.3),
            _assoc("EFO_HIGH", "high", 0.9),
            _assoc("EFO_MID", "mid", 0.6),
        ],
    )
    client = _mock_client(target, {"EFO_HIGH": [], "EFO_MID": []})

    rows = await build_candidate_rows(client, "ENSG0001", {"INHIBITOR"}, top_n=2)

    assert [r["disease_name"] for r in rows] == ["high", "mid"]
    # get_target_evidences was called with the top-2 efo_ids in sorted order.
    client.get_target_evidences.assert_awaited_once_with(
        "ENSG0001", ["EFO_HIGH", "EFO_MID"]
    )


async def test_build_candidate_rows_empty_function_descriptions_defaults_empty_string():
    target = TargetData(
        target_id="ENSG0001",
        symbol="TGT",
        function_descriptions=[],
        associations=[_assoc("EFO_001", "d", 0.5)],
    )
    client = _mock_client(target, {"EFO_001": []})

    [row] = await build_candidate_rows(client, "ENSG0001", {"AGONIST"}, top_n=5)

    assert row["target_function"] == ""


async def test_build_candidate_rows_missing_evidence_bucket_becomes_empty_list():
    """If the evidence fetch returns no bucket for a disease, the row still
    carries an empty list (not None, not missing)."""
    target = TargetData(
        target_id="ENSG0001",
        symbol="TGT",
        function_descriptions=["fn"],
        associations=[_assoc("EFO_001", "d", 0.5)],
    )
    # ev_map returns no key for EFO_001 — get() falls back to [].
    client = _mock_client(target, {})

    [row] = await build_candidate_rows(client, "ENSG0001", {"AGONIST"}, top_n=5)

    assert row["evidences"] == []


async def test_build_candidate_rows_no_associations_returns_empty_list():
    target = TargetData(
        target_id="ENSG0001",
        symbol="TGT",
        function_descriptions=["fn"],
        associations=[],
    )
    client = _mock_client(target, {})

    rows = await build_candidate_rows(client, "ENSG0001", {"AGONIST"}, top_n=5)

    assert rows == []
    # No efo_ids → get_target_evidences still gets called with []; that's
    # fine because the client itself short-circuits on empty.
    client.get_target_evidences.assert_awaited_once_with("ENSG0001", [])


async def test_build_candidate_rows_uses_first_function_description():
    """When multiple function descriptions are present, only the first is used."""
    target = TargetData(
        target_id="ENSG0001",
        symbol="TGT",
        function_descriptions=["first", "second", "third"],
        associations=[_assoc("EFO_001", "d", 0.5)],
    )
    client = _mock_client(target, {"EFO_001": []})

    [row] = await build_candidate_rows(client, "ENSG0001", {"AGONIST"}, top_n=5)

    assert row["target_function"] == "first"
