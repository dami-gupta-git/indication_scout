"""Unit tests for OpenTargetsClient."""

from unittest.mock import AsyncMock, patch

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import (
    ClinicalDisease,
    DrugData,
    DrugSummary,
    DrugTarget,
)

# --- OpenTargetsClient configuration ---


def test_default_config():
    """Test that client uses default settings."""
    client = OpenTargetsClient()

    assert client.timeout == 30.0
    assert client.max_retries == 3
    assert client.cache_dir is not None
    assert client.cache_dir == DEFAULT_CACHE_DIR


# --- get_drug_competitors: None stage guard ---


async def test_get_drug_competitors_skips_summary_with_none_stage(tmp_path):
    """Summaries with max_clinical_stage=None must be skipped without raising TypeError."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="depression")],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage=None,
            diseases=[ClinicalDisease(disease_name="anxiety")],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("testdrug", min_stage="PHASE_3")

    # anxiety (stage=None) must be absent; depression (PHASE_3) must be present
    assert "anxiety" not in result["diseases"]
    assert "depression" in result["diseases"]


# --- get_drug_competitors: raw diseases returned ---


async def test_get_drug_competitors_returns_raw_diseases(tmp_path):
    """get_drug_competitors returns CompetitorRawData with unmerged diseases and indications."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="narcolepsy")],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="narcolepsy-cataplexy syndrome")],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("testdrug", min_stage="PHASE_3")

    assert "narcolepsy" in result["diseases"]
    assert "narcolepsy-cataplexy syndrome" in result["diseases"]
    assert result["drug_indications"] == []


async def test_get_drug_competitors_groups_by_disease_id(tmp_path):
    """Diseases sharing the same disease_id collapse to a single key."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[
                ClinicalDisease(
                    disease_from_source="type 2 diabetes",
                    disease_id="MONDO_0005148",
                    disease_name="type 2 diabetes mellitus",
                ),
            ],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage="PHASE_3",
            diseases=[
                ClinicalDisease(
                    disease_from_source="diabetes mellitus, type 2",
                    disease_id="MONDO_0005148",
                    disease_name="diabetes mellitus",
                ),
            ],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("testdrug", min_stage="PHASE_3")

    assert len(result["diseases"]) == 1
    assert "type 2 diabetes mellitus" in result["diseases"]
    assert result["diseases"]["type 2 diabetes mellitus"] == {
        "competitor_a",
        "competitor_b",
    }
