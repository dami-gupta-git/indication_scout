"""Unit tests for runners/rag_runner — no network, no LLM, no DB calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.runners.rag_runner import run_rag

# --- Fixtures ---


@pytest.fixture
def mock_db() -> MagicMock:
    return MagicMock()


@pytest.fixture
def drug_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage"],
        target_gene_symbols=["PRKAA1"],
        mechanisms_of_action=["AMP-activated protein kinase activator"],
        atc_codes=["A10BA02"],
        atc_descriptions=["Biguanides"],
        drug_type="Small molecule",
    )


@pytest.fixture
def evidence_colorectal() -> EvidenceSummary:
    return EvidenceSummary(
        summary="Evidence for metformin in colorectal cancer.",
        study_count=3,
        study_types=["RCT"],
        strength="moderate",
        key_findings=["Reduced CRC incidence"],
        supporting_pmids=["11111111"],
    )


@pytest.fixture
def evidence_breast() -> EvidenceSummary:
    return EvidenceSummary(
        summary="Evidence for metformin in breast cancer.",
        study_count=2,
        study_types=["observational"],
        strength="weak",
        key_findings=["Lower recurrence rate"],
        supporting_pmids=["22222222"],
    )


def _make_mock_svc(
    top_indications: dict,
    drug_profile: DrugProfile,
    expand_return: list[str],
    fetch_return: list[str],
    search_side_effect,
    synthesize_side_effect,
) -> MagicMock:
    """Return a mock RetrievalService with all pipeline methods pre-configured."""
    mock_svc = MagicMock()
    mock_svc.get_drug_competitors = AsyncMock(return_value=top_indications)
    mock_svc.build_drug_profile = AsyncMock(return_value=drug_profile)
    mock_svc.expand_search_terms = AsyncMock(return_value=expand_return)
    mock_svc.fetch_and_cache = AsyncMock(return_value=fetch_return)
    mock_svc.semantic_search = AsyncMock(side_effect=search_side_effect)
    mock_svc.synthesize = AsyncMock(side_effect=synthesize_side_effect)
    return mock_svc


# --- Tests ---


async def test_run_rag_returns_evidence_per_disease(
    mock_db, drug_profile, evidence_colorectal, evidence_breast
):
    """run_rag returns a dict mapping each disease to its EvidenceSummary."""
    top_indications = {"colorectal cancer": {"aspirin"}, "breast cancer": {"tamoxifen"}}
    abstracts = [{"pmid": "11111111", "title": "T", "abstract": "A", "similarity": 0.9}]

    mock_svc = _make_mock_svc(
        top_indications=top_indications,
        drug_profile=drug_profile,
        expand_return=["metformin AND colorectal cancer"],
        fetch_return=["11111111", "22222222"],
        search_side_effect=[abstracts, abstracts],
        synthesize_side_effect=[evidence_colorectal, evidence_breast],
    )

    with patch(
        "indication_scout.runners.rag_runner.RetrievalService", return_value=mock_svc
    ):
        results = await run_rag("metformin", mock_db)

    assert set(results.keys()) == {"colorectal cancer", "breast cancer"}
    assert results["colorectal cancer"] is evidence_colorectal
    assert results["breast cancer"] is evidence_breast


async def test_run_rag_passes_drug_profile_to_expand_search_terms(
    mock_db, drug_profile
):
    """expand_search_terms receives the DrugProfile returned by build_drug_profile."""
    top_indications = {"colorectal cancer": {"aspirin"}}
    evidence = EvidenceSummary()

    mock_svc = _make_mock_svc(
        top_indications=top_indications,
        drug_profile=drug_profile,
        expand_return=["q"],
        fetch_return=[],
        search_side_effect=[[]],
        synthesize_side_effect=[evidence],
    )

    with patch(
        "indication_scout.runners.rag_runner.RetrievalService", return_value=mock_svc
    ):
        await run_rag("metformin", mock_db)

    mock_svc.expand_search_terms.assert_called_once_with(
        "metformin", "colorectal cancer", drug_profile
    )


async def test_run_rag_passes_pmids_to_semantic_search(mock_db, drug_profile):
    """semantic_search receives the PMID list returned by fetch_and_cache."""
    top_indications = {"colorectal cancer": {"aspirin"}}
    pmids = ["11111111", "22222222", "33333333"]
    evidence = EvidenceSummary()

    mock_svc = _make_mock_svc(
        top_indications=top_indications,
        drug_profile=drug_profile,
        expand_return=["q"],
        fetch_return=pmids,
        search_side_effect=[[]],
        synthesize_side_effect=[evidence],
    )

    with patch(
        "indication_scout.runners.rag_runner.RetrievalService", return_value=mock_svc
    ):
        await run_rag("metformin", mock_db)

    mock_svc.semantic_search.assert_called_once_with(
        "colorectal cancer", "metformin", pmids, mock_db
    )


async def test_run_rag_passes_top_abstracts_to_synthesize(mock_db, drug_profile):
    """synthesize receives the abstract list returned by semantic_search."""
    top_indications = {"colorectal cancer": {"aspirin"}}
    top_abstracts = [
        {"pmid": "11111111", "title": "T", "abstract": "A", "similarity": 0.9}
    ]
    evidence = EvidenceSummary()

    mock_svc = _make_mock_svc(
        top_indications=top_indications,
        drug_profile=drug_profile,
        expand_return=["q"],
        fetch_return=["11111111"],
        search_side_effect=[top_abstracts],
        synthesize_side_effect=[evidence],
    )

    with patch(
        "indication_scout.runners.rag_runner.RetrievalService", return_value=mock_svc
    ):
        await run_rag("metformin", mock_db)

    mock_svc.synthesize.assert_called_once_with(
        "metformin", "colorectal cancer", top_abstracts
    )
