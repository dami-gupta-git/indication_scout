"""Integration tests for services/drug_safety."""

import logging

import pytest

from indication_scout.services.drug_safety import DrugSafetyService
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


@pytest.fixture
def svc(test_cache_dir):
    """RetrievalService bound to the test cache directory."""
    return RetrievalService(test_cache_dir)


@pytest.fixture
def safety_svc(test_cache_dir):
    """DrugSafetyService bound to the test cache directory."""
    return DrugSafetyService(test_cache_dir)


async def test_safety_search_fetches_drug_level_and_disease_scoped(safety_svc):
    """safety_search fetches the DRUG-LEVEL adverse-event pool ([Majr], citation-ranked) plus the
    DISEASE-SCOPED pool, deduped. rofecoxib × colorectal cancer: the drug-wide pool surfaces the
    landmark CV papers (APPROVe, PMID 15713943), and the disease-scoped query pulls colorectal-
    context safety papers the drug-level pool alone misses."""
    results = await safety_svc.safety_search("CHEMBL122", disease="colorectal cancer")

    assert len(results.combined) > 0, "expected safety abstracts"
    pmids = {r.pmid for r in results.combined}
    # APPROVe (the trial that got rofecoxib withdrawn) — a stable drug-level landmark.
    assert "15713943" in pmids, f"expected APPROVe (15713943); got {sorted(pmids)[:10]}"


async def test_summarize_safety_prod_reports_ot_signal_and_severity(svc, safety_svc):
    """PRODUCTION summarize_safety returns a 3-tuple, states the OT-authoritative signal (rofecoxib
    withdrawal / cardiovascular), cites only provided-pool PMIDs, and severity is 'withdrawn' from
    the OT warning_type."""
    profile = await svc.build_drug_profile("CHEMBL122")
    abstracts = await safety_svc.safety_search("CHEMBL122", disease="arthritis")

    result = await safety_svc.summarize_safety(
        "CHEMBL122", "arthritis", profile, abstracts.combined
    )

    assert (
        result.safety_summary != ""
    ), "expected a non-empty safety summary for rofecoxib"
    assert "withdrawn" in result.regulatory_summary.lower()
    assert "not proof of causation" in result.pharmacovigilance_summary.lower()
    assert result.safety_severity == "withdrawn"
    assert result.safety_pmids == []


async def test_classify_indication_harm_true_for_colorectal(safety_svc):
    """classify_indication_harm returns True + cited PMIDs when the disease-scoped literature
    reports a harm for the indication. rofecoxib × colorectal has the APPROVe CV signal in
    adenoma-prevention dosing — a disease-context harm."""
    abstracts = await safety_svc.safety_search("CHEMBL122", disease="colorectal cancer")

    harm, summary, pmids = await safety_svc.classify_indication_harm(
        "CHEMBL122", "colorectal cancer", abstracts.disease_scoped
    )

    assert (
        harm is True
    ), "expected an indication-context harm for rofecoxib × colorectal"
    assert summary != ""
    pool = {r.pmid for r in abstracts.disease_scoped}
    assert all(p in pool for p in pmids), f"cited PMIDs not in provenance pool: {pmids}"
