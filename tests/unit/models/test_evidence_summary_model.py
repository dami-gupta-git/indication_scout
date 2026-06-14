"""Unit tests for EvidenceSummary — direction modeled separately from strength."""

from indication_scout.models.model_evidence_summary import EvidenceSummary


def test_defaults_strength_and_direction_none():
    es = EvidenceSummary()
    assert es.summary == ""
    assert es.study_count == 0
    assert es.strength == "none"
    assert es.direction == "none"
    assert es.key_findings == []
    assert es.supporting_pmids == []
    assert es.contradicting_pmids == []


def test_strong_contradicts_disproven_hypothesis():
    # The core new state: lots of good evidence that the drug FAILS.
    es = EvidenceSummary(
        summary="Gefitinib showed no benefit in colorectal cancer (PMID: 16062074).",
        study_count=4,
        strength="strong",
        direction="contradicts",
        key_findings=["Phase II RCT: no PFS benefit (PMID: 16062074)"],
        supporting_pmids=[],
        contradicting_pmids=["16062074", "18667394", "16361624", "20204674"],
    )
    assert es.strength == "strong"
    assert es.direction == "contradicts"
    assert es.supporting_pmids == []
    assert es.contradicting_pmids == ["16062074", "18667394", "16361624", "20204674"]


def test_contradicting_pmids_coerced_to_str():
    es = EvidenceSummary(
        strength="moderate",
        direction="contradicts",
        contradicting_pmids=[16062074, 18667394],
    )
    assert es.contradicting_pmids == ["16062074", "18667394"]


def test_missing_direction_in_old_cache_defaults_none():
    # Old cached JSON has no `direction` key — coerce to default "none", not crash.
    old_cache = {
        "summary": "supports (PMID: 12345678)",
        "study_count": 3,
        "strength": "moderate",
        "supporting_pmids": ["12345678"],
    }
    es = EvidenceSummary(**old_cache)
    assert es.direction == "none"
    assert es.strength == "moderate"
    assert es.contradicting_pmids == []
