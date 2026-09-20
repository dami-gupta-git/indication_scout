"""Unit tests for services/drug_safety — no network, no LLM calls."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.drug_safety import DrugSafetyService
from indication_scout.services.retrieval import AbstractResult

# --- Fixtures ---


@pytest.fixture
def safety_svc(tmp_path):
    return DrugSafetyService(tmp_path)


# --- safety_search (delegates to pubmed_ae.search_adverse_events: drug-level + disease-scoped) ---


def _safety_abs(pmid, title):
    from indication_scout.models.model_pubmed_abstract import PubmedAbstract

    return PubmedAbstract(pmid=pmid, title=title, abstract="body")


async def test_safety_search_fetches_and_dedupes_both_pools(safety_svc):
    """safety_search fetches the DRUG-LEVEL pool and (when disease given) the DISEASE-SCOPED pool,
    deduped drug-level-first, as AbstractResults."""
    calls = []

    async def fake_search(
        pref,
        cache_dir,
        date_before=None,
        disease=None,
        disease_aliases=None,
    ):
        calls.append((disease, disease_aliases))
        if disease is None:
            return [_safety_abs("111", "drug-wide"), _safety_abs("222", "shared")]
        return [_safety_abs("222", "shared"), _safety_abs("333", "disease-only")]

    from indication_scout.utils.cache import cache_set

    cache_set(
        "disease_aliases",
        {"chembl_id": "CHEMBL122", "disease": "colorectal cancer"},
        ["bowel cancer"],
        safety_svc.cache_dir,
    )

    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.search_adverse_events",
            new=fake_search,
        ),
    ):
        result = await safety_svc.safety_search(
            "CHEMBL122", disease="colorectal cancer"
        )

    # One drug-level call (disease=None) + one disease-scoped call.
    assert calls == [(None, None), ("colorectal cancer", ["bowel cancer"])]
    assert [r.pmid for r in result.drug_level] == ["111", "222"]
    assert [r.pmid for r in result.disease_scoped] == ["222", "333"]
    assert [r.pmid for r in result.combined] == ["111", "222", "333"]
    assert result.drug_level[0].title == "drug-wide"


async def test_safety_search_drug_level_only_when_no_disease(safety_svc):
    """With no disease, safety_search runs only the drug-level pool (one call)."""
    calls = []

    async def fake_search(pref, cache_dir, date_before=None, disease=None):
        calls.append(disease)
        return [_safety_abs("111", "x")]

    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.search_adverse_events",
            new=fake_search,
        ),
    ):
        result = await safety_svc.safety_search("CHEMBL122")

    assert calls == [None]
    assert [r.pmid for r in result.drug_level] == ["111"]
    assert result.disease_scoped == []
    assert [r.pmid for r in result.combined] == ["111"]


# --- summarize_safety (source-separated production facts; literature-only holdout) ---

_SAFETY_ABSTRACTS = [
    AbstractResult(
        pmid="11696466",
        title="Cardiovascular thrombotic events in trials of rofecoxib.",
        abstract="Rofecoxib increased cardiovascular thrombotic events versus placebo.",
        similarity=0.0,
    ),
]

_SAMPLE_SAFETY_LLM_RESPONSE = json.dumps(
    {
        "verdicts": [
            {
                "pmid": "11696466",
                "status": "confirmed_harm",
                "adverse_outcome": "increased cardiovascular thrombotic events",
                "evidence_quote": (
                    "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                ),
            }
        ]
    }
)


def _safety_profile() -> DrugProfile:
    from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning

    return DrugProfile(
        chembl_id="CHEMBL122",
        drug_warnings=[
            DrugWarning(
                warning_type="Withdrawn",
                description="Increased risk of cardiovascular events",
                toxicity_class="cardiotoxicity",
                year=2004,
            ),
        ],
        adverse_events=[
            AdverseEvent(
                name="myocardial infarction", count=100, log_likelihood_ratio=8913.0
            ),
        ],
    )


async def test_summarize_safety_prod_uses_ot_signal_and_severity(safety_svc):
    """Production separates exact label text from Open Targets warning and FAERS metadata."""
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    with (
        patch(
            "indication_scout.services.drug_safety.DrugSafetyService._get_label_safety_records",
            new=AsyncMock(
                return_value=[
                    FDALabelSafetyRecord(
                        set_id="set-1",
                        effective_time="20260101",
                        boxed_warnings=["WARNING: Cardiovascular thrombotic risk."],
                    )
                ]
            ),
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(
                side_effect=AssertionError("production must be deterministic")
            ),
        ),
    ):
        result = await safety_svc.summarize_safety(
            "CHEMBL122", "arthritis", _safety_profile(), _SAFETY_ABSTRACTS
        )

    assert "WARNING: Cardiovascular thrombotic risk." in result.regulatory_summary
    assert "warning type: Withdrawn" in result.regulatory_summary
    assert "toxicity categories: cardiotoxicity" in result.regulatory_summary
    assert "myocardial infarction" in result.pharmacovigilance_summary
    assert "not proof of causation" in result.pharmacovigilance_summary
    assert result.literature_summary == ""
    assert result.safety_pmids == []
    assert result.safety_severity == "withdrawn"
    assert result.label_data_available is True


async def test_format_regulatory_safety_single_text_skips_llm(safety_svc):
    """When every label agrees verbatim, no LLM call is needed — return the text directly."""
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    label_records = [
        FDALabelSafetyRecord(
            set_id="set-old",
            effective_time="20240101",
            brand_names=["WELLBUTRIN SR"],
            boxed_warnings=["WARNING: the same warning."],
        ),
        FDALabelSafetyRecord(
            set_id="set-new",
            effective_time="20260101",
            brand_names=["Bupropion Hydrochloride XL"],
            boxed_warnings=["WARNING: the same warning."],
        ),
    ]

    with (
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
        ),
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(
                side_effect=AssertionError("get_all_drug_names must not be called")
            ),
        ),
    ):
        summary, full_labels = await safety_svc._format_regulatory_safety(
            "CHEMBL894", label_records, warnings=[]
        )

    assert summary == (
        "FDA label boxed-warning text (all 2 FDA-approved product labels agree "
        "verbatim):\nWARNING: the same warning."
    )
    assert "WELLBUTRIN SR" in full_labels
    assert "Bupropion Hydrochloride XL" in full_labels
    assert full_labels.count("WARNING: the same warning.") == 2


async def test_format_regulatory_safety_digests_distinct_texts_via_llm(safety_svc):
    """Distinct boxed-warning texts across products go to the LLM for a summarized digest;
    the full verbatim text of every product is still returned separately for the appendix.
    """
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    label_records = [
        FDALabelSafetyRecord(
            set_id="set-old",
            effective_time="20240101",
            brand_names=["WELLBUTRIN SR"],
            boxed_warnings=["WARNING: older wording of the same warning."],
        ),
        FDALabelSafetyRecord(
            set_id="set-new",
            effective_time="20260101",
            brand_names=["Bupropion Hydrochloride XL"],
            boxed_warnings=["WARNING: newest wording, with a differing age cutoff."],
        ),
    ]
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Both labels warn of suicidality risk; the newest label differs on age cutoff."

    with (
        patch("indication_scout.services.drug_safety.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion", "wellbutrin"]),
        ),
    ):
        summary, full_labels = await safety_svc._format_regulatory_safety(
            "CHEMBL894", label_records, warnings=[]
        )

    assert "bupropion" in captured["prompt"]
    assert "WARNING: older wording of the same warning." in captured["prompt"]
    assert "WARNING: newest wording, with a differing age cutoff." in captured["prompt"]
    assert "2 product labels, LLM-summarized" in summary
    assert "differs on age cutoff" in summary
    assert "WELLBUTRIN SR" in full_labels
    assert "Bupropion Hydrochloride XL" in full_labels
    assert "WARNING: older wording of the same warning." in full_labels
    assert "WARNING: newest wording, with a differing age cutoff." in full_labels


async def test_summarize_safety_holdout_omits_ot_signal(safety_svc):
    """Holdout (date_before set): OT warnings/AEs are OMITTED from the prompt (undateable → would
    leak); literature findings require a source-verifiable exact quote."""
    from datetime import date

    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_SAFETY_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch("indication_scout.services.drug_safety.query_llm", new=capture_llm),
    ):
        result = await safety_svc.summarize_safety(
            "CHEMBL122",
            "arthritis",
            _safety_profile(),
            _SAFETY_ABSTRACTS,
            date_before=date(2003, 1, 1),
        )

    # OT warning text is NOT in the prompt (suppressed in holdout).
    assert "Withdrawn" not in captured["prompt"]
    assert "myocardial infarction" not in captured["prompt"]
    assert result.regulatory_summary == ""
    assert result.pharmacovigilance_summary == ""
    assert result.literature_summary == (
        'Date-eligible literature reported: "Rofecoxib increased cardiovascular thrombotic '
        'events versus placebo." '
        "(PMID: 11696466)."
    )
    assert result.safety_pmids == ["11696466"]
    assert result.safety_severity is None
    assert result.label_data_available is None


async def test_summarize_safety_no_signal_returns_unavailable(safety_svc):
    """No source signal returns empty summaries and unavailable severity."""
    with (
        patch.object(
            safety_svc, "_get_label_safety_records", new=AsyncMock(return_value=[])
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
        ),
    ):
        result = await safety_svc.summarize_safety(
            "CHEMBL999", "arthritis", DrugProfile(chembl_id="CHEMBL999"), []
        )

    assert result.regulatory_summary == ""
    assert result.pharmacovigilance_summary == ""
    assert result.literature_summary == ""
    assert result.safety_summary == ""
    assert result.safety_pmids == []
    assert result.safety_severity is None
    assert result.label_data_available is True


async def test_summarize_safety_handles_null_adverse_event_fields(safety_svc):
    """Regression: an OT AdverseEvent with null count / log_likelihood_ratio must not crash the
    prompt's format string (the ':.1f' would raise TypeError on None)."""
    from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning

    profile = DrugProfile(
        chembl_id="CHEMBL122",
        drug_warnings=[DrugWarning(warning_type="Black Box Warning")],
        adverse_events=[
            AdverseEvent(name="hepatotoxicity", count=None, log_likelihood_ratio=None)
        ],
    )
    with (
        patch.object(
            safety_svc, "_get_label_safety_records", new=AsyncMock(return_value=[])
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(
                side_effect=AssertionError("production must be deterministic")
            ),
        ),
    ):
        result = await safety_svc.summarize_safety(
            "CHEMBL122", "arthritis", profile, _SAFETY_ABSTRACTS
        )

    assert "warning type: Black Box Warning" in result.regulatory_summary
    assert result.pharmacovigilance_summary == ""
    assert "reports: 0" not in result.safety_summary
    assert "logLR: 0.0" not in result.safety_summary
    assert result.safety_severity == "black_box"


# --- classify_indication_harm (the validated concrete disease-specific question) ---


async def test_classify_indication_harm_parses_true(safety_svc):
    """A confirmed per-PMID harm with a source quote is aggregated."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": "patients",
                    "adverse_outcome": "increased cardiovascular thrombotic events",
                    "evidence_quote": (
                        "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                    ),
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await safety_svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is True
    assert summary == (
        "Disease-scoped literature for rofecoxib in colorectal cancer reported: "
        "increased cardiovascular thrombotic events — "
        '"Rofecoxib increased cardiovascular thrombotic events versus placebo." '
        "(PMID: 11696466)."
    )
    assert pmids == ["11696466"]


async def test_classify_indication_harm_false_clears_summary_and_pmids(safety_svc):
    """A fully reviewed safety-assessment-only paper is not a harm finding."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "safety_assessed_only",
                    "study_subjects": "patients",
                    "adverse_outcome": None,
                    "evidence_quote": None,
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await safety_svc.classify_indication_harm(
            "CHEMBL122", "migraine", _SAFETY_ABSTRACTS
        )

    assert harm is False
    assert summary == ""
    assert pmids == []


async def test_classify_indication_harm_empty_abstracts_no_llm(safety_svc):
    """No disease-scoped abstracts leaves the harm result unavailable."""
    with patch(
        "indication_scout.services.drug_safety.query_llm",
        new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
    ):
        harm, summary, pmids = await safety_svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", []
        )

    assert harm is None
    assert summary == ""
    assert pmids == []


async def test_classify_indication_harm_rejects_unverified_quote(safety_svc):
    """A claimed harm whose quote is absent from the source remains unavailable."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": "patients",
                    "adverse_outcome": "renal failure",
                    "evidence_quote": "Metformin caused renal failure.",
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(return_value=resp),
        ),
        patch("indication_scout.services.drug_safety.cache_set") as mock_cache_set,
    ):
        harm, summary, pmids = await safety_svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is None
    assert summary == ""
    assert pmids == []
    # The per-paper verdict is still cached: it records what the model read, and the verbatim check
    # that rejects it is deterministic and re-runs on every read. Only the harm itself is withheld.
    assert mock_cache_set.call_count == 1
    namespace, params, value, _cache_dir = mock_cache_set.call_args.args
    assert namespace == "indication_harm_verdict"
    assert params["pmid"] == "11696466"
    assert value["evidence_quote"] == "Metformin caused renal failure."


@pytest.mark.parametrize("subjects", ["animals", "cells_or_tissue", "unclear", None])
async def test_classify_indication_harm_rejects_non_patient_study(safety_svc, subjects):
    """A verified harm from non-human work is not patient risk for the indication."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": subjects,
                    "adverse_outcome": "increased cardiovascular thrombotic events",
                    "evidence_quote": (
                        "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                    ),
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.drug_safety.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.drug_safety.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await safety_svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is False
    assert summary == ""
    assert pmids == []
