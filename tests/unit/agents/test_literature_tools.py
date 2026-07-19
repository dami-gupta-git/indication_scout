"""Unit tests for literature_tools.build_literature_tools."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolCall

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

CHEMBL_ID = "CHEMBL1431"

DRUG_PROFILE = DrugProfile(
    chembl_id=CHEMBL_ID,
    target_gene_symbols=["PRKAA1"],
    mechanisms_of_action=["AMP-activated protein kinase activator"],
    atc_codes=["A10BA02"],
    atc_descriptions=["Biguanides"],
    drug_type="Small molecule",
)

SEARCH_TERMS = [
    "metformin colorectal cancer",
    "metformin colon neoplasm AMPK",
    "biguanide colorectal carcinoma",
]

PMIDS = ["111", "222", "333"]

SEMANTIC_RESULTS = [
    AbstractResult(
        pmid="111", title="Metformin and CRC", abstract="Study A.", similarity=0.91
    ),
    AbstractResult(
        pmid="222", title="AMPK pathway", abstract="Study B.", similarity=0.85
    ),
]

EVIDENCE = EvidenceSummary(
    strength="moderate",
    study_count=2,
    key_findings=["Metformin reduces tumor growth"],
    supporting_pmids=["111", "222"],
    summary="Moderate evidence supports metformin in colorectal cancer based on 2 RCTs.",
)

SAFETY_ABSTRACTS = [
    AbstractResult(
        pmid="111", title="Metformin lactic acidosis risk", abstract="Study C.", similarity=0.7
    ),
]

SAFETY_SUMMARY = "Metformin carries a lactic acidosis risk in renal impairment (PMID: 111)."
SAFETY_PMIDS = ["111"]
SAFETY_SEVERITY = "serious"
HARM_SUMMARY = "Lactic acidosis reported in T2D patients (PMID: 111)."
HARM_PMIDS = ["111"]


def _make_svc() -> MagicMock:
    svc = MagicMock()
    svc.cache_dir = Path("/tmp/test_cache")
    svc.build_drug_profile = AsyncMock(return_value=DRUG_PROFILE)
    svc.expand_search_terms = AsyncMock(return_value=SEARCH_TERMS)
    svc.fetch_and_cache = AsyncMock(return_value=PMIDS)
    svc.semantic_search = AsyncMock(return_value=SEMANTIC_RESULTS)
    svc.safety_search = AsyncMock(return_value=SAFETY_ABSTRACTS)
    svc.summarize_safety = AsyncMock(
        return_value=(SAFETY_SUMMARY, SAFETY_PMIDS, SAFETY_SEVERITY)
    )
    svc.classify_indication_harm = AsyncMock(
        return_value=(True, HARM_SUMMARY, HARM_PMIDS)
    )
    # Fresh copy per call — the synthesize tool mutates the returned EvidenceSummary (merges in the
    # safety fields), so a shared module-level constant would leak state across tests.
    svc.synthesize = AsyncMock(side_effect=lambda *a, **k: EVIDENCE.model_copy())
    return svc


def _patch_resolve():
    return patch(
        "indication_scout.agents.literature.literature_tools.resolve_drug_name",
        new=AsyncMock(return_value=CHEMBL_ID),
    )


def _build(svc, **kwargs):
    tools = build_literature_tools(svc=svc, db=MagicMock(), **kwargs)
    return {t.name: t for t in tools}


# ------------------------------------------------------------------
# build_drug_profile
# ------------------------------------------------------------------


async def test_build_drug_profile_calls_svc_and_returns_artifact():
    """build_drug_profile resolves drug_name → chembl_id and returns the DrugProfile as artifact."""
    svc = _make_svc()
    tool_map = _build(svc)

    with _patch_resolve():
        msg = await tool_map["build_drug_profile"].ainvoke(
            ToolCall(
                name="build_drug_profile",
                args={"drug_name": "metformin"},
                id="tc0",
                type="tool_call",
            )
        )

    svc.build_drug_profile.assert_awaited_once_with(CHEMBL_ID)
    assert isinstance(msg.artifact, DrugProfile)
    assert msg.artifact.chembl_id == CHEMBL_ID
    assert msg.artifact.target_gene_symbols == ["PRKAA1"]
    assert "metformin" in msg.content
    assert CHEMBL_ID in msg.content


# ------------------------------------------------------------------
# expand_search_terms
# ------------------------------------------------------------------


async def test_expand_search_terms_uses_profile_from_store():
    """expand_search_terms reads drug_profile from store and does not call build_drug_profile."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    # Inject profile into the shared store via build_drug_profile side-effect
    # by calling build_drug_profile first so the store is populated
    with _patch_resolve():
        await tool_map["build_drug_profile"].ainvoke(
            ToolCall(
                name="build_drug_profile",
                args={"drug_name": "metformin"},
                id="tc_pre",
                type="tool_call",
            )
        )
        svc.build_drug_profile.reset_mock()

        msg = await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc1",
                type="tool_call",
            )
        )

    svc.build_drug_profile.assert_not_awaited()
    svc.expand_search_terms.assert_awaited_once_with(
        CHEMBL_ID, "colorectal cancer", DRUG_PROFILE
    )
    assert msg.artifact == SEARCH_TERMS
    assert "Generated 3 queries" in msg.content


async def test_expand_search_terms_builds_profile_when_store_empty():
    """expand_search_terms calls svc.build_drug_profile when no profile is in the store."""
    svc = _make_svc()
    tool_map = _build(svc)

    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc2",
                type="tool_call",
            )
        )

    svc.build_drug_profile.assert_awaited_once_with(CHEMBL_ID)


# ------------------------------------------------------------------
# fetch_and_cache
# ------------------------------------------------------------------


async def test_fetch_and_cache_reads_queries_from_store_and_returns_pmids():
    """fetch_and_cache reads queries written by expand_search_terms and returns PMIDs."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    # Populate store via expand_search_terms
    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre",
                type="tool_call",
            )
        )

    msg = await tool_map["fetch_and_cache"].ainvoke(
        ToolCall(
            name="fetch_and_cache",
            args={"drug_name": "metformin"},
            id="tc3",
            type="tool_call",
        )
    )

    svc.fetch_and_cache.assert_awaited_once()
    assert svc.fetch_and_cache.call_args.args[0] == SEARCH_TERMS
    assert msg.artifact == PMIDS
    assert "Fetched 3 PMIDs" in msg.content


async def test_fetch_and_cache_returns_empty_when_no_queries():
    """fetch_and_cache returns early with an empty list and informative message when store has no queries."""
    svc = _make_svc()
    tool_map = _build(svc)

    msg = await tool_map["fetch_and_cache"].ainvoke(
        ToolCall(
            name="fetch_and_cache",
            args={"drug_name": "metformin"},
            id="tc4",
            type="tool_call",
        )
    )

    svc.fetch_and_cache.assert_not_awaited()
    assert msg.artifact == []
    assert "expand_search_terms" in msg.content


async def test_fetch_and_cache_passes_date_before():
    """fetch_and_cache forwards date_before from closure to svc."""
    from datetime import date

    svc = _make_svc()
    cutoff = date(2020, 1, 1)
    tools = build_literature_tools(svc=svc, db=MagicMock(), date_before=cutoff)
    tool_map = {t.name: t for t in tools}
    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre",
                type="tool_call",
            )
        )

    await tool_map["fetch_and_cache"].ainvoke(
        ToolCall(
            name="fetch_and_cache",
            args={"drug_name": "metformin"},
            id="tc5",
            type="tool_call",
        )
    )

    assert svc.fetch_and_cache.call_args.kwargs["date_before"] == cutoff


# ------------------------------------------------------------------
# semantic_search
# ------------------------------------------------------------------


async def test_semantic_search_reads_pmids_from_store_and_returns_results():
    """semantic_search reads PMIDs written by fetch_and_cache and returns ranked AbstractResults."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    # Populate store
    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre1",
                type="tool_call",
            )
        )
        await tool_map["fetch_and_cache"].ainvoke(
            ToolCall(
                name="fetch_and_cache",
                args={"drug_name": "metformin"},
                id="tc_pre2",
                type="tool_call",
            )
        )

        msg = await tool_map["semantic_search"].ainvoke(
            ToolCall(
                name="semantic_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc6",
                type="tool_call",
            )
        )

    svc.semantic_search.assert_awaited_once()
    call = svc.semantic_search.call_args
    assert call.args[1] == CHEMBL_ID
    assert call.args[2] == PMIDS

    assert len(msg.artifact) == 2
    assert msg.artifact[0].pmid == "111"
    assert msg.artifact[0].title == "Metformin and CRC"
    assert msg.artifact[0].abstract == "Study A."
    assert msg.artifact[0].similarity == 0.91
    assert msg.artifact[1].pmid == "222"
    assert msg.artifact[1].title == "AMPK pathway"
    assert msg.artifact[1].abstract == "Study B."
    assert msg.artifact[1].similarity == 0.85
    assert "0.91" in msg.content


async def test_semantic_search_returns_empty_when_no_pmids():
    """semantic_search returns early with empty list when store has no PMIDs."""
    svc = _make_svc()
    tool_map = _build(svc)

    with _patch_resolve():
        msg = await tool_map["semantic_search"].ainvoke(
            ToolCall(
                name="semantic_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc7",
                type="tool_call",
            )
        )

    svc.semantic_search.assert_not_awaited()
    assert msg.artifact == []
    assert "fetch_and_cache" in msg.content


# ------------------------------------------------------------------
# safety_search
# ------------------------------------------------------------------


async def test_safety_search_fetches_summarizes_and_classifies():
    """safety_search: builds/uses the drug profile, calls svc.safety_search(chembl_id, date_before,
    disease), svc.summarize_safety (drug-level 3-tuple), and svc.classify_indication_harm
    (disease-specific), returning an EvidenceSummary carrying BOTH."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    with _patch_resolve():
        await tool_map["build_drug_profile"].ainvoke(
            ToolCall(
                name="build_drug_profile",
                args={"drug_name": "metformin"},
                id="tc_pre",
                type="tool_call",
            )
        )
        svc.build_drug_profile.reset_mock()

        msg = await tool_map["safety_search"].ainvoke(
            ToolCall(
                name="safety_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_safety1",
                type="tool_call",
            )
        )

    svc.build_drug_profile.assert_not_awaited()  # profile came from the store
    svc.safety_search.assert_awaited_once_with(
        CHEMBL_ID, date_before=None, disease="colorectal cancer"
    )
    svc.summarize_safety.assert_awaited_once_with(
        CHEMBL_ID, "colorectal cancer", DRUG_PROFILE, SAFETY_ABSTRACTS, date_before=None
    )
    svc.classify_indication_harm.assert_awaited_once_with(
        CHEMBL_ID, "colorectal cancer", SAFETY_ABSTRACTS
    )

    es = msg.artifact
    assert isinstance(es, EvidenceSummary)
    # Drug-level fields.
    assert es.safety_summary == SAFETY_SUMMARY
    assert es.safety_pmids == SAFETY_PMIDS
    assert es.safety_severity == SAFETY_SEVERITY
    # Disease-specific fields.
    assert es.indication_harm is True
    assert es.indication_harm_summary == HARM_SUMMARY
    assert es.indication_harm_pmids == HARM_PMIDS
    assert "Safety signal" in msg.content
    assert "Indication-specific harm" in msg.content


async def test_safety_search_builds_profile_when_store_empty():
    """safety_search builds the drug profile on the fly when the store has none."""
    svc = _make_svc()
    tool_map = _build(svc)

    with _patch_resolve():
        msg = await tool_map["safety_search"].ainvoke(
            ToolCall(
                name="safety_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_safety2",
                type="tool_call",
            )
        )

    svc.build_drug_profile.assert_awaited_once_with(CHEMBL_ID)
    svc.safety_search.assert_awaited_once_with(
        CHEMBL_ID, date_before=None, disease="colorectal cancer"
    )
    assert msg.artifact.safety_summary == SAFETY_SUMMARY


async def test_safety_search_no_signal_content_is_explicit():
    """No drug-level signal AND no indication harm → content says so explicitly."""
    svc = _make_svc()
    svc.summarize_safety = AsyncMock(return_value=("", [], "none"))
    svc.classify_indication_harm = AsyncMock(return_value=(False, "", []))
    tool_map = _build(svc)
    with _patch_resolve():
        msg = await tool_map["safety_search"].ainvoke(
            ToolCall(
                name="safety_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_safety3",
                type="tool_call",
            )
        )

    assert msg.artifact.safety_summary == ""
    assert msg.artifact.safety_pmids == []
    assert msg.artifact.indication_harm is False
    assert "No safety signal found" in msg.content


# ------------------------------------------------------------------
# finalize_analysis
# ------------------------------------------------------------------


async def test_finalize_analysis_stores_summary_and_returns_artifact():
    """finalize_analysis returns the summary string as artifact and echoes it in content."""
    svc = _make_svc()
    tool_map = _build(svc)

    text = "Metformin shows moderate evidence in colorectal cancer based on 2 RCTs."
    msg = await tool_map["finalize_analysis"].ainvoke(
        ToolCall(
            name="finalize_analysis",
            args={"summary": text},
            id="tc_fin",
            type="tool_call",
        )
    )

    assert msg.artifact == text
    assert "Analysis complete" in msg.content


# ------------------------------------------------------------------
# synthesize
# ------------------------------------------------------------------


async def test_synthesize_reads_abstracts_from_store_and_returns_evidence():
    """synthesize reads abstracts written by semantic_search, merges in safety_search's result
    (called first in the chain), and returns one EvidenceSummary carrying both."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    # Populate store through the full chain, including safety_search
    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre1",
                type="tool_call",
            )
        )
        await tool_map["fetch_and_cache"].ainvoke(
            ToolCall(
                name="fetch_and_cache",
                args={"drug_name": "metformin"},
                id="tc_pre2",
                type="tool_call",
            )
        )
        await tool_map["semantic_search"].ainvoke(
            ToolCall(
                name="semantic_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre3",
                type="tool_call",
            )
        )
        await tool_map["safety_search"].ainvoke(
            ToolCall(
                name="safety_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre4",
                type="tool_call",
            )
        )

        msg = await tool_map["synthesize"].ainvoke(
            ToolCall(
                name="synthesize",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc8",
                type="tool_call",
            )
        )

    svc.synthesize.assert_awaited_once_with(
        CHEMBL_ID, "colorectal cancer", SEMANTIC_RESULTS, approved_indications=None
    )
    assert isinstance(msg.artifact, EvidenceSummary)
    assert msg.artifact.strength == "moderate"
    assert msg.artifact.study_count == 2
    assert msg.artifact.key_findings == ["Metformin reduces tumor growth"]
    assert msg.artifact.supporting_pmids == ["111", "222"]
    assert (
        msg.artifact.summary
        == "Moderate evidence supports metformin in colorectal cancer based on 2 RCTs."
    )
    assert msg.artifact.safety_summary == SAFETY_SUMMARY
    assert msg.artifact.safety_pmids == SAFETY_PMIDS
    assert msg.artifact.indication_harm is True
    assert msg.artifact.indication_harm_summary == HARM_SUMMARY
    assert "moderate" in msg.content


async def test_synthesize_defaults_safety_fields_when_safety_search_skipped():
    """If the ReAct agent skips safety_search (should not happen — it's a REQUIRED step per
    literature.txt — but must not crash), synthesize's EvidenceSummary defaults to ""/[] rather
    than fabricating a safety verdict."""
    svc = _make_svc()
    tools = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    with _patch_resolve():
        await tool_map["expand_search_terms"].ainvoke(
            ToolCall(
                name="expand_search_terms",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre1",
                type="tool_call",
            )
        )
        await tool_map["fetch_and_cache"].ainvoke(
            ToolCall(
                name="fetch_and_cache",
                args={"drug_name": "metformin"},
                id="tc_pre2",
                type="tool_call",
            )
        )
        await tool_map["semantic_search"].ainvoke(
            ToolCall(
                name="semantic_search",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc_pre3",
                type="tool_call",
            )
        )
        # safety_search deliberately NOT called here.

        msg = await tool_map["synthesize"].ainvoke(
            ToolCall(
                name="synthesize",
                args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
                id="tc9",
                type="tool_call",
            )
        )

    assert msg.artifact.safety_summary == ""
    assert msg.artifact.safety_pmids == []
    assert msg.artifact.indication_harm is False
    assert msg.artifact.indication_harm_pmids == []
