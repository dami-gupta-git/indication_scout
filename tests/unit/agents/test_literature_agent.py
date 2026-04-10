"""Unit tests for run_literature_agent output assembly.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_literature_agent correctly
extracts artifacts and the narrative summary into a LiteratureOutput.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from indication_scout.agents.literature.literature_agent import run_literature_agent
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

SEARCH_TERMS = ["metformin colorectal cancer", "AMPK colon neoplasm"]
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
    study_types=["RCT"],
    key_findings=["Metformin reduces tumor growth"],
    has_adverse_effects=False,
    supporting_pmids=["111", "222"],
    summary="Moderate evidence supports metformin in colorectal cancer.",
)
NARRATIVE = "Metformin shows moderate evidence for colorectal cancer repurposing based on 2 RCTs."


def _make_agent(messages: list) -> MagicMock:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": messages})
    return agent


def _tool_msg(name: str, artifact) -> ToolMessage:
    return ToolMessage(
        content=f"result of {name}",
        artifact=artifact,
        name=name,
        tool_call_id=f"id_{name}",
    )


# ------------------------------------------------------------------
# Full happy path
# ------------------------------------------------------------------


async def test_run_literature_agent_assembles_all_fields():
    """run_literature_agent extracts all tool artifacts and the narrative into LiteratureOutput."""
    messages = [
        HumanMessage(content="Analyze metformin in colorectal cancer"),
        _tool_msg("expand_search_terms", SEARCH_TERMS),
        _tool_msg("fetch_and_cache", PMIDS),
        _tool_msg("semantic_search", SEMANTIC_RESULTS),
        _tool_msg("synthesize", EVIDENCE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_literature_agent(agent, "metformin", "colorectal cancer")

    assert isinstance(output, LiteratureOutput)
    assert output.search_results == SEARCH_TERMS
    assert output.pmids == PMIDS
    assert len(output.semantic_search_results) == 2
    assert output.semantic_search_results[0].pmid == "111"
    assert output.semantic_search_results[0].title == "Metformin and CRC"
    assert output.semantic_search_results[0].abstract == "Study A."
    assert output.semantic_search_results[0].similarity == 0.91
    assert output.semantic_search_results[1].pmid == "222"
    assert output.semantic_search_results[1].title == "AMPK pathway"
    assert output.semantic_search_results[1].abstract == "Study B."
    assert output.semantic_search_results[1].similarity == 0.85
    assert isinstance(output.evidence_summary, EvidenceSummary)
    assert output.evidence_summary.strength == "moderate"
    assert output.evidence_summary.study_count == 2
    assert output.evidence_summary.study_types == ["RCT"]
    assert output.evidence_summary.key_findings == ["Metformin reduces tumor growth"]
    assert output.evidence_summary.has_adverse_effects is False
    assert output.evidence_summary.supporting_pmids == ["111", "222"]
    assert output.summary == NARRATIVE


# ------------------------------------------------------------------
# Summary extraction: comes from finalize_analysis artifact
# ------------------------------------------------------------------


async def test_run_literature_agent_summary_from_finalize_analysis():
    """The narrative summary is taken from the finalize_analysis ToolMessage artifact."""
    final_narrative = "Final summary from finalize_analysis."
    messages = [
        HumanMessage(content="Analyze metformin in colorectal cancer"),
        _tool_msg("expand_search_terms", SEARCH_TERMS),
        _tool_msg("fetch_and_cache", PMIDS),
        _tool_msg("finalize_analysis", final_narrative),
    ]
    agent = _make_agent(messages)

    output = await run_literature_agent(agent, "metformin", "colorectal cancer")

    assert output.summary == final_narrative


# ------------------------------------------------------------------
# Partial runs — missing tools leave defaults
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "present_tools,missing_field",
    [
        (["fetch_and_cache", "semantic_search", "synthesize"], "search_results"),
        (["expand_search_terms", "semantic_search", "synthesize"], "pmids"),
        (
            ["expand_search_terms", "fetch_and_cache", "synthesize"],
            "semantic_search_results",
        ),
        (
            ["expand_search_terms", "fetch_and_cache", "semantic_search"],
            "evidence_summary",
        ),
    ],
)
async def test_run_literature_agent_missing_tool_leaves_default(
    present_tools, missing_field
):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    artifact_map = {
        "expand_search_terms": SEARCH_TERMS,
        "fetch_and_cache": PMIDS,
        "semantic_search": SEMANTIC_RESULTS,
        "synthesize": EVIDENCE,
    }
    messages = [HumanMessage(content="Analyze metformin in colorectal cancer")]
    for name in present_tools:
        messages.append(_tool_msg(name, artifact_map[name]))
    messages.append(_tool_msg("finalize_analysis", NARRATIVE))

    agent = _make_agent(messages)
    output = await run_literature_agent(agent, "metformin", "colorectal cancer")

    default_map = {
        "search_results": [],
        "pmids": [],
        "semantic_search_results": [],
        "evidence_summary": None,
    }
    assert getattr(output, missing_field) == default_map[missing_field]


# ------------------------------------------------------------------
# build_drug_profile ToolMessage is ignored (no field_map entry)
# ------------------------------------------------------------------


async def test_run_literature_agent_ignores_build_drug_profile_message():
    """build_drug_profile ToolMessages are correctly ignored — no field on LiteratureOutput."""
    from indication_scout.models.model_drug_profile import DrugProfile

    profile = DrugProfile(
        name="metformin",
        synonyms=[],
        target_gene_symbols=[],
        mechanisms_of_action=[],
        atc_codes=[],
        atc_descriptions=[],
        drug_type="Small molecule",
    )
    messages = [
        HumanMessage(content="Analyze metformin in colorectal cancer"),
        _tool_msg("build_drug_profile", profile),
        _tool_msg("expand_search_terms", SEARCH_TERMS),
        _tool_msg("fetch_and_cache", PMIDS),
        _tool_msg("semantic_search", SEMANTIC_RESULTS),
        _tool_msg("synthesize", EVIDENCE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_literature_agent(agent, "metformin", "colorectal cancer")

    assert output.search_results == SEARCH_TERMS
    assert output.pmids == PMIDS
    assert len(output.semantic_search_results) == 2
    assert isinstance(output.evidence_summary, EvidenceSummary)
    assert output.summary == NARRATIVE
