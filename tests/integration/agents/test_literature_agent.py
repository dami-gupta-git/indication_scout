"""Integration tests for the literature agent.

Hits real Anthropic, PubMed, Open Targets, and ChEMBL APIs.
Uses the test database (scout_test) via db_session_truncating.
"""

import logging
from datetime import date

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Semaglutide + NASH, cutoff 2025-01-01
#
# PMIDs and similarity scores verified by a live run on 2026-04-06.
# ------------------------------------------------------------------

_CUTOFF = date(2025, 1, 1)

# PMIDs that must appear in fetch_and_cache output (stable, pre-cutoff papers)
_EXPECTED_PMIDS = {
    "38701096",  # Semaglutide + empagliflozin RCT protocol in NAFLD+T2DM
    "39223865",  # Oral semaglutide 48-week observational in MASLD+T2DM
    "39286709",  # Mechanisms review: semaglutide in NAFLD/NASH
    "39663847",  # Survodutide vs GLP-1RAs in MASH
}

# PMIDs post-cutoff that must not appear
_EXCLUDED_PMIDS = {
    "40000000",  # placeholder — any future PMID well above the cutoff window
}

# Top-5 semantic search results verified by live run on 2026-04-06
_EXPECTED_TOP5 = [
    ("38701096", "Semaglutide combined with empagliflozin"),
    ("39223865", "Beneficial effect of oral semaglutide"),
    ("39286709", "Mechanisms of Non-alcoholic Fatty Liver Disease"),
    ("39385875", "Semaglutide Versus Other Glucagon-Like Peptide-1 Agonists"),
    ("39663847", "Survodutide in MASH"),
]

# Evidence summary fields verified by live run on 2026-04-06
_EXPECTED_SUPPORTING_PMIDS = {"38701096", "39223865", "39286709", "39663847"}


async def test_semaglutide_nash_literature_agent(db_session_truncating, test_cache_dir):
    """End-to-end: literature agent produces correct LiteratureOutput for Semaglutide + NASH.

    Verifies:
    - search_results are non-empty keyword queries
    - known PMIDs are present in fetch_and_cache output
    - top-5 semantic results match expected PMIDs and title prefixes
    - evidence_summary is moderate with correct supporting PMIDs
    - narrative summary is non-empty
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    svc = RetrievalService(test_cache_dir)
    agent = build_literature_agent(
        llm,
        svc,
        db_session_truncating,
        date_before=_CUTOFF,
        max_search_results=20,
        num_top_k=5,
    )

    output = await run_literature_agent(agent, "Semaglutide", "NASH")

    assert isinstance(output, LiteratureOutput)

    # --- search_results ---
    assert len(output.search_results) >= 5
    queries_lower = [q.lower() for q in output.search_results]
    assert any("semaglutide" in q or "glp-1" in q or "glp1" in q for q in queries_lower)
    assert any(
        "nash" in q or "fatty liver" in q or "steatohepatitis" in q
        for q in queries_lower
    )

    # --- pmids ---
    assert len(output.pmids) >= 20
    assert _EXPECTED_PMIDS.issubset(set(output.pmids))

    # --- semantic_search_results ---
    assert len(output.semantic_search_results) == 5
    result_pmids = [r.pmid for r in output.semantic_search_results]
    for expected_pmid, expected_title_fragment in _EXPECTED_TOP5:
        assert (
            expected_pmid in result_pmids
        ), f"Expected PMID {expected_pmid} not in top-5"
        match = next(
            r for r in output.semantic_search_results if r.pmid == expected_pmid
        )
        assert (
            expected_title_fragment in match.title
        ), f"PMID {expected_pmid}: expected title fragment '{expected_title_fragment}', got '{match.title}'"
        assert isinstance(match.abstract, str) and len(match.abstract) > 0
        assert 0.0 < match.similarity <= 1.0

    similarities = [r.similarity for r in output.semantic_search_results]
    assert similarities == sorted(similarities, reverse=True)

    # --- evidence_summary ---
    assert isinstance(output.evidence_summary, EvidenceSummary)
    assert output.evidence_summary.strength == "moderate"
    assert output.evidence_summary.study_count >= 2
    assert output.evidence_summary.has_adverse_effects
    assert _EXPECTED_SUPPORTING_PMIDS.issubset(
        set(output.evidence_summary.supporting_pmids)
    )
    assert len(output.evidence_summary.key_findings) >= 2

    # --- summary ---
    assert len(output.summary) > 100
