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
# PMIDs and similarity scores verified by a live run on 2026-04-12.
# ------------------------------------------------------------------

_CUTOFF = date(2025, 1, 1)

# PMIDs that must appear in fetch_and_cache output (stable, pre-cutoff papers)
_EXPECTED_PMIDS = {"39735270", "39412509"}

# PMIDs post-cutoff that must not appear
_EXCLUDED_PMIDS = {
    "40000000",  # placeholder — any future PMID well above the cutoff window
}

# Top-5 semantic search results verified by live run on 2026-05-12
# (post pubtype-aware rerank — primary RCT readouts now surface above reviews
# and preclinical papers; PMID 38847460 is a survodutide comparator trial that
# ranks in for NASH/RCT but is correctly not in _EXPECTED_CITED_PMIDS).
_EXPECTED_TOP5 = [
    ("36934740", "Semaglutide 2"),
    ("37328931", "Improved health-related quality of life with semaglutide"),
    ("33185364", "A Placebo-Controlled Trial of Subcutaneous Semaglutide"),
    ("38847460", "A Phase 2 Randomized Trial of Survodutide"),
    ("37646192", "Comparison of clinical efficacy and safety of weekly glucagon"),
]

# Core NASH RCTs that must be CITED in the synthesis (in supporting OR contradicting — which
# list a given trial lands in is LLM judgment that varies run-to-run: e.g. 36934740 is the
# cirrhosis trial that FAILED its endpoint, so it may be classified supporting [drug studied]
# or contradicting [drug failed]). We only assert the key trials were surfaced, not their
# list membership. 33185364 (the positive NEJM NASH-resolution RCT) is the one unambiguous
# supporter.
_EXPECTED_CITED_PMIDS = {
    "33185364",
    "36934740",
    "37328931",
}


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
    # Drug-specific semaglutide NASH RCTs → drug_specific basis, strong evidence (multiple
    # phase 2/3 RCTs). strength/direction/evidence_basis come from the combined synthesize call
    # (the single author over the abstracts).
    assert output.evidence_summary.evidence_basis == "drug_specific"
    assert output.evidence_summary.strength in {"strong", "moderate"}
    assert output.evidence_summary.study_count >= 2
    # The core NASH RCTs must be CITED — in supporting, contradicting, OR neutral (list membership
    # is LLM-variable; see _EXPECTED_CITED_PMIDS; e.g. 37328931 is a quality-of-life readout that
    # pmid_direction may set neutral). 33185364 (positive NEJM RCT) must support.
    cited = (
        set(output.evidence_summary.supporting_pmids)
        | set(output.evidence_summary.contradicting_pmids)
        | set(output.evidence_summary.neutral_pmids)
    )
    assert _EXPECTED_CITED_PMIDS.issubset(cited)
    assert "33185364" in output.evidence_summary.supporting_pmids
    assert len(output.evidence_summary.key_findings) >= 2

    # --- summary ---
    # The agent's summary is a free-text status line whose exact wording varies run-to-run
    # ("Strong evidence retrieved...", "Evidence synthesis complete...", etc.) — only assert it
    # is non-empty. The authoritative grade is checked above via evidence_summary.strength.
    assert output.summary.strip()

async def test_random_literature_agent(db_session_truncating, test_cache_dir):

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    svc = RetrievalService(test_cache_dir)
    agent = build_literature_agent(
        llm,
        svc,
        db_session_truncating,
        date_before=_CUTOFF,
    )
    test_cache_dir="abc"
    output = await run_literature_agent(agent, "Rofecoxib", "arthritis")

    assert isinstance(output, LiteratureOutput)