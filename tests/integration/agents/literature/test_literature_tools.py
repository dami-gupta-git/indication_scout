"""Integration tests for the literature agent's tools.

Each tool is invoked directly via tool.ainvoke({...}) — the LLM agent loop is bypassed so we can
assert each tool's content/artifact in isolation. Tools share a closure-scoped store dict, so
each test that exercises a downstream tool first runs the upstream tools that populate the store.

Hits real Anthropic, PubMed, Open Targets, and ChEMBL APIs.
Uses the test database (scout_test) via db_session_truncating.
"""

import itertools
import logging
from datetime import date

from langchain_core.messages.tool import ToolCall

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult, RetrievalService

logger = logging.getLogger(__name__)

# Counter to give each ToolCall a unique id within a test run
_tc_id = itertools.count()


def _tc(name: str, **args) -> ToolCall:
    """Build a ToolCall — required so .ainvoke() returns a ToolMessage with .artifact."""
    return ToolCall(name=name, args=args, id=f"tc{next(_tc_id)}", type="tool_call")


# ------------------------------------------------------------------
# Test target: Semaglutide + NASH, cutoff 2025-01-01
# Expected values reused from test_literature_agent.py (live run 2026-04-12)
# ------------------------------------------------------------------

_DRUG = "Semaglutide"
_DISEASE = "NASH"
_CUTOFF = date(2025, 1, 1)
_CHEMBL_ID = "CHEMBL2108724"

# PMIDs that must appear in fetch_and_cache output
_EXPECTED_PMIDS = {"39735270", "39412509"}

# Top-5 semantic search results (from test_literature_agent.py)
# Updated 2026-05-12 for pubtype-aware rerank — primary RCT readouts surface
# above reviews/preclinical. PMID 38847460 (Survodutide) is correctly excluded
# from _EXPECTED_SUPPORTING_PMIDS — it ranks in the top-5 as a NASH RCT but is
# not Semaglutide-specific evidence.
_EXPECTED_TOP5 = [
    ("36934740", "Semaglutide 2"),
    ("37328931", "Improved health-related quality of life with semaglutide"),
    ("33185364", "A Placebo-Controlled Trial of Subcutaneous Semaglutide"),
    ("38847460", "A Phase 2 Randomized Trial of Survodutide"),
    ("37646192", "Comparison of clinical efficacy and safety of weekly glucagon"),
]
_EXPECTED_SUPPORTING_PMIDS = {
    "33185364",
    "36934740",
    "37328931",
    "37646192",
}

# --- Values to fill in from a live run -----------------------------
# build_drug_profile expected values (Semaglutide / CHEMBL2108724)
_EXPECTED_TARGET_GENE_SYMBOLS: list[str] = []  # TODO: fill in, e.g. ["GLP1R"]
_EXPECTED_MECHANISMS_OF_ACTION: list[str] = []  # TODO: fill in
_EXPECTED_ATC_CODES: list[str] = []  # TODO: fill in, e.g. ["A10BJ06"]
_EXPECTED_ATC_DESCRIPTIONS: list[str] = []  # TODO: fill in
_EXPECTED_DRUG_TYPE: str = "Protein"  # TODO: fill in, e.g. "Protein"

# Synthesize expected values
_EXPECTED_STRENGTH: str = "moderate"  # confirmed by literature_agent test
_EXPECTED_MIN_STUDY_COUNT: int = 2  # confirmed by literature_agent test
# -------------------------------------------------------------------


def _tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}


def _build_tools(svc: RetrievalService, db):
    return build_literature_tools(svc, db, date_before=_CUTOFF)


async def test_build_drug_profile(db_session_truncating, test_cache_dir):
    """build_drug_profile resolves drug name → ChEMBL → DrugProfile with target/mechanism/ATC."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_DRUG)
    )

    profile: DrugProfile = msg.artifact
    assert isinstance(profile, DrugProfile)
    assert profile.chembl_id == _CHEMBL_ID
    assert profile.drug_type == _EXPECTED_DRUG_TYPE

    # content string reflects actual counts
    assert msg.content == (
        f"Profile for {_DRUG} ({_CHEMBL_ID}): "
        f"{len(profile.target_gene_symbols)} targets, "
        f"{len(profile.mechanisms_of_action)} mechanisms"
    )


async def test_expand_search_terms(db_session_truncating, test_cache_dir):
    """expand_search_terms returns deduplicated PubMed queries that mention the drug and disease."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    # build_drug_profile populates the store; expand_search_terms reads from it
    await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_DRUG)
    )
    msg = await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )

    queries: list[str] = msg.artifact
    assert isinstance(queries, list)
    assert all(isinstance(q, str) and q for q in queries)
    # case-insensitive dedup by retrieval.expand_search_terms
    assert len({q.lower().strip() for q in queries}) == len(queries)

    queries_lower = [q.lower() for q in queries]
    assert any("semaglutide" in q or "glp-1" in q or "glp1" in q for q in queries_lower)
    assert any(
        "nash" in q or "fatty liver" in q or "steatohepatitis" in q
        for q in queries_lower
    )

    assert msg.content == f"Generated {len(queries)} queries"


async def test_fetch_and_cache_without_queries(db_session_truncating, test_cache_dir):
    """fetch_and_cache short-circuits to an empty PMID list when expand_search_terms has not run."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["fetch_and_cache"].ainvoke(
        _tc("fetch_and_cache", drug_name=_DRUG)
    )

    assert msg.artifact == []
    assert msg.content == "No queries — call expand_search_terms first."


async def test_fetch_and_cache(db_session_truncating, test_cache_dir):
    """fetch_and_cache returns deduplicated PMIDs and persists abstracts to pgvector."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_DRUG)
    )
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    msg = await tools["fetch_and_cache"].ainvoke(
        _tc("fetch_and_cache", drug_name=_DRUG)
    )

    pmids: list[str] = msg.artifact
    assert isinstance(pmids, list)
    assert all(isinstance(p, str) and p.isdigit() for p in pmids)
    # dedup
    assert len(pmids) == len(set(pmids))

    assert _EXPECTED_PMIDS.issubset(set(pmids))
    assert msg.content == f"Fetched {len(pmids)} PMIDs"


async def test_semantic_search_without_pmids(db_session_truncating, test_cache_dir):
    """semantic_search short-circuits to an empty list when fetch_and_cache has not run."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )

    assert msg.artifact == []
    assert msg.content == "No PMIDs — call fetch_and_cache first."


async def test_semantic_search(db_session_truncating, test_cache_dir):
    """semantic_search returns top-k AbstractResults sorted by descending similarity."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_DRUG)
    )
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))
    msg = await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )

    results: list[AbstractResult] = msg.artifact
    assert len(results) == 5
    assert all(isinstance(r, AbstractResult) for r in results)

    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0.0 < s <= 1.0 for s in similarities)

    result_pmids = [r.pmid for r in results]
    for expected_pmid, expected_title_fragment in _EXPECTED_TOP5:
        assert (
            expected_pmid in result_pmids
        ), f"Expected PMID {expected_pmid} not in top-5"
        match = next(r for r in results if r.pmid == expected_pmid)
        assert expected_title_fragment in match.title
        assert isinstance(match.abstract, str) and len(match.abstract) > 0

    assert (
        msg.content
        == f"Found {len(results)} abstracts (top sim: {results[0].similarity:.2f})"
    )


async def test_synthesize(db_session_truncating, test_cache_dir):
    """synthesize turns the top abstracts into an EvidenceSummary with strength + supporting PMIDs."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_DRUG)
    )
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))
    await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )
    msg = await tools["synthesize"].ainvoke(
        _tc("synthesize", drug_name=_DRUG, disease_name=_DISEASE)
    )

    evidence: EvidenceSummary = msg.artifact
    assert isinstance(evidence, EvidenceSummary)
    assert evidence.strength == _EXPECTED_STRENGTH
    assert evidence.study_count >= _EXPECTED_MIN_STUDY_COUNT
    assert _EXPECTED_SUPPORTING_PMIDS.issubset(set(evidence.supporting_pmids))
    assert len(evidence.key_findings) >= 2

    assert msg.content == f"Evidence strength: {evidence.strength}"


async def test_finalize_analysis(db_session_truncating, test_cache_dir):
    """finalize_analysis echoes the summary string back as both content and artifact."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    summary_text = "Semaglutide shows weak preclinical evidence for NASH in the retrieved abstracts."
    msg = await tools["finalize_analysis"].ainvoke(
        _tc("finalize_analysis", summary=summary_text)
    )

    assert msg.artifact == summary_text
    assert msg.content == "Analysis complete."


# ------------------------------------------------------------------
# Cutoff plumbing — date_before passed to build_literature_tools must
# reach PubMed search so every returned PMID has a publication date
# strictly before the cutoff. Mirrors the cutoff contract verified for
# the clinical trials supervisor wrapper.
#
# Uses a query (semaglutide × MASH) with a sharp empirical cutoff edge:
# 119 hits today, ~1 hit before 2024-01-01, 0 before 2021-01-01 (verified
# live against eutils on 2026-05-02). A working cutoff yields a result
# set every member of which predates the cutoff; a broken cutoff lets
# the post-2024 publication explosion through.
# ------------------------------------------------------------------

_CUTOFF_DRUG_LIT = "semaglutide"
_CUTOFF_DISEASE_LIT = "MASH"
_LIT_CUTOFF = date(2024, 1, 1)


async def test_fetch_and_cache_respects_date_before(
    db_session_truncating, test_cache_dir
):
    """build_literature_tools(date_before=cutoff) must forward the cutoff to PubMed
    so every PMID returned by fetch_and_cache has a publication date strictly
    before the cutoff.

    Verifies the invariant by re-running the production date filter
    (PubMedClient._filter_pmids_by_date) over the result and demanding zero
    PMIDs are dropped — i.e. the cutoff was already correctly applied upstream.
    """
    from indication_scout.data_sources.pubmed import PubMedClient

    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(
        build_literature_tools(svc, db_session_truncating, date_before=_LIT_CUTOFF)
    )

    await tools["build_drug_profile"].ainvoke(
        _tc("build_drug_profile", drug_name=_CUTOFF_DRUG_LIT)
    )
    await tools["expand_search_terms"].ainvoke(
        _tc(
            "expand_search_terms",
            drug_name=_CUTOFF_DRUG_LIT,
            disease_name=_CUTOFF_DISEASE_LIT,
        )
    )
    msg = await tools["fetch_and_cache"].ainvoke(
        _tc("fetch_and_cache", drug_name=_CUTOFF_DRUG_LIT)
    )

    pmids: list[str] = msg.artifact
    assert isinstance(pmids, list)
    assert pmids, (
        f"fetch_and_cache returned no PMIDs for {_CUTOFF_DRUG_LIT} × {_CUTOFF_DISEASE_LIT} "
        f"with cutoff {_LIT_CUTOFF}; expected at least a handful of pre-2024 abstracts. "
        f"A no-result run trivially satisfies the cutoff invariant and is not a useful test."
    )

    # The production filter drops any PMID whose sortpubdate is on or after the
    # cutoff. If the cutoff was correctly applied during fetch_and_cache, running
    # the same filter again must drop nothing.
    async with PubMedClient(cache_dir=test_cache_dir) as client:
        kept = await client._filter_pmids_by_date(pmids, _LIT_CUTOFF)

    leaked = sorted(set(pmids) - set(kept))
    assert not leaked, (
        f"fetch_and_cache returned {len(leaked)} PMID(s) whose sortpubdate is on or "
        f"after the cutoff {_LIT_CUTOFF.isoformat()} — cutoff was not applied at the "
        f"PubMed search layer. Leaked PMIDs: {leaked[:10]}"
    )


# ------------------------------------------------------------------
# Cutoff-shifting test — the cutoff must MATERIALLY change the result
# set, not just satisfy "every returned PMID < cutoff" trivially.
#
# The single-cutoff test above can pass if PubMed quietly ignores the
# date filter, since the LLM's query expansion tends to surface mostly
# recent papers anyway. To prove the filter actually fires, run the
# same drug × disease through fetch_and_cache twice with two cutoffs
# (2021 vs 2024) and assert:
#   - Both runs respect their own cutoff (no PMID leaks past it).
#   - The later cutoff returns at least one PMID published in the
#     [early_cutoff, late_cutoff) window — proving the cutoff actually
#     narrows the result set rather than being silently ignored.
#
# We do NOT assert strict-subset (early ⊂ late). PubMed's eutils caps
# results at `pubmed_max_results` per query, ranked by relevance — so a
# wider date window crowds out older papers from the top-N cap, and the
# 2021 set can legitimately contain papers the 2024 set didn't surface.
# That's a property of paginated ranked results, not a leak.
#
# Empirical edge for semaglutide × MASH (verified live 2026-05-02 on
# eutils): 0 hits before 2021-01-01, 1 hit before 2024-01-01, 119 today.
# So the late set should contain a small handful of [2021, 2024) papers.
# ------------------------------------------------------------------


_EARLY_CUTOFF = date(2021, 1, 1)
_LATE_CUTOFF = date(2024, 1, 1)
# Cap each PubMed query at a small N so this test runs in ~1 minute
# instead of ~4. Real holdouts use ~200; 20 still exercises the full
# pipeline (search → date filter → fetch → embed → store).
_TEST_PUBMED_MAX_RESULTS = 20


async def test_fetch_and_cache_cutoff_shift_narrows_result_set(
    db_session_truncating, test_cache_dir, monkeypatch
):
    """Run fetch_and_cache twice on the same drug × disease with two
    different cutoffs. Each run must respect its own cutoff, and the
    later cutoff must surface at least one PMID dated in the window
    between the two cutoffs — proving the filter actually fires.
    """
    from indication_scout.data_sources.pubmed import PubMedClient
    from indication_scout.services import retrieval as retrieval_module

    # Cap PubMed results per query so the test stays fast. The cutoff
    # behavior we want to verify is independent of N. Settings is a frozen
    # pydantic model, so swap the module-level reference rather than
    # mutating attributes in place.
    test_settings = retrieval_module._settings.model_copy(
        update={"pubmed_max_results": _TEST_PUBMED_MAX_RESULTS}
    )
    monkeypatch.setattr(retrieval_module, "_settings", test_settings)

    svc = RetrievalService(test_cache_dir)

    async def _run(cutoff: date) -> list[str]:
        tools = _tool_map(
            build_literature_tools(svc, db_session_truncating, date_before=cutoff)
        )
        await tools["build_drug_profile"].ainvoke(
            _tc("build_drug_profile", drug_name=_CUTOFF_DRUG_LIT)
        )
        await tools["expand_search_terms"].ainvoke(
            _tc(
                "expand_search_terms",
                drug_name=_CUTOFF_DRUG_LIT,
                disease_name=_CUTOFF_DISEASE_LIT,
            )
        )
        msg = await tools["fetch_and_cache"].ainvoke(
            _tc("fetch_and_cache", drug_name=_CUTOFF_DRUG_LIT)
        )
        return msg.artifact

    early_pmids = await _run(_EARLY_CUTOFF)
    late_pmids = await _run(_LATE_CUTOFF)

    early_set = set(early_pmids)
    late_set = set(late_pmids)

    # Each result must respect its own cutoff (re-run the production filter).
    async with PubMedClient(cache_dir=test_cache_dir) as client:
        early_kept = set(await client._filter_pmids_by_date(early_pmids, _EARLY_CUTOFF))
        late_kept = set(await client._filter_pmids_by_date(late_pmids, _LATE_CUTOFF))

    early_leaked = sorted(early_set - early_kept)
    late_leaked = sorted(late_set - late_kept)
    assert not early_leaked, (
        f"{len(early_leaked)} PMID(s) leaked past the {_EARLY_CUTOFF} cutoff: "
        f"{early_leaked[:10]}"
    )
    assert not late_leaked, (
        f"{len(late_leaked)} PMID(s) leaked past the {_LATE_CUTOFF} cutoff: "
        f"{late_leaked[:10]}"
    )

    # The cutoff must actually narrow the result set. The late run must
    # surface at least one PMID dated in [early_cutoff, late_cutoff) that
    # the early run could not. Verify by re-filtering the late set against
    # the EARLY cutoff: anything that survives the late filter but NOT the
    # early one is a paper published in the window between the two cutoffs.
    async with PubMedClient(cache_dir=test_cache_dir) as client:
        late_under_early = set(
            await client._filter_pmids_by_date(late_pmids, _EARLY_CUTOFF)
        )
    in_window = late_set - late_under_early
    assert in_window, (
        f"{_LATE_CUTOFF} cutoff returned no PMIDs in the window "
        f"[{_EARLY_CUTOFF}, {_LATE_CUTOFF}) for {_CUTOFF_DRUG_LIT} × "
        f"{_CUTOFF_DISEASE_LIT}. Either no such literature exists (pick a "
        f"different test pair) or the cutoff is not being applied — both "
        f"runs returned only pre-{_EARLY_CUTOFF} papers."
    )
