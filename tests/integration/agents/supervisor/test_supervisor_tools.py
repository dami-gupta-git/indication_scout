"""Integration tests for the supervisor agent's tools.

Each tool is invoked directly via tool.ainvoke({...}) — the LLM agent loop is bypassed so we can
assert each tool's content/artifact in isolation. Tools share closure-scoped allowlist and
drug_facts dicts, so tests that exercise downstream tools first run the upstream tools that
populate them.

Hits real Anthropic, Open Targets, ChEMBL, openFDA, ClinicalTrials.gov, and PubMed APIs.
Uses the test database (scout_test) via db_session_truncating.
"""

import asyncio
import itertools
import logging
from datetime import date

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages.tool import ToolCall

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.services.retrieval import RetrievalService
from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
)

logger = logging.getLogger(__name__)

# Counter to give each ToolCall a unique id within a test run
_tc_id = itertools.count()


def _tc(name: str, **args) -> ToolCall:
    """Build a ToolCall — required so .ainvoke() returns a ToolMessage with .artifact."""
    return ToolCall(name=name, args=args, id=f"tc{next(_tc_id)}", type="tool_call")


def _tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}


def _allowlist_state(tools: dict) -> tuple[dict, dict]:
    """Reach the closure-scoped allowlist dicts behind find_candidates.

    The competitor-seed + mechanism-merge logic that populates the allowlist now
    lives in _find_candidates_impl, which closes over both allowed_diseases and
    allowed_efo_ids. The outer find_candidates coroutine wraps that impl (plus
    the seed-phase asyncio.Event); walk one layer down to the impl to find them.
    """
    fc = tools["find_candidates"]
    outer = dict(zip(fc.coroutine.__code__.co_freevars, fc.coroutine.__closure__))
    impl = outer["_find_candidates_impl"].cell_contents
    inner = dict(zip(impl.__code__.co_freevars, impl.__closure__))
    return (
        inner["allowed_diseases"].cell_contents,
        inner["allowed_efo_ids"].cell_contents,
    )


def _preset_mechanism_gate(tools: dict) -> None:
    """Pre-set analyze_mechanism_done so find_candidates can run in isolation.

    find_candidates blocks on analyze_mechanism_done before returning so its result
    reflects the merged competitor + mechanism allowlist. In tests that only
    exercise the competitor path (i.e. don't invoke analyze_mechanism), pre-set the
    gate so find_candidates returns immediately after its competitor work.

    The gate lives in the OUTER analyze_mechanism wrapper's closure (the try/finally
    references it directly); the _analyze_mechanism_impl inner function doesn't close
    over it because it doesn't reference it by name. Read from the outer wrapper.
    """
    am = tools["analyze_mechanism"]
    outer = dict(zip(am.coroutine.__code__.co_freevars, am.coroutine.__closure__))
    outer["analyze_mechanism_done"].cell_contents.set()


def _preset_both_gates(tools: dict) -> None:
    """Pre-set both seed-phase events (find_candidates_done + analyze_mechanism_done).

    analyze_literature / analyze_clinical_trials block on BOTH events before doing
    anything (so parallel tool calls can't race ahead of the seed phase). Tests that
    invoke those tools in isolation — without first calling find_candidates /
    analyze_mechanism — must pre-set both, or the awaits block forever. Both events
    are closed over by the analyze_literature wrapper, so read them from there.
    """
    al = tools["analyze_literature"]
    closure = dict(zip(al.coroutine.__code__.co_freevars, al.coroutine.__closure__))
    closure["find_candidates_done"].cell_contents.set()
    closure["analyze_mechanism_done"].cell_contents.set()


@pytest.fixture
def llm():
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)


# ------------------------------------------------------------------
# find_candidates — Open Targets + ChEMBL + openFDA seeding
#
# Metformin competitor candidates from Open Targets. This is a live,
# drift-prone set, so we assert a small stable subset rather than the full
# list. Re-verified against the live find_candidates output on 2026-06-06.
# ------------------------------------------------------------------

_DRUG = "metformin"
_EXPECTED_CANDIDATES_SUBSET = {
    "polycystic ovary syndrome",
    "gestational diabetes",
}

# TODO: fill in expected metformin aliases from a live ChEMBL run
# _EXPECTED_ALIASES_SUBSET: set[str] = {"metformin", "glucophage"}

# TODO: fill in expected FDA-approved indications seeded for metformin from openFDA
# _EXPECTED_APPROVED_INDICATIONS_SUBSET: set[str] = {"type 2 diabetes mellitus"}


async def test_find_candidates_random(llm, db_session_truncating, test_cache_dir):
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)
    _preset_mechanism_gate(tools)
    msg = await tools["find_candidates"].ainvoke(
        _tc("find_candidates", drug_name="imatinib")
    )
    diseases: list[str] = msg.artifact
    print(diseases)


async def my_test(llm, db_session_truncating, test_cache_dir):
    """find_candidates returns Open Targets candidate diseases, populates the closure-scoped
    allowlist, and seeds drug aliases + FDA-approved indications into the briefing store.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)
    _preset_mechanism_gate(tools)

    msg = await tools["find_candidates"].ainvoke(
        _tc("find_candidates", drug_name="bupropion")
    )

    diseases: list[str] = msg.artifact
    assert isinstance(diseases, list)


async def test_find_candidates_metformin(llm, db_session_truncating, test_cache_dir):
    """find_candidates returns Open Targets candidate diseases, populates the closure-scoped
    allowlist, and seeds drug aliases + FDA-approved indications into the briefing store.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)
    _preset_mechanism_gate(tools)

    msg = await tools["find_candidates"].ainvoke(
        _tc("find_candidates", drug_name=_DRUG)
    )

    diseases: list[str] = msg.artifact
    assert isinstance(diseases, list)
    assert all(isinstance(d, str) and d for d in diseases)
    assert _EXPECTED_CANDIDATES_SUBSET.issubset(set(diseases))

    # content string format from supervisor_tools.find_candidates
    assert msg.content.startswith(
        f"Found {len(diseases)} candidate diseases for {_DRUG}"
    )

    # Briefing must reflect the seeded aliases + FDA-approved indications
    briefing = tools["get_drug_briefing"].invoke({"drug_name": _DRUG})
    assert briefing.startswith(f"DRUG INTAKE: {_DRUG}")
    assert "Trade/generic names:" in briefing
    assert "(not yet resolved)" not in briefing  # aliases were seeded
    # TODO: tighten once expected aliases are known
    # for alias in _EXPECTED_ALIASES_SUBSET:
    #     assert alias.lower() in briefing.lower()

    # FDA-approved indications were seeded from the label
    assert "FDA-approved indications:" in briefing
    assert "(none discovered in this run)" not in briefing
    # TODO: tighten once expected approved indications are known
    # for ind in _EXPECTED_APPROVED_INDICATIONS_SUBSET:
    #     assert ind.lower() in briefing.lower()

    # Mechanism hasn't run yet
    assert "(mechanism agent has not run)" in briefing


# ------------------------------------------------------------------
# Reject path — analyze_literature / analyze_clinical_trials reject
# diseases that are not in the allowlist. No external APIs hit.
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,empty_artifact_type",
    [
        ("analyze_literature", LiteratureOutput),
        ("analyze_clinical_trials", ClinicalTrialsOutput),
    ],
)
async def test_analyze_rejects_unlisted_disease(
    llm, db_session_truncating, test_cache_dir, tool_name, empty_artifact_type
):
    """analyze_literature / analyze_clinical_trials must reject any disease that wasn't
    surfaced by find_candidates or promoted by analyze_mechanism. Returns an empty artifact
    of the right type and a REJECTED: content message naming the tool.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)
    # analyze_literature / analyze_clinical_trials await both seed-phase events
    # before the reject check. This test invokes them in isolation (no
    # find_candidates / analyze_mechanism call), so pre-set both events or the
    # awaits block forever. The allowlist stays empty, so the nonsense disease
    # is correctly rejected.
    _preset_both_gates(tools)

    msg = await tools[tool_name].ainvoke(
        _tc(tool_name, drug_name=_DRUG, disease_name="not-a-real-disease")
    )

    assert msg.content.startswith(
        "REJECTED: 'not-a-real-disease' is not in the allowed"
    )
    assert tool_name in msg.content
    assert isinstance(msg.artifact, empty_artifact_type)
    # An empty default-constructed artifact has no sub-results populated
    assert msg.artifact == empty_artifact_type()


# ------------------------------------------------------------------
# finalize_supervisor — returns the canonical completion content and a
# {summary, blurbs} artifact dict. No external APIs.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# analyze_mechanism merge — three-step dedup against competitor allowlist
#
# Imatinib's mechanism agent surfaces leukemia-class diseases that also appear
# as competitor entries (other BCR-ABL/c-KIT inhibitors trial the same diseases).
# We exercise the real pipeline end-to-end and assert structural properties of
# the post-merge allowlist:
#   - No two allowlist rows share an OT disease ID (the original bug would
#     produce two — one tagged "competitor", one tagged "mechanism").
#   - At least one row is tagged "both", proving the merge actually fired
#     (otherwise the no-collision assertion holds trivially).
# Mechanism candidates that fail all three dedup steps legitimately become
# new "mechanism"-tagged rows; we don't constrain their count.
# ------------------------------------------------------------------


_MERGE_DRUG = "semaglutide"


async def test_analyze_mechanism_dedups_against_competitor_allowlist(
    llm, db_session_truncating, test_cache_dir
):
    """analyze_mechanism must not produce duplicate allowlist entries for diseases
    that competitor and mechanism both surface. Verifies the three-step dedup:
    (1) ID match, (2) exact-name match, (3) OT name-resolve fallback.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)

    # find_candidates blocks on analyze_mechanism_done before returning, so the two
    # seed-phase tools must run concurrently (mirrors how the LLM invokes them in
    # parallel in production).
    await asyncio.gather(
        tools["find_candidates"].ainvoke(_tc("find_candidates", drug_name=_MERGE_DRUG)),
        tools["analyze_mechanism"].ainvoke(
            _tc("analyze_mechanism", drug_name=_MERGE_DRUG)
        ),
    )

    allowed_diseases, allowed_efo_ids = _allowlist_state(tools)

    assert (
        allowed_diseases
    ), "find_candidates should have seeded competitor allowlist entries"

    # No two allowlist rows share an OT disease ID — the original bug.
    id_to_keys: dict[str, list[str]] = {}
    for disease_id, key in allowed_efo_ids.items():
        id_to_keys.setdefault(disease_id, []).append(key)
    duplicates = {did: keys for did, keys in id_to_keys.items() if len(keys) > 1}
    assert (
        not duplicates
    ), f"OT disease IDs must map to a single allowlist row; found duplicates: {duplicates}"

    # At least one row should be tagged "both" — confirms the merge fired and the
    # assertion above isn't holding trivially because the merge never touched anything.
    sources = [source for _, source in allowed_diseases.values()]
    assert "both" in sources, (
        f"Expected at least one disease tagged 'both' after analyze_mechanism, "
        f"got sources: {sorted(set(sources))}"
    )


# ------------------------------------------------------------------
# analyze_mechanism — mechanism-only promotion
#
# Imatinib's mechanism agent (BCR-ABL / KIT / PDGFR targets) surfaces leukemia
# and mastocytosis-class diseases that imatinib is approved or trialed for, but
# that don't show up in the competitor allowlist (because competitors trial
# different conditions, or imatinib's own approvals get filtered out by the FDA
# approval check in find_candidates). Those diseases must be promoted by
# analyze_mechanism with source="mechanism" so they're investigatable downstream.
# ------------------------------------------------------------------


_MECH_ONLY_DRUG = "imatinib"


async def test_analyze_mechanism_promotes_mechanism_only_candidates(
    llm, db_session_truncating, test_cache_dir
):
    """For imatinib, analyze_mechanism must add at least one mechanism-only
    candidate (source='mechanism') that did not appear in the competitor list.
    This is the path that makes mechanism-surfaced diseases reachable by
    analyze_literature / analyze_clinical_trials.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)

    # find_candidates blocks on analyze_mechanism_done before returning, so the two
    # seed-phase tools must run concurrently (mirrors how the LLM invokes them in
    # parallel in production).
    await asyncio.gather(
        tools["find_candidates"].ainvoke(
            _tc("find_candidates", drug_name=_MECH_ONLY_DRUG)
        ),
        tools["analyze_mechanism"].ainvoke(
            _tc("analyze_mechanism", drug_name=_MECH_ONLY_DRUG)
        ),
    )

    allowed_diseases, _ = _allowlist_state(tools)

    # Entries originating from find_candidates carry source 'competitor' or 'both' (the
    # latter if analyze_mechanism's merge upgraded them). Mechanism-only entries carry
    # source 'mechanism'. The two sets must be disjoint by construction.
    competitor_or_both_keys = {
        k
        for k, (_, source) in allowed_diseases.items()
        if source in ("competitor", "both")
    }
    assert (
        competitor_or_both_keys
    ), f"find_candidates should seed competitor entries for {_MECH_ONLY_DRUG}"

    mechanism_only_keys = {
        k for k, (_, source) in allowed_diseases.items() if source == "mechanism"
    }
    assert mechanism_only_keys, (
        f"analyze_mechanism must promote at least one mechanism-only candidate for "
        f"{_MECH_ONLY_DRUG}; got sources: "
        f"{sorted({s for _, s in allowed_diseases.values()})}"
    )
    # Mechanism-only entries must not overlap with competitor/both entries.
    assert not (mechanism_only_keys & competitor_or_both_keys), (
        f"Mechanism-only entries should not overlap with competitor/both keys; "
        f"overlap: {mechanism_only_keys & competitor_or_both_keys}"
    )


# ------------------------------------------------------------------
# analyze_mechanism — mechanism-promoted disease is investigatable downstream
#
# Closes the loop on the promotion contract: a disease that only enters the
# allowlist via analyze_mechanism (source='mechanism') must be accepted by
# analyze_literature and analyze_clinical_trials. Without this, promotion is
# inert — the test above proves the row gets added, but not that downstream
# tools honor it.
# ------------------------------------------------------------------


async def test_mechanism_promoted_disease_is_investigatable_downstream(
    llm, db_session_truncating, test_cache_dir
):
    """A disease promoted by analyze_mechanism (source='mechanism') must be accepted
    by analyze_literature and analyze_clinical_trials — i.e. no REJECTED message and
    the artifact is the real output type, not the empty default.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)

    # find_candidates blocks on analyze_mechanism_done before returning, so the two
    # seed-phase tools must run concurrently (mirrors how the LLM invokes them in
    # parallel in production).
    await asyncio.gather(
        tools["find_candidates"].ainvoke(
            _tc("find_candidates", drug_name=_MECH_ONLY_DRUG)
        ),
        tools["analyze_mechanism"].ainvoke(
            _tc("analyze_mechanism", drug_name=_MECH_ONLY_DRUG)
        ),
    )

    allowed_diseases, _ = _allowlist_state(tools)
    mechanism_only = [
        canonical
        for _, (canonical, source) in allowed_diseases.items()
        if source == "mechanism"
    ]
    assert mechanism_only, (
        f"No mechanism-only candidates promoted for {_MECH_ONLY_DRUG}; "
        f"cannot exercise downstream investigation path."
    )

    # Pick a deterministic candidate (sorted) so failures are reproducible.
    target_disease = sorted(mechanism_only)[0]

    lit_msg = await tools["analyze_literature"].ainvoke(
        _tc(
            "analyze_literature",
            drug_name=_MECH_ONLY_DRUG,
            disease_name=target_disease,
        )
    )
    assert not lit_msg.content.startswith("REJECTED:"), (
        f"analyze_literature rejected mechanism-promoted disease {target_disease!r}: "
        f"{lit_msg.content}"
    )
    assert isinstance(lit_msg.artifact, LiteratureOutput)
    assert lit_msg.artifact != LiteratureOutput(), (
        f"analyze_literature returned an empty default artifact for mechanism-promoted "
        f"disease {target_disease!r} — investigation did not run."
    )

    ct_msg = await tools["analyze_clinical_trials"].ainvoke(
        _tc(
            "analyze_clinical_trials",
            drug_name=_MECH_ONLY_DRUG,
            disease_name=target_disease,
        )
    )
    assert not ct_msg.content.startswith("REJECTED:"), (
        f"analyze_clinical_trials rejected mechanism-promoted disease "
        f"{target_disease!r}: {ct_msg.content}"
    )
    assert isinstance(ct_msg.artifact, ClinicalTrialsOutput)
    assert ct_msg.artifact != ClinicalTrialsOutput(), (
        f"analyze_clinical_trials returned an empty default artifact for "
        f"mechanism-promoted disease {target_disease!r} — investigation did not run."
    )


async def test_finalize_supervisor_echoes_summary(
    llm, db_session_truncating, test_cache_dir
):
    """finalize_supervisor returns ('Supervisor analysis complete.', {summary, blurbs}).

    Blurbs for diseases not in the allowlist are dropped at the tool boundary; with no prior
    find_candidates / analyze_mechanism call, the allowlist is empty so all blurbs drop.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)

    summary_text = (
        "Metformin shows the strongest repurposing signal in polycystic ovary syndrome, "
        "with supporting literature and active trials."
    )
    msg = await tools["finalize_supervisor"].ainvoke(
        _tc(
            "finalize_supervisor",
            summary=summary_text,
            blurbs=[],
        )
    )

    assert msg.content == "Supervisor analysis complete."
    assert isinstance(msg.artifact, dict)
    assert msg.artifact["summary"] == summary_text
    assert msg.artifact["blurbs"] == []


# ------------------------------------------------------------------
# Cutoff plumbing — date_before given to build_supervisor_tools must
# reach analyze_clinical_trials so the supervisor's investigation is
# reproducible across runs. Mirrors the cutoff contract verified in
# tests/integration/agents/clinical_trials/test_clinical_trials_tools.py
# but driven through the supervisor's tool wrapper.
# ------------------------------------------------------------------

_CUTOFF = date(2025, 1, 1)
_CUTOFF_DRUG = "semaglutide"


async def test_analyze_clinical_trials_respects_cutoff(
    llm, db_session_truncating, test_cache_dir
):
    """When build_supervisor_tools is given date_before=2025-01-01, analyze_clinical_trials
    must forward the cutoff to its sub-agent so every returned trial has a
    start_date strictly before the cutoff.
    """
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating, date_before=_CUTOFF
    )
    tools = _tool_map(tools_list)
    _preset_mechanism_gate(tools)

    # Populate the allowlist; pick a real candidate from the live result set so the
    # test doesn't bake in an assumption about a specific OT disease being present.
    find_msg = await tools["find_candidates"].ainvoke(
        _tc("find_candidates", drug_name=_CUTOFF_DRUG)
    )
    candidates: list[str] = find_msg.artifact
    assert candidates, (
        f"find_candidates returned no diseases for {_CUTOFF_DRUG}; "
        f"cannot exercise cutoff path"
    )
    target_disease = candidates[0]

    ct_msg = await tools["analyze_clinical_trials"].ainvoke(
        _tc(
            "analyze_clinical_trials",
            drug_name=_CUTOFF_DRUG,
            disease_name=target_disease,
        )
    )

    assert not ct_msg.content.startswith("REJECTED:"), (
        f"analyze_clinical_trials rejected {target_disease!r} despite it being "
        f"returned by find_candidates: {ct_msg.content}"
    )

    output: ClinicalTrialsOutput = ct_msg.artifact
    assert isinstance(output, ClinicalTrialsOutput)

    # Every trial in every scope (search / completed / terminated) must
    # have started before the cutoff. CT.gov start_date is "YYYY-MM-DD" or
    # "YYYY-MM" — both compare lexicographically against the ISO cutoff.
    cutoff_iso = _CUTOFF.isoformat()
    scopes: list[tuple[str, list]] = []
    if output.search is not None:
        scopes.append(("search", output.search.trials))
    if output.completed is not None:
        scopes.append(("completed", output.completed.trials))
    if output.terminated is not None:
        scopes.append(("terminated", output.terminated.trials))

    assert any(trials for _, trials in scopes), (
        f"No trials returned in any scope for {_CUTOFF_DRUG} × {target_disease!r}; "
        f"cannot verify cutoff was applied (a no-result run trivially satisfies it)"
    )

    for scope_name, trials in scopes:
        for t in trials:
            assert t.start_date is not None, (
                f"{scope_name} trial {t.nct_id} has no start_date — "
                f"cannot verify cutoff compliance"
            )
            assert t.start_date < cutoff_iso, (
                f"{scope_name} trial {t.nct_id} start_date={t.start_date} "
                f"is not before cutoff {cutoff_iso}"
            )


# ------------------------------------------------------------------
# Cutoff plumbing — date_before given to build_supervisor_tools must
# also reach analyze_literature so PubMed queries respect the same
# temporal cutoff. Mirrors the literature-tools cutoff test, but driven
# through the supervisor's analyze_literature wrapper which spins up
# its own literature sub-agent per call.
# ------------------------------------------------------------------


async def test_analyze_literature_respects_cutoff(
    llm, db_session_truncating, test_cache_dir
):
    """When build_supervisor_tools is given date_before=2025-01-01, analyze_literature
    must forward the cutoff to its sub-agent so every PMID returned has a
    publication date strictly before the cutoff.
    """
    from indication_scout.data_sources.pubmed import PubMedClient

    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating, date_before=_CUTOFF
    )
    tools = _tool_map(tools_list)
    _preset_mechanism_gate(tools)

    find_msg = await tools["find_candidates"].ainvoke(
        _tc("find_candidates", drug_name=_CUTOFF_DRUG)
    )
    candidates: list[str] = find_msg.artifact
    assert candidates, (
        f"find_candidates returned no diseases for {_CUTOFF_DRUG}; "
        f"cannot exercise cutoff path"
    )
    target_disease = candidates[0]

    lit_msg = await tools["analyze_literature"].ainvoke(
        _tc(
            "analyze_literature",
            drug_name=_CUTOFF_DRUG,
            disease_name=target_disease,
        )
    )

    assert not lit_msg.content.startswith("REJECTED:"), (
        f"analyze_literature rejected {target_disease!r} despite it being "
        f"returned by find_candidates: {lit_msg.content}"
    )

    output: LiteratureOutput = lit_msg.artifact
    assert isinstance(output, LiteratureOutput)
    assert output.pmids, (
        f"analyze_literature returned no PMIDs for {_CUTOFF_DRUG} × {target_disease!r}; "
        f"cannot verify cutoff was applied (a no-result run trivially satisfies it)"
    )

    # Re-run the production date filter over the returned PMIDs. If the cutoff
    # was correctly applied during PubMed search, none should be dropped.
    # The PubMedClient stores publication dates as text in mixed formats
    # ("2024", "2024-Mar", "2024-03-15"), so this avoids re-implementing date
    # parsing and uses esummary's sortpubdate (always YYYY/MM/DD) — the same
    # field the production filter uses.
    async with PubMedClient(cache_dir=test_cache_dir) as client:
        kept = await client._filter_pmids_by_date(output.pmids, _CUTOFF)

    leaked = sorted(set(output.pmids) - set(kept))
    assert not leaked, (
        f"analyze_literature returned {len(leaked)} PMID(s) whose sortpubdate "
        f"is on or after the cutoff {_CUTOFF.isoformat()} — cutoff was not applied "
        f"at the PubMed search layer. Leaked PMIDs: {leaked[:10]}"
    )


# ------------------------------------------------------------------
# Phase-tier consistency: a REAL LLM, given a completed-Phase-3 pair, must not
# leave a sub-Phase-3 understatement ("exploratory only", "post-Phase 2", "no
# development program") in a blurb's free-text fields, where it would contradict
# the authoritative `stage`. Proves the supervisor prompt's phase-tier rule under
# a live model — the deterministic `stage` override fixes `stage` itself; this
# guards the sibling fields the override does NOT touch. (See the bupropion ×
# cocaine and semaglutide × T1DM report regressions.)
# ------------------------------------------------------------------

from indication_scout.agents.clinical_trials.clinical_trials_output import (  # noqa: E402
    TrialSignals,
)
from indication_scout.models.model_clinical_trials import (  # noqa: E402
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary  # noqa: E402

# Lowercased phrases that name a phase tier BELOW the authoritative completed Phase 3.
# Any of these in a sibling field contradicts stage="Phase 3 completed".
_UNDERSTATEMENT_PHRASES = (
    "exploratory only",
    "phase 4 only",
    "phase 4 exploratory",
    "post-phase 2",
    "post-phase 1",
    "stalled post-phase",
    "no dedicated development program",
    "no formal development program",
    "no phase 2 or phase 3",
    "early-phase only",
    "no development program",
)


def _finalize_closure(tools: dict) -> tuple[dict, dict]:
    """Reach finalize_supervisor's closure-scoped findings_local + allowed_diseases."""
    fin = tools["finalize_supervisor"]
    closure = dict(zip(fin.coroutine.__code__.co_freevars, fin.coroutine.__closure__))
    return (
        closure["findings_local"].cell_contents,
        closure["allowed_diseases"].cell_contents,
    )


async def test_finalize_blurb_fields_have_no_subphase3_understatement_live(
    llm, db_session_truncating, test_cache_dir
):
    """A REAL LLM run: dev_stage=completed_phase3 (a completed Phase 2/Phase 3 trial), and the
    incoming blurb deliberately seeds 'exploratory only' / 'no dedicated development program'
    understatements in verdict, key_risk, and prose. After the live finalize (which runs the
    real fact critic), the authoritative `stage` must say Phase 3 completed AND no sibling
    field may carry a sub-Phase-3 understatement that contradicts it."""
    svc = RetrievalService(test_cache_dir)
    tools_list, _, _ = build_supervisor_tools(
        llm=llm, svc=svc, db=db_session_truncating
    )
    tools = _tool_map(tools_list)
    findings_local, allowed_diseases = _finalize_closure(tools)

    disease = "cocaine dependence"
    allowed_diseases[disease] = (disease, "competitor")
    # Real-shaped trial data: a completed Phase 2/Phase 3 → dev_stage completed_phase3.
    completed = [
        Trial(
            nct_id="NCT02111798",
            title="Bupropion-Enhanced Contingency Management for Cocaine Dependence",
            phase="Phase 2/Phase 3",
            overall_status="COMPLETED",
        ),
        Trial(
            nct_id="NCT00227812",
            title="Bupropion in cocaine users",
            phase="Phase 2",
            overall_status="COMPLETED",
        ),
    ]
    findings_local[disease] = {
        "literature": LiteratureOutput(
            pmids=["1", "2", "3", "4"],
            evidence_summary=EvidenceSummary(
                summary="RCTs of bupropion for cocaine dependence, mixed efficacy.",
                strength="strong",
                direction="mixed",
                study_count=4,
                is_observational=False,
            ),
        ),
        "clinical_trials": ClinicalTrialsOutput(
            search=SearchTrialsResult(total_count=10, trials=completed),
            completed=CompletedTrialsResult(total_count=2, trials=completed),
            terminated=TerminatedTrialsResult(total_count=0, trials=[]),
            relevant_nct_ids=["NCT02111798", "NCT00227812"],
            signals=TrialSignals(
                highest_completed_phase="Phase 2/Phase 3",
                has_completed_phase3=True,
                completed_phase3_nct_ids=["NCT02111798"],
                dev_stage="completed_phase3",
            ),
        ),
    }

    # critique_ranking must precede finalize (finalize rejects otherwise).
    await tools["critique_ranking"].ainvoke(_tc("critique_ranking", blurbs=[]))

    seeded_blurb = {
        "disease": disease,
        "stage": "Phase 4 exploratory only; no dedicated development program",
        "literature": "Strong, mixed RCTs",
        "blocker": "Stalled post-Phase 2",
        "active_programs": "None active",
        "key_risk": "No dedicated development program; exploratory only",
        "verdict": "Exploratory only, no formal development program",
        "prose": (
            "Bupropion has been studied primarily through early exploratory work with no "
            "dedicated Phase 2 or Phase 3 program. The hypothesis remains unproven."
        ),
    }
    msg = await tools["finalize_supervisor"].ainvoke(
        _tc(
            "finalize_supervisor",
            summary=f"Ranked repurposing signals:\n1. {disease} — strong but mixed",
            blurbs=[seeded_blurb],
        )
    )

    blurbs = msg.artifact["blurbs"]
    assert len(blurbs) == 1, f"expected 1 blurb, got {blurbs}"
    b = blurbs[0]

    # stage is the authoritative override — it MUST state the completed Phase 3.
    assert "Phase 3 completed for this indication" in b["stage"], b["stage"]

    # No sibling free-text field may carry a sub-Phase-3 understatement contradicting stage.
    for field in ("verdict", "blocker", "active_programs", "key_risk", "prose"):
        val = (b.get(field) or "").lower()
        leaked = [p for p in _UNDERSTATEMENT_PHRASES if p in val]
        assert not leaked, (
            f"blurb field {field!r} contradicts stage=Phase 3 completed with "
            f"understatement(s) {leaked}: {b.get(field)!r}"
        )
