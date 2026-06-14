"""Unit tests for supervisor_tools — briefing rendering."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import (
    MechanismCandidate,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary

# --- semaglutide × NAFLD regression: briefing surfaces MASH so the prompt's ---
# --- APPROVED-CANDIDATE SHORT-CIRCUIT case C can fire on NAFLD. -----------------


def _build_tools_with_drug_facts(
    drug_name: str,
    approved_indications: list[str],
    drug_aliases: list[str] | None = None,
):
    """Build supervisor tools and prepopulate drug_facts for `drug_name`.

    Returns (tools_by_name, drug_facts) so tests can call get_drug_briefing
    and verify the rendered output that the supervisor's APPROVED-CANDIDATE
    SHORT-CIRCUIT depends on.
    """
    llm = MagicMock()
    svc = MagicMock()
    db = MagicMock()
    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_clinical_trials_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_mechanism_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
    ):
        tools, _, _ = build_supervisor_tools(llm=llm, svc=svc, db=db)

    by_name = {t.name: t for t in tools}

    # Reach drug_facts via _ensure_drug_entry's closure. None of the tools
    # close over drug_facts directly; they go through _ensure_drug_entry
    # (the writers) and _render_briefing (the reader), both of which do.
    # find_candidates's outer coroutine wraps _find_candidates_impl in a
    # try/finally; _ensure_drug_entry lives in the impl's closure.
    fc = by_name["find_candidates"]
    fc_outer = dict(zip(fc.coroutine.__code__.co_freevars, fc.coroutine.__closure__))
    fc_impl = fc_outer["_find_candidates_impl"].cell_contents
    fc_closure = dict(zip(fc_impl.__code__.co_freevars, fc_impl.__closure__))
    ensure_fn = fc_closure["_ensure_drug_entry"].cell_contents
    ensure_closure = dict(zip(ensure_fn.__code__.co_freevars, ensure_fn.__closure__))
    drug_facts = ensure_closure["drug_facts"].cell_contents

    drug_facts[drug_name.lower().strip()] = {
        "drug_name": drug_name,
        "drug_aliases": drug_aliases or [],
        "approved_indications": list(approved_indications),
        "mechanism_targets": [],
        "mechanism_disease_associations": [],
    }
    return by_name, drug_facts


def test_semaglutide_briefing_lists_mash_when_seeded():
    """The briefing the supervisor reads MUST include MASH for semaglutide.

    This is the regression that prevents the original failure: without MASH
    in the briefing, the supervisor cannot apply the APPROVED-CANDIDATE
    SHORT-CIRCUIT case C ("NAFLD is a SUPERSET of approved MASH") and would
    demote NAFLD as settled-unfavorable on the strength of completed Phase 3
    trials with no approval.
    """
    by_name, _ = _build_tools_with_drug_facts(
        drug_name="semaglutide",
        approved_indications=[
            "type 2 diabetes mellitus",
            "chronic weight management",
            "MASH",
        ],
        drug_aliases=["Ozempic", "Wegovy", "Rybelsus"],
    )

    briefing = by_name["get_drug_briefing"].invoke({"drug_name": "semaglutide"})

    assert "DRUG INTAKE: semaglutide" in briefing
    assert "Trade/generic names: Ozempic, Wegovy, Rybelsus" in briefing
    assert "FDA-approved indications:" in briefing
    assert "- MASH" in briefing
    assert "- type 2 diabetes mellitus" in briefing
    assert "- chronic weight management" in briefing


def test_briefing_handles_unknown_drug_gracefully():
    """get_drug_briefing on a drug with no facts should NOT crash, just return a
    well-formed empty-state briefing — the supervisor relies on this when it
    calls the tool before any sub-agent has populated drug_facts."""
    by_name, _ = _build_tools_with_drug_facts(
        drug_name="semaglutide",
        approved_indications=["MASH"],
    )

    briefing = by_name["get_drug_briefing"].invoke({"drug_name": "metformin"})

    assert "DRUG INTAKE: metformin" in briefing
    assert "no facts collected yet" in briefing


# --- analyze_mechanism merge: EFO ID dedup against competitor allowlist --------


def _build_tools_and_allowlists(
    competitors: dict[str, str],
):
    """Build supervisor tools and seed the closure-scoped competitor allowlist.

    competitors maps lowercase disease name → EFO ID. Each entry is registered as a
    "competitor"-source allowlist row, with its EFO ID indexed in allowed_efo_ids.

    Returns (merge_and_dedup, mechanism_candidates_buffer, allowed_diseases,
    allowed_efo_ids). The merge logic now lives on _merge_and_dedup_impl
    (reached via find_candidates → _find_candidates_impl → merge_and_dedup),
    so tests seed the buffer directly and invoke merge_and_dedup, bypassing
    the seed-phase gates and the analyze_mechanism path entirely.
    """
    llm = MagicMock()
    svc = MagicMock()
    db = MagicMock()
    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_clinical_trials_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_mechanism_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
    ):
        tools, _, _ = build_supervisor_tools(llm=llm, svc=svc, db=db)

    by_name = {t.name: t for t in tools}
    fc = by_name["find_candidates"]
    # find_candidates wraps _find_candidates_impl; _find_candidates_impl closes
    # over allowed_diseases, allowed_efo_ids, and merge_and_dedup. The merge
    # function in turn closes over _merge_and_dedup_impl which holds the same
    # allowlist dicts plus mechanism_candidates_buffer.
    fc_outer = dict(zip(fc.coroutine.__code__.co_freevars, fc.coroutine.__closure__))
    fc_impl = fc_outer["_find_candidates_impl"].cell_contents
    fc_closure = dict(zip(fc_impl.__code__.co_freevars, fc_impl.__closure__))
    allowed_diseases = fc_closure["allowed_diseases"].cell_contents
    allowed_efo_ids = fc_closure["allowed_efo_ids"].cell_contents
    merge_and_dedup = fc_closure["merge_and_dedup"].cell_contents
    # Reach mechanism_candidates_buffer through _merge_and_dedup_impl.
    md_closure = dict(
        zip(merge_and_dedup.__code__.co_freevars, merge_and_dedup.__closure__)
    )
    md_impl = md_closure["_merge_and_dedup_impl"].cell_contents
    md_impl_closure = dict(zip(md_impl.__code__.co_freevars, md_impl.__closure__))
    mechanism_candidates_buffer = md_impl_closure[
        "mechanism_candidates_buffer"
    ].cell_contents

    allowed_diseases.clear()
    allowed_efo_ids.clear()
    mechanism_candidates_buffer.clear()
    for name, efo_id in competitors.items():
        allowed_diseases[name] = (name, "competitor")
        allowed_efo_ids[efo_id] = name

    return (
        merge_and_dedup,
        mechanism_candidates_buffer,
        allowed_diseases,
        allowed_efo_ids,
    )


def _ot_client_mock(name_to_resolved_id: dict[str, str | None] | None = None):
    """Return a MagicMock shaped like OpenTargetsClient as an async context manager.

    The yielded client exposes `resolve_disease_id(name)` as an AsyncMock whose
    return value is looked up in `name_to_resolved_id` (default: returns None for
    every name, simulating "no OT search hit").
    """
    mapping = name_to_resolved_id or {}

    async def _resolve(name: str) -> str | None:
        return mapping.get(name)

    client = MagicMock()
    client.resolve_disease_id = AsyncMock(side_effect=_resolve)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)

    return MagicMock(return_value=ctx)


@pytest.mark.parametrize(
    "competitors, mech_candidates, resolved_ids, expected_diseases, expected_efo_ids",
    [
        # Path 1: Same EFO, different name → upgrade to "both", no duplicate row.
        (
            {"non-small cell lung cancer": "EFO_0003060"},
            [("NSCLC", "EFO_0003060")],
            {},
            {"non-small cell lung cancer": ("non-small cell lung cancer", "both")},
            {"EFO_0003060": "non-small cell lung cancer"},
        ),
        # No path matches → two separate rows; mechanism candidate is added.
        (
            {"narcolepsy": "EFO_0003757"},
            [("type 2 diabetes mellitus", "EFO_0001360")],
            {},
            {
                "narcolepsy": ("narcolepsy", "competitor"),
                "type 2 diabetes mellitus": ("type 2 diabetes mellitus", "mechanism"),
            },
            {"EFO_0003757": "narcolepsy", "EFO_0001360": "type 2 diabetes mellitus"},
        ),
        # Path 2: Mechanism candidate has no EFO but the name matches a competitor →
        # name-fallback upgrades the existing row to "both".
        (
            {"narcolepsy": "EFO_0003757"},
            [("narcolepsy", None)],
            {},
            {"narcolepsy": ("narcolepsy", "both")},
            {"EFO_0003757": "narcolepsy"},
        ),
        # Mechanism candidate has no EFO and no name match → added as a new mechanism row;
        # allowed_efo_ids is unchanged.
        (
            {"narcolepsy": "EFO_0003757"},
            [("alzheimer disease", None)],
            {},
            {
                "narcolepsy": ("narcolepsy", "competitor"),
                "alzheimer disease": ("alzheimer disease", "mechanism"),
            },
            {"EFO_0003757": "narcolepsy"},
        ),
        # Path 2 (with no-EFO competitor): Competitor has no EFO (dropped during LLM merge);
        # mechanism EFO doesn't match an existing entry. Name match still works.
        (
            {"narcolepsy": "EFO_0003757", "depression": ""},
            [("depression", "EFO_0003761")],
            {},
            {
                "narcolepsy": ("narcolepsy", "competitor"),
                "depression": ("depression", "both"),
            },
            {"EFO_0003757": "narcolepsy", "EFO_0003761": "depression"},
        ),
        # Path 3: candidate ID missing AND name doesn't match — OT search resolves
        # the candidate name to an existing competitor ID, upgrading the existing row.
        (
            {"non-small cell lung cancer": "EFO_0003060"},
            [("nsclc adenocarcinoma", None)],
            {"nsclc adenocarcinoma": "EFO_0003060"},
            {"non-small cell lung cancer": ("non-small cell lung cancer", "both")},
            {"EFO_0003060": "non-small cell lung cancer"},
        ),
    ],
)
async def test_analyze_mechanism_merges_by_efo_id(
    competitors,
    mech_candidates,
    resolved_ids,
    expected_diseases,
    expected_efo_ids,
):
    """merge_and_dedup dedups against the competitor allowlist via three steps:
    (1) ID match, (2) exact-name match, (3) OT name-resolve fallback. The merge
    runs in find_candidates after analyze_mechanism buffers raw candidates; this
    test drives merge_and_dedup directly with a pre-seeded buffer."""
    # Drop empty-string EFOs from the seeded allowed_efo_ids — they're sentinels
    # for "competitor present but no EFO known" and the helper would index them.
    competitors_with_efo = {n: e for n, e in competitors.items() if e}
    (
        merge_and_dedup,
        mechanism_candidates_buffer,
        allowed_diseases,
        allowed_efo_ids,
    ) = _build_tools_and_allowlists(competitors_with_efo)

    # Re-add the no-EFO competitor entries (those don't enter allowed_efo_ids).
    for name, efo_id in competitors.items():
        if not efo_id:
            allowed_diseases[name] = (name, "competitor")

    for name, efo in mech_candidates:
        mechanism_candidates_buffer.append(
            MechanismCandidate(disease_name=name, disease_id=efo)
        )

    with patch(
        "indication_scout.agents.supervisor.supervisor_tools.OpenTargetsClient",
        new=_ot_client_mock(resolved_ids),
    ):
        await merge_and_dedup(drug_name="testdrug")

    assert allowed_diseases == expected_diseases
    assert allowed_efo_ids == expected_efo_ids


# --- holdout summary reconstruction: byte-equal output against the format ------
# --- documented in supervisor_holdout.txt. Verifies the deterministic path -----
# --- replaces whatever string the LLM returns when date_before is set. ---------


def _make_lit(
    strength: str, n_pmids: int, study_count: int, direction: str = "supports"
) -> LiteratureOutput:
    return LiteratureOutput(
        pmids=[str(1000000 + i) for i in range(n_pmids)],
        evidence_summary=EvidenceSummary(
            strength=strength,
            direction=direction,
            study_count=study_count,
            summary="",
            key_findings=[],
        ),
    )


def _make_ct(total: int, completed: int, terminated: int) -> ClinicalTrialsOutput:
    return ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=total),
        completed=CompletedTrialsResult(total_count=completed),
        terminated=TerminatedTrialsResult(total_count=terminated),
    )


def _holdout_tools_and_closure(cutoff: date):
    """Build supervisor tools with date_before set, return (tools_by_name,
    findings_local, allowed_diseases) seeded for direct closure manipulation."""
    llm = MagicMock()
    svc = MagicMock()
    db = MagicMock()
    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_clinical_trials_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.build_mechanism_agent",
            new=MagicMock(return_value=MagicMock()),
        ),
    ):
        tools, _, _ = build_supervisor_tools(
            llm=llm, svc=svc, db=db, date_before=cutoff
        )

    by_name = {t.name: t for t in tools}
    fin = by_name["finalize_supervisor"]
    # finalize_supervisor is a direct @tool decoration — closure lives on
    # .coroutine.__closure__ alongside the closure-scoped state we need to seed.
    fin_closure = dict(
        zip(fin.coroutine.__code__.co_freevars, fin.coroutine.__closure__)
    )
    findings_local = fin_closure["findings_local"].cell_contents
    allowed_diseases = fin_closure["allowed_diseases"].cell_contents
    return by_name, findings_local, allowed_diseases


async def test_holdout_summary_reconstructed_imatinib():
    """imatinib"""
    pass

async def test_holdout_summary_reconstructed_for_sildenafil_2005():
    """Sildenafil holdout at 2005-06-01: deterministic reconstruction must produce
    the structured fact list documented in supervisor_holdout.txt, regardless of
    what string the LLM passes in. Reproduces the per-disease artifact values
    visible in snapshots/holdouts/sildenafil_holdout_2005-06-01_2026-05-11_21-58-29.md
    and asserts byte-equality with the expected ranked block."""
    by_name, findings_local, allowed_diseases = _holdout_tools_and_closure(
        date(2005, 6, 1)
    )

    # Disease → (strength, direction, n_pmids, study_count, total, completed, terminated).
    # Values pulled from the per-disease sections of the May-11 broken snapshot.
    # PMID counts taken from the bulleted exclusions block (Coronary Artery
    # Disease=72, Covid-19=32) and from a synthetic spread for the ranked
    # candidates; PMID ranking only matters for tiebreaks and doesn't affect
    # this case because primary keys (strength, total trials) already separate
    # every candidate. direction is the gate key: "supports"/"contradicts"/"mixed"
    # is real evidence (kept); "none" with 0 trials is excluded. The two
    # zero-study_count pairs (coronary, covid) are excluded via study_count==0.
    cases = {
        "pulmonary hypertension": ("moderate", "supports", 300, 3, 9, 5, 0),
        "pulmonary arterial hypertension": ("moderate", "supports", 381, 3, 4, 0, 0),
        "benign prostatic hyperplasia": ("moderate", "supports", 101, 3, 1, 1, 0),
        "raynaud disease": ("weak", "supports", 552, 2, 0, 0, 0),
        "cardiovascular disease": ("weak", "supports", 200, 3, 0, 0, 0),
        "hypertension": ("none", "supports", 518, 1, 15, 7, 0),
        "coronary artery disease": ("none", "none", 72, 0, 0, 0, 0),
        "covid-19": ("none", "none", 32, 0, 0, 0, 0),
    }
    canonical_case = {
        "pulmonary hypertension": "pulmonary hypertension",
        "pulmonary arterial hypertension": "pulmonary arterial hypertension",
        "benign prostatic hyperplasia": "benign prostatic hyperplasia",
        "raynaud disease": "raynaud disease",
        "cardiovascular disease": "cardiovascular disease",
        "hypertension": "hypertension",
        "coronary artery disease": "coronary artery disease",
        "covid-19": "covid-19",
    }
    for lower, (
        strength,
        direction,
        n_pmids,
        study_count,
        total,
        comp,
        term,
    ) in cases.items():
        allowed_diseases[lower] = (canonical_case[lower], "competitor")
        findings_local[lower] = {
            "literature": _make_lit(strength, n_pmids, study_count, direction),
            "clinical_trials": _make_ct(total, comp, term),
        }

    # Ordering gate: finalize_supervisor is rejected until critique_ranking has
    # run this session. Empty blurbs short-circuits before any LLM call.
    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "test_critique",
            "type": "tool_call",
        }
    )

    msg = await by_name["finalize_supervisor"].ainvoke(
        {
            "name": "finalize_supervisor",
            "args": {"summary": "LLM PROSE THAT SHOULD BE DISCARDED", "blurbs": []},
            "id": "test_call",
            "type": "tool_call",
        }
    )

    expected = (
        "1. pulmonary hypertension — literature: moderate, 300 PMIDs; "
        "trials: 9 total, 5 completed, 0 terminated.\n"
        "2. pulmonary arterial hypertension — literature: moderate, 381 PMIDs; "
        "trials: 4 total, 0 completed, 0 terminated.\n"
        "3. benign prostatic hyperplasia — literature: moderate, 101 PMIDs; "
        "trials: 1 total, 1 completed, 0 terminated.\n"
        "4. raynaud disease — literature: weak, 552 PMIDs; "
        "trials: 0 total, 0 completed, 0 terminated.\n"
        "5. cardiovascular disease — literature: weak, 200 PMIDs; "
        "trials: 0 total, 0 completed, 0 terminated.\n"
        "6. hypertension — literature: none, 518 PMIDs; "
        "trials: 15 total, 7 completed, 0 terminated.\n"
        "\n"
        "Evidence gate exclusions: coronary artery disease, covid-19."
    )
    assert msg.artifact["summary"] == expected
    assert msg.artifact["blurbs"] == []


async def test_holdout_summary_contradicts_ranks_bottom_and_not_excluded():
    """A robustly-disproven pair (strong evidence, direction=contradicts, 0 trials) must
    survive the evidence gate AND rank below every supporting pair, with a "contradicts"
    note. This locks in the direction/strength split: strong contradiction is not erased."""
    by_name, findings_local, allowed_diseases = _holdout_tools_and_closure(
        date(2005, 6, 1)
    )

    # (strength, direction, n_pmids, study_count, total, completed, terminated).
    cases = {
        "disease a": ("weak", "supports", 50, 2, 0, 0, 0),
        "disease b": ("strong", "contradicts", 80, 5, 0, 0, 0),
    }
    for lower, (
        strength,
        direction,
        n_pmids,
        study_count,
        total,
        comp,
        term,
    ) in cases.items():
        allowed_diseases[lower] = (lower, "competitor")
        findings_local[lower] = {
            "literature": _make_lit(strength, n_pmids, study_count, direction),
            "clinical_trials": _make_ct(total, comp, term),
        }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "test_critique",
            "type": "tool_call",
        }
    )
    msg = await by_name["finalize_supervisor"].ainvoke(
        {
            "name": "finalize_supervisor",
            "args": {"summary": "DISCARDED", "blurbs": []},
            "id": "test_call",
            "type": "tool_call",
        }
    )

    expected = (
        "1. disease a — literature: weak, 50 PMIDs; "
        "trials: 0 total, 0 completed, 0 terminated.\n"
        "2. disease b — literature: strong, contradicts, 80 PMIDs; "
        "trials: 0 total, 0 completed, 0 terminated."
    )
    # disease b has stronger evidence but ranks LAST (contradicts), and is NOT in
    # any "Evidence gate exclusions" footer — it is surfaced as a negative.
    assert msg.artifact["summary"] == expected
    assert "Evidence gate exclusions" not in msg.artifact["summary"]
