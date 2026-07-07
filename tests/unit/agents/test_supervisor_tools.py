"""Unit tests for supervisor_tools — briefing rendering."""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
    TrialSignals,
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


@pytest.fixture(autouse=True)
def _mock_judge_interpretive():
    """Stub the isolated interpretive call to a no-op (returns None) so finalize_supervisor
    tests stay hermetic and exercise only the deterministic override/assembly logic. When the
    call returns None the enrich pass leaves the LLM-written interpretive fields untouched.
    Tests that assert interpretive output patch it themselves."""
    with patch(
        "indication_scout.agents.supervisor.supervisor_tools.judge_interpretive",
        new=AsyncMock(return_value=None),
    ):
        yield


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
        tools, _, _, _ = build_supervisor_tools(llm=llm, svc=svc, db=db)

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
        tools, _, _, _ = build_supervisor_tools(llm=llm, svc=svc, db=db)

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


# --- finalize_supervisor closure helpers (shared by the blurb-repair tests below) --


def _make_lit(
    strength: str,
    n_pmids: int,
    study_count: int,
    direction: str = "supports",
    is_observational: bool | None = None,
) -> LiteratureOutput:
    return LiteratureOutput(
        pmids=[str(1000000 + i) for i in range(n_pmids)],
        evidence_summary=EvidenceSummary(
            strength=strength,
            direction=direction,
            study_count=study_count,
            is_observational=is_observational,
            summary="",
            key_findings=[],
        ),
    )


def _make_ct(
    total: int,
    completed: int,
    terminated: int,
    signals: "TrialSignals | None" = None,
) -> ClinicalTrialsOutput:
    return ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=total),
        completed=CompletedTrialsResult(total_count=completed),
        terminated=TerminatedTrialsResult(total_count=terminated),
        signals=signals,
    )


def _finalize_tools_and_closure(cutoff: date | None = None):
    """Build supervisor tools (date_before=cutoff), return (tools_by_name,
    findings_local, allowed_diseases) seeded for direct closure manipulation of the
    finalize_supervisor blurb path."""
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
        tools, _, _, _ = build_supervisor_tools(
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


async def test_finalize_keeps_observational_when_fact_is_true():
    """The repair is one-directional: when is_observational is True, a legitimate
    'observational' claim must be left untouched (never add/alter design wording)."""
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["disease x"] = ("disease x", "competitor")
    findings_local["disease x"] = {
        "literature": _make_lit(
            "moderate", 5, 2, direction="supports", is_observational=True
        ),
        "clinical_trials": _make_ct(0, 0, 0),
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
            "args": {
                "summary": "Ranked repurposing signals:\n1. disease x — x",
                "blurbs": [
                    {
                        "disease": "disease x",
                        "key_risk": "Evidence is observational only",
                        "prose": "",
                        "literature": "Moderate, supports",
                    }
                ],
            },
            "id": "test_call",
            "type": "tool_call",
        }
    )

    blurbs = msg.artifact["blurbs"]
    assert len(blurbs) == 1
    assert blurbs[0]["key_risk"] == "Evidence is observational only"


async def test_finalize_repairs_false_no_phase3_stage():
    """Deterministic dev_stage override: when dev_stage=completed_phase3, finalize must rewrite a
    `stage` that falsely claims 'Phase 4 exploratory only / Phase 3 never initiated'."""
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["pcos"] = ("pcos", "competitor")
    findings_local["pcos"] = {
        "literature": _make_lit("strong", 10, 5, direction="supports"),
        "clinical_trials": _make_ct(
            50,
            22,
            0,
            signals=TrialSignals(
                highest_completed_phase="Phase 3",
                has_completed_phase3=True,
                has_completed_pure_phase3=True,
                completed_phase3_nct_ids=["NCT00068861"],
                dev_stage="completed_phase3",
            ),
        ),
    }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "test_critique",
            "type": "tool_call",
        }
    )
    # critique got [] (didn't cover this blurb) → finalize runs the fact critic. Mock it to
    # return the blurb UNCHANGED, proving the deterministic `stage` floor corrects stage even
    # when the critic doesn't touch it. (Hermetic: no live LLM call.)
    critic_out = json.dumps(
        {
            "ordering": "consistent",
            "blurbs": [
                {
                    "disease": "pcos",
                    "stage": "Phase 4 exploratory only (no dedicated development program)",
                    "prose": "",
                }
            ],
        }
    )
    with patch(
        "indication_scout.agents.supervisor.supervisor_tools.query_llm",
        new=AsyncMock(return_value=critic_out),
    ):
        msg = await by_name["finalize_supervisor"].ainvoke(
            {
                "name": "finalize_supervisor",
                "args": {
                    "summary": "Ranked repurposing signals:\n1. pcos — x",
                    "blurbs": [
                        {
                            "disease": "pcos",
                            "stage": "Phase 4 exploratory only (no dedicated development program)",
                            "prose": "",
                        }
                    ],
                },
                "id": "test_call",
                "type": "tool_call",
            }
        )

    b = msg.artifact["blurbs"][0]
    assert "exploratory only" not in b["stage"].lower()
    assert b["stage"] == "Phase 3 completed for this indication"


async def test_finalize_keeps_stage_when_no_completed_phase3():
    """dev_stage=exploratory_phase4_only renders the authoritative Phase 4 phrase — a
    legitimate 'exploratory only' stage is preserved (the override never invents a phase).
    """
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["disease y"] = ("disease y", "competitor")
    findings_local["disease y"] = {
        "literature": _make_lit("moderate", 4, 2, direction="supports"),
        "clinical_trials": _make_ct(
            5,
            2,
            0,
            signals=TrialSignals(
                highest_completed_phase="Phase 4",
                has_completed_phase3=False,
                dev_stage="exploratory_phase4_only",
            ),
        ),
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
            "args": {
                "summary": "Ranked repurposing signals:\n1. disease y — x",
                "blurbs": [
                    {
                        "disease": "disease y",
                        "stage": "Phase 4 exploratory only (no dedicated development program)",
                        "prose": "",
                    }
                ],
            },
            "id": "test_call",
            "type": "tool_call",
        }
    )

    b = msg.artifact["blurbs"][0]
    assert b["stage"] == (
        "Phase 4 exploratory only (post-approval off-label study; no dedicated "
        "development program for this indication)"
    )


async def test_finalize_repairs_false_stage_in_demotion_footer():
    """The T1DM bug: a demoted candidate's footer line falsely says 'Phase 4 exploratory only,
    no dedicated development program' while dev_stage=active_phase3. finalize must overwrite the
    false stage clause with the authoritative dev_stage phrase (incl. the active NCTs).
    """
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["type 1 diabetes mellitus"] = (
        "type 1 diabetes mellitus",
        "competitor",
    )
    findings_local["type 1 diabetes mellitus"] = {
        "literature": _make_lit("moderate", 6, 3, direction="supports"),
        "clinical_trials": _make_ct(
            16,
            4,
            0,
            signals=TrialSignals(
                has_active_phase3=True,
                active_phase3_nct_ids=["NCT06909006", "NCT06894784"],
                dev_stage="active_phase3",
            ),
        ),
    }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "c",
            "type": "tool_call",
        }
    )
    summary = (
        "Ranked repurposing signals:\n"
        "Demoted — approval relationship:\n"
        "- Type 1 Diabetes Mellitus — related_family (T2DM approved; T1DM is a distinct "
        "disease; Phase 4 exploratory only, no dedicated development program; "
        "moderate literature)"
    )
    msg = await by_name["finalize_supervisor"].ainvoke(
        {
            "name": "finalize_supervisor",
            "args": {"summary": summary, "blurbs": []},
            "id": "test_footer",
            "type": "tool_call",
        }
    )

    out = msg.artifact["summary"]
    assert "Phase 4 exploratory only" not in out
    assert "no dedicated development program" not in out
    assert "Active Phase 3 development on record" in out
    assert "NCT06909006" in out
    # The rest of the footer line (relationship, literature) is preserved.
    assert "related_family" in out
    assert "moderate literature" in out


async def test_finalize_leaves_footer_when_dev_stage_agrees():
    """A demoted candidate whose dev_stage is genuinely exploratory_phase4_only keeps its
    'Phase 4 exploratory only' footer text — the repair only fires on a program-stage.
    """
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["disease z"] = ("disease z", "competitor")
    findings_local["disease z"] = {
        "literature": _make_lit("weak", 2, 1, direction="supports"),
        "clinical_trials": _make_ct(
            3, 0, 0, signals=TrialSignals(dev_stage="exploratory_phase4_only")
        ),
    }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "c",
            "type": "tool_call",
        }
    )
    summary = (
        "Ranked repurposing signals:\n"
        "Demoted — approval relationship:\n"
        "- Disease Z — related_family (Phase 4 exploratory only, no dedicated development "
        "program; weak literature)"
    )
    msg = await by_name["finalize_supervisor"].ainvoke(
        {
            "name": "finalize_supervisor",
            "args": {"summary": summary, "blurbs": []},
            "id": "test_footer_agree",
            "type": "tool_call",
        }
    )

    assert "Phase 4 exploratory only" in msg.artifact["summary"]


async def test_finalize_repairs_false_stage_in_ranked_summary_line():
    """The ranked-line analog of the footer leak: a ranked summary line falsely says 'no formal
    development program' while dev_stage=active_phase3. finalize must overwrite the false clause
    with the authoritative dev_stage phrase, keeping the rest of the line intact."""
    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()

    allowed_diseases["type 1 diabetes mellitus"] = (
        "type 1 diabetes mellitus",
        "competitor",
    )
    findings_local["type 1 diabetes mellitus"] = {
        "literature": _make_lit("moderate", 6, 3, direction="supports"),
        "clinical_trials": _make_ct(
            16,
            4,
            0,
            signals=TrialSignals(
                has_active_phase3=True,
                active_phase3_nct_ids=["NCT06909006"],
                dev_stage="active_phase3",
            ),
        ),
    }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "c",
            "type": "tool_call",
        }
    )
    summary = (
        "Ranked repurposing signals:\n"
        "1. Type 1 Diabetes Mellitus — moderate literature; no formal development program; "
        "watch nothing"
    )
    msg = await by_name["finalize_supervisor"].ainvoke(
        {
            "name": "finalize_supervisor",
            "args": {
                "summary": summary,
                "blurbs": [
                    {"disease": "type 1 diabetes mellitus", "stage": "x", "prose": "p"}
                ],
            },
            "id": "test_ranked",
            "type": "tool_call",
        }
    )

    out = msg.artifact["summary"]
    assert "no formal development program" not in out
    assert "Active Phase 3 development on record" in out
    assert "NCT06909006" in out
    assert "moderate literature" in out


async def test_finalize_enrich_overwrites_interpretive_fields():
    """The enrich pass: when judge_interpretive returns a judgment, finalize overwrites the
    blurb's blocker/key_risk/verdict/prose from it (one source of truth — the LLM no longer
    authors them). Patches judge_interpretive to a known value (the autouse fixture is
    overridden here)."""
    from indication_scout.services.judge_interpretive import InterpretiveJudgment

    by_name, findings_local, allowed_diseases = _finalize_tools_and_closure()
    allowed_diseases["pcos"] = ("pcos", "competitor")
    findings_local["pcos"] = {
        "literature": _make_lit("strong", 10, 5, direction="supports"),
        "clinical_trials": _make_ct(
            50,
            22,
            0,
            signals=TrialSignals(
                highest_completed_phase="Phase 3",
                has_completed_phase3=True,
                completed_phase3_nct_ids=["NCT1"],
                dev_stage="completed_phase3",
                active_programs="Phase 3 recruiting (NCT1)",
            ),
        ),
    }

    await by_name["critique_ranking"].ainvoke(
        {
            "name": "critique_ranking",
            "args": {"blurbs": []},
            "id": "c",
            "type": "tool_call",
        }
    )

    interp = InterpretiveJudgment(
        blocker="Regulatory differentiation burden",
        key_risk="May not beat approved family member",
        verdict="Live but bottlenecked",
        prose="A completed pivotal program exists with active follow-on trials. The "
        "remaining question is commercial differentiation.",
    )
    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.query_llm",
            new=AsyncMock(return_value=json.dumps({"ordering": "ok", "blurbs": []})),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.judge_interpretive",
            new=AsyncMock(return_value=interp),
        ),
    ):
        msg = await by_name["finalize_supervisor"].ainvoke(
            {
                "name": "finalize_supervisor",
                "args": {
                    "summary": "Ranked repurposing signals:\n1. pcos — x",
                    "blurbs": [
                        {
                            "disease": "pcos",
                            "stage": "draft",
                            "verdict": "LLM-WROTE-THIS-SHOULD-BE-REPLACED",
                            "blocker": "LLM-WROTE-THIS",
                            "key_risk": "LLM-WROTE-THIS",
                            "prose": "LLM wrote this and it should be replaced.",
                        }
                    ],
                },
                "id": "f",
                "type": "tool_call",
            }
        )

    b = msg.artifact["blurbs"][0]
    assert b["verdict"] == "Live but bottlenecked"
    assert b["blocker"] == "Regulatory differentiation burden"
    assert b["key_risk"] == "May not beat approved family member"
    assert b["prose"].startswith("A completed pivotal program")
    # stage still came from the dev_stage override (not the LLM draft).
    assert "Phase 3 completed" in b["stage"]


# ------------------------------------------------------------------
# _literature_oneliner — class_level basis must not read as direct drug strength
# ------------------------------------------------------------------

from indication_scout.agents.supervisor.supervisor_tools import (  # noqa: E402
    _literature_oneliner,
)


def test_literature_oneliner_drug_specific_renders_strength_direction_design():
    es = EvidenceSummary(
        strength="strong",
        direction="supports",
        is_observational=False,
        evidence_basis="drug_specific",
    )
    assert _literature_oneliner(es) == "strong, supports, RCT-backed / controlled"


def test_literature_oneliner_class_level_states_basis_not_strength():
    """The Parkinson bug: class-level evidence must NOT render 'strong, ..., RCT-backed'."""
    es = EvidenceSummary(
        strength="none",
        direction="none",
        is_observational=None,
        evidence_basis="class_level",
    )
    assert (
        _literature_oneliner(es)
        == "class-level signal (no direct evidence for this drug)"
    )


def test_literature_oneliner_none_summary_returns_none():
    assert _literature_oneliner(None) == "None"
