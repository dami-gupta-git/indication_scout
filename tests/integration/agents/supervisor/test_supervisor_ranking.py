"""Integration tests for the supervisor RANKING judgment and the `watch` NCT rule.

These exercise the REAL supervisor.txt prompt against the live LLM in isolation (one query_llm
call with supervisor.txt as the system prompt) — not the full agent loop. They lock in two
prompt-only changes made when ranking moved from a fixed tier ladder to LLM judgment:

  1. RANKING is the LLM's judgment over the labels it sees (dev_stage tier, literature
     strength/direction/design/evidence_basis, closure). Mirrors scratch/ranking_judgment_harness.py.
  2. The `watch` blurb field must cite only THIS candidate's own NCTs — never another
     candidate's. Mirrors scratch/watch_nct_harness.py (the Parkinson × T1D-NCT leak).

Candidates use invented disease names so the model ranks on the LABELS, not real-world priors.
Tagged approval_aware (these guard the approval-aware ranking/relevance behavior).
"""

import json
import logging
import re
from pathlib import Path

import pytest

from indication_scout.services.llm import query_llm

logger = logging.getLogger(__name__)

_PROMPTS_DIR = (
    Path(__file__).parents[4] / "src" / "indication_scout" / "prompts"
)
SUPERVISOR_PROMPT = (_PROMPTS_DIR / "supervisor.txt").read_text()


def _candidate_block(c: dict) -> str:
    """Render one candidate the way the supervisor sees it (literature header + derived signals)."""
    lines = [
        f"### Candidate: {c['disease']}",
        (
            f"Literature for DRUG × {c['disease']}: {c.get('pmids', 0)} PMIDs, "
            f"strength={c['strength']}, direction={c['direction']}, "
            f"study_design={c['design']}, evidence_basis={c['basis']}."
        ),
        "DERIVED SIGNALS (authoritative facts — relevant trials only):",
        f"  highest_completed_phase: {c.get('highest_completed_phase', 'none')}",
        f"  completed_phase_3: {c.get('completed_phase_3', 'no')}",
        f"  active_phase_3: {c.get('active_phase_3', 'no')}",
        f"  relevant_phase3_terminated_for_cause: {c.get('terminated_for_cause', 'no')}",
        f"  dev_stage: {c['dev_stage']}",
        f"Sub-agent closure verdict (trust, do not re-judge): closure={c.get('closure', 'live')}",
    ]
    return "\n".join(lines)


def _rank_task(candidates: list[dict]) -> str:
    blocks = "\n\n".join(_candidate_block(c) for c in candidates)
    names = [c["disease"] for c in candidates]
    return (
        "You have already investigated the following candidates for drug DRUG this run "
        "(both analyze_literature and analyze_clinical_trials ran for each). Using the RANKING "
        "guidance in your instructions, rank them best-repurposing-signal first.\n\n"
        f"{blocks}\n\n"
        f"Candidates to rank: {names}\n\n"
        'Respond with ONLY a JSON object: {"order": ["<disease>", ...]} listing every candidate '
        "exactly once, best first. No prose, no fences."
    )


def _parse_order(text: str) -> list[str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return list(json.loads(m.group(0)).get("order", []))
    except json.JSONDecodeError:
        return []


def _before(order: list[str], a: str, b: str) -> bool:
    return a in order and b in order and order.index(a) < order.index(b)


# --- RANKING cases (label, candidates, expect predicate over the returned order) ---

_RANK_CASES = [
    (
        # genuine drug-specific supportive literature beats a higher-stage approved-basis candidate
        "drug_specific_supports_beats_higher_stage_approved",
        [
            {
                "disease": "Alphosis",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "none",
                "direction": "none",
                "design": "undetermined",
                "basis": "approved",
            },
            {
                "disease": "Betalgia",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "strong",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        lambda o: _before(o, "Betalgia", "Alphosis"),
    ),
    (
        # the clean supportive candidate is #1; closed + contradicts both sink below it
        "closed_and_contradicts_both_below_clean",
        [
            {
                "disease": "Epsilitis",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "moderate",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
            {
                "disease": "Zetawasting",
                "dev_stage": "phase3_terminated_for_cause",
                "terminated_for_cause": "yes",
                "highest_completed_phase": "none",
                "strength": "weak",
                "direction": "mixed",
                "design": "observational",
                "basis": "drug_specific",
                "closure": "closed — safety termination",
            },
            {
                "disease": "Etacline",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "strong",
                "direction": "contradicts",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        lambda o: o
        and o[0] == "Epsilitis"
        and _before(o, "Epsilitis", "Zetawasting")
        and _before(o, "Epsilitis", "Etacline"),
    ),
]


@pytest.mark.approval_aware
@pytest.mark.parametrize(
    "label,candidates,expect", _RANK_CASES, ids=[c[0] for c in _RANK_CASES]
)
async def test_supervisor_ranking_judgment(label, candidates, expect):
    """The supervisor prompt alone (no critique_ranking) ranks correctly from the labels."""
    response = await query_llm(_rank_task(candidates), system=SUPERVISOR_PROMPT)
    order = _parse_order(response)
    assert order, f"{label}: no parseable order from: {response!r}"
    assert sorted(order) == sorted(
        c["disease"] for c in candidates
    ), f"{label}: order is not a permutation of the candidates: {order}"
    assert expect(order), f"{label}: wrong order {order}"


# --- WATCH-NCT case: a candidate's watch must cite only its own NCTs ---

_WATCH_CANDIDATES = [
    {
        "disease": "Alphadiabetes",
        "own_ncts": ["NCT11110001", "NCT11110002", "NCT11110003", "NCT11110004"],
        "lit": (
            "Literature: 6 PMIDs, strength=moderate, direction=supports, "
            "study_design=rct_or_controlled, evidence_basis=drug_specific."
        ),
        "ct_summary": (
            "Active Phase 3 development. Recruiting Phase 3: NCT11110001 (CV outcomes), "
            "NCT11110002 (adjunct), NCT11110003 (combination), NCT11110004 (not yet recruiting)."
        ),
        "active_programs": "4 Phase 3 recruiting (NCT11110001, NCT11110002, NCT11110003, NCT11110004)",
    },
    {
        "disease": "Betasteatosis",
        "own_ncts": ["NCT22220001", "NCT22220002"],
        "lit": (
            "Literature: 0 PMIDs, strength=none, direction=none, study_design=undetermined, "
            "evidence_basis=approved."
        ),
        "ct_summary": (
            "Phase 2 completed, no Phase 3. Completed: NCT22220001 (Phase 2), NCT22220002 "
            "(Phase 2). No active pivotal program."
        ),
        "active_programs": "None active",
    },
    {
        "disease": "Gammakinson",  # Parkinson-shape crux: no active trial → watch must be empty
        "own_ncts": ["NCT33330001"],
        "lit": (
            "Literature: 0 PMIDs, strength=none, direction=none, study_design=undetermined, "
            "evidence_basis=class_level."
        ),
        "ct_summary": (
            "Early-phase only, no completed pivotal readout. Single Phase 2 study NCT33330001 "
            "has unknown status. No active or recruiting trials."
        ),
        "active_programs": "None active",
    },
]


def _watch_task() -> str:
    blocks = "\n\n".join(
        f"### Candidate: {c['disease']}\n{c['lit']}\n"
        f"clinical_trials.summary: {c['ct_summary']}\n"
        f"active_programs: {c['active_programs']}"
        for c in _WATCH_CANDIDATES
    )
    names = [c["disease"] for c in _WATCH_CANDIDATES]
    return (
        "You investigated these candidates for drug DRUG this run (both sub-agent calls ran for "
        "each). Write the per-candidate blurb fields, especially `watch`, following your "
        "instructions.\n\n"
        f"{blocks}\n\n"
        f"Candidates: {names}\n\n"
        'Respond with ONLY a JSON object: {"blurbs": [{"disease": "...", "watch": "..."}, ...]} '
        "with one entry per candidate. watch is a string (may be empty)."
    )


def _ncts_in(text: str) -> set[str]:
    return set(re.findall(r"NCT\d+", text or ""))


def _parse_blurbs(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return list(json.loads(m.group(0)).get("blurbs", []))
    except json.JSONDecodeError:
        return []


@pytest.mark.approval_aware
async def test_supervisor_watch_only_cites_own_ncts():
    """A candidate's `watch` line must never cite another candidate's NCT (the Parkinson × T1D
    leak), and a candidate with no active trial must leave `watch` empty."""
    response = await query_llm(_watch_task(), system=SUPERVISOR_PROMPT)
    blurbs = _parse_blurbs(response)
    assert blurbs, f"no parseable blurbs from: {response!r}"
    own_by_disease = {
        c["disease"].lower(): set(c["own_ncts"]) for c in _WATCH_CANDIDATES
    }
    for b in blurbs:
        disease = (b.get("disease") or "").strip().lower()
        cited = _ncts_in(b.get("watch", ""))
        own = own_by_disease.get(disease, set())
        foreign = cited - own
        assert not foreign, (
            f"{b.get('disease')}: watch cites foreign NCT(s) {sorted(foreign)} "
            f"(own={sorted(own)})"
        )
        if disease == "gammakinson":
            assert not cited, (
                f"Gammakinson has no active trial — watch should be empty, "
                f"got {sorted(cited)}"
            )
