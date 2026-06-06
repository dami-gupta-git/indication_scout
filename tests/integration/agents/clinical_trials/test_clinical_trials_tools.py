"""Integration test for the clinical trials tool layer.

Hits real NCBI (MeSH resolver) and real ClinicalTrials.gov.

Verifies that the tool layer resolves the indication to a MeSH preferred term
and forwards it to the client, which then uses CT.gov's server-side
`AREA[ConditionMeshTerm]` filter so that returned counts and trials are both
restricted to the systemic-hypertension descriptor (D006973).

Expected values verified live on 2026-04-25 with:
  drug=semaglutide, indication=hypertension, date_before=2025-01-01
"""

import logging
from datetime import date

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)
_HYPERTENSION_MESH_ID = "D006973"


async def test_search_trials_tool_uses_server_side_mesh_filter():
    """search_trials tool: resolves hypertension → ('D006973', 'Hypertension')
    and forwards the preferred term to the client, which builds
    `AREA[ConditionMeshTerm]"Hypertension"` server-side.

    This is a live integration test (real NCBI + real CT.gov), so exact trial
    counts drift as new trials register. Assert the stable invariants the test
    exists to prove, not a pinned count:
      - the descriptor resolves to D006973 / "Hypertension"
      - the client forwards it as the server-side AREA[ConditionMeshTerm] filter
      - every returned trial carries that descriptor (direct mesh_conditions OR
        mesh_ancestors — ancestor matches are accepted because the server-side
        AREA filter matches ancestors too; see the KNOWN BUG note on
        clinical_trials._mesh_cond)
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    search_trials = next(t for t in tools if t.name == "search_trials")

    msg = await search_trials.ainvoke(
        {
            "name": "search_trials",
            "args": {"drug": "semaglutide", "indication": "hypertension"},
            "id": "test-call-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, SearchTrialsResult)

    # Content string leads with the resolved pair; count is live-variable so
    # only assert the stable prefix.
    assert "Search for semaglutide × hypertension:" in msg.content

    # by_status is an exhaustive partition of the returned trials.
    assert set(result.by_status) == {
        "RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "WITHDRAWN",
        "UNKNOWN",
    }
    assert sum(result.by_status.values()) == result.total_count

    # Core invariant: every returned trial is filtered to the hypertension
    # descriptor (D006973), via direct conditions or ancestors.
    assert result.trials, "expected at least one semaglutide × hypertension trial"
    for t in result.trials:
        mesh_ids = {m.id for m in t.mesh_conditions} | {
            m.id for m in t.mesh_ancestors
        }
        assert _HYPERTENSION_MESH_ID in mesh_ids, (
            f"{t.nct_id} not filtered to {_HYPERTENSION_MESH_ID}; "
            f"mesh ids {sorted(mesh_ids)}"
        )


# ------------------------------------------------------------------
# search_trials — date_before cutoff
# ------------------------------------------------------------------


async def test_search_trials_tool_respects_date_before_cutoff():
    """search_trials with date_before forwards the cutoff to the client, which
    excludes any trial whose start_date is on/after the cutoff.

    Pair: dupilumab × eosinophilic esophagitis. FDA approved this repurposing
    on 2022-05-20, so a cutoff of that date should still surface the
    pre-approval Phase 3 pivotal trial(s) that supported the approval. This
    test catches a regression where the cutoff filters out all Phase 3
    trials in a window where they are known to exist.
    """
    cutoff = date(2022, 5, 20)
    tools = build_clinical_trials_tools(date_before=cutoff)
    search_trials = next(t for t in tools if t.name == "search_trials")

    msg = await search_trials.ainvoke(
        {
            "name": "search_trials",
            "args": {
                "drug": "dupilumab",
                "indication": "eosinophilic esophagitis",
            },
            "id": "test-cutoff-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, SearchTrialsResult)

    # Invariant: every returned trial started strictly before the cutoff.
    # CT.gov start_date is "YYYY-MM-DD" or "YYYY-MM" — both compare
    # lexicographically against an ISO cutoff prefix.
    cutoff_iso = cutoff.isoformat()
    for t in result.trials:
        assert t.start_date is not None, f"{t.nct_id} missing start_date"
        assert (
            t.start_date < cutoff_iso
        ), f"{t.nct_id} start_date={t.start_date} >= cutoff {cutoff_iso}"

    # At least one Phase 3 trial must survive the cutoff — the dupilumab
    # eosinophilic esophagitis approval was supported by Phase 3 trials that
    # started before 2022-05-20.
    phase_3_trials = [t for t in result.trials if "Phase 3" in t.phase]
    assert len(phase_3_trials) >= 1, (
        "expected at least one Phase 3 dupilumab × eosinophilic esophagitis "
        f"trial with start_date < {cutoff_iso}; got phases "
        f"{[t.phase for t in result.trials]}"
    )


# ------------------------------------------------------------------
# check_fda_approval — hits real ChEMBL, openFDA, and Anthropic
# ------------------------------------------------------------------


# Verified live on 2026-04-25: ChEMBL resolves semaglutide to 9 drug names —
# the four brand/INN forms plus ChEMBL/Novo Nordisk research codes and the
# Spanish INN. Lock the full set so a regression in the resolver surfaces here.
_SEMAGLUTIDE_EXPECTED_NAMES = {
    "semaglutide",
    "semaglutida",
    "ozempic",
    "wegovy",
    "rybelsus",
    "nn-9535",
    "nn9535",
    "nnc 0113-0217",
    "nnc-0113-0217",
}


@pytest.mark.parametrize(
    "indication,is_approved,matched_indication,expected_content",
    [
        (
            "NASH",
            True,
            "NASH",
            "FDA approval check for semaglutide × NASH: APPROVED (checked 9 drug names)",
        ),
        (
            "alzheimer disease",
            False,
            None,
            "FDA approval check for semaglutide × alzheimer disease: "
            "not on FDA label (checked 9 drug names)",
        ),
    ],
)
async def test_check_fda_approval_semaglutide(
    indication, is_approved, matched_indication, expected_content
):
    """End-to-end: ChEMBL resolution, openFDA label fetch, and LLM approval
    extraction for semaglutide.

    - NASH: semaglutide IS FDA-approved → is_approved=True.
    - alzheimer disease: label exists, indication not on it → is_approved=False.
    matched_indication is the caller's input verbatim when is_approved is True,
    otherwise None.

    Numbers verified live on 2026-04-25: ChEMBL resolves to 9 drug names for
    semaglutide.
    """
    tools = build_clinical_trials_tools(date_before=None)
    check_fda_approval = next(t for t in tools if t.name == "check_fda_approval")

    msg = await check_fda_approval.ainvoke(
        {
            "name": "check_fda_approval",
            "args": {"drug": "semaglutide", "indication": indication},
            "id": f"it-fda-{indication}",
            "type": "tool_call",
        }
    )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is is_approved
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == matched_indication
    assert set(msg.artifact.drug_names_checked) == _SEMAGLUTIDE_EXPECTED_NAMES
    assert msg.content == expected_content


# ------------------------------------------------------------------
# get_completed — hits real NCBI (MeSH), CT.gov, and ChEMBL (alias filter)
# ------------------------------------------------------------------


async def test_get_completed_tool_semaglutide_hypertension():
    """get_completed tool: resolves indication → MeSH preferred term, queries
    CT.gov for COMPLETED trials filtered by `AREA[ConditionMeshTerm]`, and
    drops trials that don't intervene with the drug (alias filter).

    Content string format:
      "Completed for {drug} × {indication}: {N} total[; top 50 shown][; dropped {K} ...]"
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_completed = next(t for t in tools if t.name == "get_completed")

    msg = await get_completed.ainvoke(
        {
            "name": "get_completed",
            "args": {"drug": "semaglutide", "indication": "hypertension"},
            "id": "test-completed-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, CompletedTrialsResult)

    # TODO: fill in expected total_count for semaglutide × hypertension
    # completed trials with cutoff 2025-01-01
    # assert result.total_count == ...
    # assert "Completed for semaglutide × hypertension:" in msg.content

    # TODO: fill in expected NCT IDs and per-trial fields for spot-checks
    # nct_ids = {t.nct_id for t in result.trials}
    # assert nct_ids == {...}
    #
    # spot = next(t for t in result.trials if t.nct_id == "...")
    # assert spot.phase == "..."
    # assert spot.overall_status == "COMPLETED"
    # assert spot.enrollment == ...
    # assert spot.sponsor == "..."
    # assert spot.start_date == "..."

    # Every kept trial must carry the indication's MeSH descriptor in
    # mesh_conditions or mesh_ancestors (server-side AREA filter guarantee).
    for t in result.trials:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert _HYPERTENSION_MESH_ID in (cond_ids | anc_ids), (
            f"{t.nct_id} missing {_HYPERTENSION_MESH_ID} in both "
            f"mesh_conditions={cond_ids} and mesh_ancestors={anc_ids}"
        )
        assert t.overall_status == "COMPLETED"


# ------------------------------------------------------------------
# get_completed — date_before cutoff
# ------------------------------------------------------------------


async def test_get_completed_tool_respects_date_before_cutoff():
    """get_completed with date_before forwards the cutoff to the client, which
    excludes any completed trial whose start_date is on/after the cutoff.

    Pair: dupilumab × eosinophilic esophagitis. FDA approved this repurposing
    on 2022-05-20, so a cutoff of that date should still surface at least one
    completed trial that started before the cutoff (the Phase 3 pivotal trials
    supporting the approval started in 2017–2019). This catches a regression
    where the cutoff filters out all completed trials in a window where they
    are known to exist.
    """
    cutoff = date(2022, 5, 20)
    tools = build_clinical_trials_tools(date_before=cutoff)
    get_completed = next(t for t in tools if t.name == "get_completed")

    msg = await get_completed.ainvoke(
        {
            "name": "get_completed",
            "args": {
                "drug": "dupilumab",
                "indication": "eosinophilic esophagitis",
            },
            "id": "test-completed-cutoff-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, CompletedTrialsResult)

    # Invariant: every returned trial started strictly before the cutoff.
    # CT.gov start_date is "YYYY-MM-DD" or "YYYY-MM" — both compare
    # lexicographically against an ISO cutoff prefix.
    cutoff_iso = cutoff.isoformat()
    for t in result.trials:
        assert t.start_date is not None, f"{t.nct_id} missing start_date"
        assert (
            t.start_date < cutoff_iso
        ), f"{t.nct_id} start_date={t.start_date} >= cutoff {cutoff_iso}"

    # At least one completed trial must survive the cutoff. The dupilumab ×
    # eosinophilic esophagitis approval was supported by Phase 3 trials that
    # started before 2022-05-20 and had completed by the approval date.
    assert len(result.trials) >= 1, (
        f"expected at least one completed dupilumab × eosinophilic esophagitis "
        f"trial with start_date < {cutoff_iso}; got {result.total_count} total, "
        f"{len(result.trials)} shown"
    )


# ------------------------------------------------------------------
# get_terminated — hits real NCBI (MeSH), CT.gov, and ChEMBL (alias filter)
# ------------------------------------------------------------------


async def test_get_terminated_tool_metformin_hypertension():
    """get_terminated tool: resolves indication → MeSH preferred term, queries
    CT.gov for TERMINATED trials filtered by `AREA[ConditionMeshTerm]`, drops
    non-intervention trials via the alias filter, and computes a
    safety/efficacy stop-category count over the shown set.

    Content string format:
      "Terminated for {drug} × {indication}: {N} total ({K} safety/efficacy in shown set)..."
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_terminated = next(t for t in tools if t.name == "get_terminated")

    msg = await get_terminated.ainvoke(
        {
            "name": "get_terminated",
            "args": {"drug": "metformin", "indication": "hypertension"},
            "id": "test-terminated-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, TerminatedTrialsResult)

    # TODO: fill in expected total_count for metformin × hypertension
    # terminated trials with cutoff 2025-01-01
    # assert result.total_count == ...
    # assert "Terminated for metformin × hypertension:" in msg.content
    # assert "safety/efficacy in shown set" in msg.content

    # TODO: fill in expected NCT IDs and per-trial fields (including
    # why_stopped) for spot-checks
    # nct_ids = {t.nct_id for t in result.trials}
    # assert nct_ids == {...}
    #
    # spot = next(t for t in result.trials if t.nct_id == "...")
    # assert spot.overall_status == "TERMINATED"
    # assert spot.why_stopped is not None
    # assert "..." in spot.why_stopped.lower()

    # Every kept trial must carry the indication's MeSH descriptor and have
    # status TERMINATED.
    for t in result.trials:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert _HYPERTENSION_MESH_ID in (cond_ids | anc_ids), (
            f"{t.nct_id} missing {_HYPERTENSION_MESH_ID} in both "
            f"mesh_conditions={cond_ids} and mesh_ancestors={anc_ids}"
        )
        assert t.overall_status == "TERMINATED"


# ------------------------------------------------------------------
# get_landscape — hits real NCBI (MeSH) and CT.gov
# ------------------------------------------------------------------


async def test_get_landscape_tool_gastroparesis():
    """get_landscape tool: resolves indication → MeSH preferred term and
    forwards it to the client. Mirrors the data-source-level assertions in
    tests/integration/data_sources/test_clinical_trials.py::test_get_landscape
    but driven through the tool layer (no date_before — landscape values
    verified live without a cutoff).

    Content string format:
      "Landscape for {indication}: {N} competitors"
    """
    tools = build_clinical_trials_tools(date_before=None)
    get_landscape = next(t for t in tools if t.name == "get_landscape")

    msg = await get_landscape.ainvoke(
        {
            "name": "get_landscape",
            "args": {"indication": "gastroparesis"},
            "id": "test-landscape-1",
            "type": "tool_call",
        }
    )

    result = msg.artifact
    assert isinstance(result, IndicationLandscape)
    assert msg.content == "Landscape for gastroparesis: 10 competitors"

    # total trial count — gastroparesis has ~300+ trials on CT.gov
    assert result.total_trial_count > 200

    # tool requests top_n=10 from the client
    assert len(result.competitors) == 10

    # phase distribution sanity bounds (verified live in data-source test)
    assert 10 < result.phase_distribution["Phase 2"] < 100
    assert 2 < result.phase_distribution["Phase 3"] < 50
    assert 1 <= result.phase_distribution["Phase 4"] < 10

    # Recent starts include Vanda Pharmaceuticals' Tradipitant trial
    assert len(result.recent_starts) >= 1
    [tradipitant] = [rs for rs in result.recent_starts if rs.nct_id == "NCT06836557"]
    assert tradipitant.nct_id == "NCT06836557"
    assert tradipitant.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant.drug == "Tradipitant"
    assert tradipitant.phase == "Phase 3"

    # Vaccine exclusion — competitor names must not contain vaccine keywords
    for c in result.competitors:
        name_lower = c.drug_name.lower()
        assert "vaccine" not in name_lower
        assert "vax" not in name_lower
        assert "immuniz" not in name_lower

    # Tradipitant competitor entry — Phase 3, two trials, most recent 2024-01-09
    [tradipitant_comp] = [
        c
        for c in result.competitors
        if c.sponsor == "Vanda Pharmaceuticals" and c.drug_name == "Tradipitant"
    ]
    assert tradipitant_comp.drug_type == "Drug"
    assert tradipitant_comp.max_phase == "Phase 3"
    assert tradipitant_comp.trial_count == 2
    assert tradipitant_comp.statuses == {"COMPLETED", "RECRUITING"}
    assert tradipitant_comp.total_enrollment == 1092
    assert tradipitant_comp.most_recent_start == "2024-01-09"
