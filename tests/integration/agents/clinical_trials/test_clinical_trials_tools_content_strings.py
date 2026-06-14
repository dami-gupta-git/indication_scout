"""Integration tests for the clinical_trials tool content strings.

Background: tools use `@tool(response_format="content_and_artifact")`. The LLM
sees only the `content` string — the artifact stays Python-side. These tests
guard the prompt-side contract (see `agent_data_contracts.md` at the project
root): they assert the LLM-visible content string carries per-trial NCT ids,
phase, and title. search_trials still renders a mesh column; the completed and
terminated relevance-classification views render interventions (drugs) and a
truncated brief_summary instead (and, for terminated, classified stop reasons),
with MeSH dropped from those views.

Bug being guarded against: with content-only-aggregates strings, the
sub-agent's prose claimed "no Phase 3" while Phase 3 trials existed in the
artifact (sildenafil × diastolic heart failure, RELAX trial NCT00763867).

Hits real NCBI (MeSH resolver), real ClinicalTrials.gov, and ChEMBL (alias
filter inside the sub-agent's tools).

Expected values verified live on 2026-05-01 with date_before=2025-01-01:
  drug=sildenafil, indication=diastolic heart failure → 4 completed
  drug=sildenafil, indication=stroke → 2 terminated
"""

import logging
from datetime import date

from indication_scout.agents._trial_formatting import (
    _borda_rank_by_enrollment_and_recency,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
    build_clinical_trials_tools,
)
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)


# ------------------------------------------------------------------
# search_trials — sildenafil × diastolic heart failure
# ------------------------------------------------------------------


async def test_search_trials_content_string_sildenafil_hfpef():
    """search_trials renders status column, phase distribution, and per-row
    table with NCT id + phase + status + mesh + title.

    Sildenafil × diastolic heart failure has 4 trials, all COMPLETED — so
    by_status counts are all zero (search excludes COMPLETED/TERMINATED
    from its breakdown but returns the full pair total).
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    search_trials = next(t for t in tools if t.name == "search_trials")

    msg = await search_trials.ainvoke(
        {
            "name": "search_trials",
            "args": {"drug": "sildenafil", "indication": "diastolic heart failure"},
            "id": "test-search-1",
            "type": "tool_call",
        }
    )

    artifact = msg.artifact
    assert isinstance(artifact, SearchTrialsResult)
    assert artifact.total_count == 4
    assert artifact.by_status == {
        "RECRUITING": 0,
        "ACTIVE_NOT_RECRUITING": 0,
        "WITHDRAWN": 0,
        "UNKNOWN": 0,
    }
    assert len(artifact.trials) == 4

    expected_content = (
        "Search for sildenafil × diastolic heart failure: 4 trials "
        "(recruiting=0, active=0, withdrawn=0, unknown=0)\n"
        "Phase distribution (shown): Phase 4=1, Phase 3=2, Phase 2/Phase 3=1\n"
        "Trials shown (top 20 by enrollment):\n"
        "  NCT00763867 | Phase 3          | COMPLETED | "
        "mesh: Heart Failure; Heart Failure, Diastolic | "
        "Evaluating the Effectiveness of Sildenafil at Improving Health Outcomes "
        "and Exercise Ability in People With Diastolic Heart Failure (The RELAX Study)\n"
        "  NCT01046838 | Phase 4          | COMPLETED | "
        "mesh: Heart Failure, Diastolic | "
        "SIDAMI - Sildenafil and Diastolic Dysfunction After Acute Myocardial "
        "Infarction (AMI)\n"
        "  NCT01726049 | Phase 3          | COMPLETED | "
        "mesh: Heart Failure, Diastolic; Hypertension, Pulmonary | "
        "Sildenafil in HFpEF (Heart Failure With Preserved Ejection Fraction) and PH\n"
        "  NCT01156636 | Phase 2/Phase 3  | COMPLETED | "
        "mesh: Hypertension, Pulmonary; Heart Failure, Diastolic | "
        "Phosphodiesterase-5 (PDE5) Inhibition and Pulmonary Hypertension in "
        "Diastolic Heart Failure"
    )
    assert msg.content == expected_content


# ------------------------------------------------------------------
# get_completed — sildenafil × diastolic heart failure
# ------------------------------------------------------------------


async def test_get_completed_content_string_sildenafil_hfpef():
    """get_completed renders the un-capped relevance-classification view: per-row
    NCT id, phase, interventions (drugs), title, and truncated brief_summary.
    MeSH is dropped from this view (it over-recalls subtypes); the agent judges
    relevance from the drugs/title/summary instead. No status column (all
    COMPLETED), no dates (delegated to the supervisor view).

    Trials are ordered by enrollment desc in the artifact (216, 70, 52, 44),
    and the view is un-capped, so all 4 are rendered in that order.
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_completed = next(t for t in tools if t.name == "get_completed")

    msg = await get_completed.ainvoke(
        {
            "name": "get_completed",
            "args": {"drug": "sildenafil", "indication": "diastolic heart failure"},
            "id": "test-completed-1",
            "type": "tool_call",
        }
    )

    artifact = msg.artifact
    assert isinstance(artifact, CompletedTrialsResult)
    assert artifact.total_count == 4
    assert len(artifact.trials) == 4
    # Spot-check the RELAX trial (motivated this test fixture).
    relax = next(t for t in artifact.trials if t.nct_id == "NCT00763867")
    assert relax.phase == "Phase 3"
    assert relax.enrollment == 216
    assert relax.start_date == "2008-09"
    assert relax.completion_date == "2012-09"
    assert [(m.id, m.term) for m in relax.mesh_conditions] == [
        ("D006333", "Heart Failure"),
        ("D054144", "Heart Failure, Diastolic"),
    ]

    expected_content = (
        "Completed for sildenafil × diastolic heart failure: 4 total\n"
        "Judge relevance from the drugs (is sildenafil the studied drug?), title, and "
        "summary — is the disease THIS indication, not a distinct subtype?\n"
        "Phase distribution (shown): Phase 4=1, Phase 3=2, Phase 2/Phase 3=1\n"
        "Trials shown (all 4 — classify EVERY one):\n"
        "  NCT00763867 | Phase 3          | drugs: Placebo; Sildenafil | "
        "Evaluating the Effectiveness of Sildenafil at Improving Health Outcomes "
        "and Exercise Ability in People With Diastolic Heart Failure (The RELAX Study) | "
        "summary: Diastolic heart failure (DHF), which affects older individuals and "
        "women at a disproportionate rate, is a condition that can lead to shortness "
        "of breath and flu…\n"
        "  NCT01046838 | Phase 4          | drugs: Sildenafil; Placebo | "
        "SIDAMI - Sildenafil and Diastolic Dysfunction After Acute Myocardial "
        "Infarction (AMI) | "
        "summary: In patients with Doppler echocardiographic signs of elevated LV "
        "filling pressures despite preserved LV systolic function after AMI treatment "
        "with the phosphodie…\n"
        "  NCT01726049 | Phase 3          | drugs: Sildenafil; Placebo | "
        "Sildenafil in HFpEF (Heart Failure With Preserved Ejection Fraction) and PH | "
        "summary: Aim of the study is to investigate whether Sildenafil treatment "
        "results in a reduction of pulmonary artery pressure without decrease of "
        "cardiac output (CO) and…\n"
        "  NCT01156636 | Phase 2/Phase 3  | drugs: Sildenafil; Placebo | "
        "Phosphodiesterase-5 (PDE5) Inhibition and Pulmonary Hypertension in "
        "Diastolic Heart Failure | "
        "summary: Prevalence of heart failure (HF) with left ventricular (LV) "
        "diastolic dysfunction and preserved ejection fraction (EF) (HFpEF) is "
        "increasing. Prognosis worsens…"
    )
    assert msg.content == expected_content


# ------------------------------------------------------------------
# get_terminated — sildenafil × stroke
# ------------------------------------------------------------------


async def test_get_terminated_content_string_sildenafil_stroke():
    """get_terminated renders the un-capped relevance-classification view:
    interventions (drugs), classified stop reason, title, and truncated
    brief_summary, with an indented why_stopped excerpt under each row (MeSH
    dropped from this view). Sildenafil × stroke has 2 terminated trials
    (NCT00452582 'Failure to recruit...' → other; NCT02628847
    'Recruitment was problematic' → enrollment).

    Trials are ordered by enrollment desc (20, 11). 0 safety/efficacy stops
    on this pair.
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_terminated = next(t for t in tools if t.name == "get_terminated")

    msg = await get_terminated.ainvoke(
        {
            "name": "get_terminated",
            "args": {"drug": "sildenafil", "indication": "stroke"},
            "id": "test-terminated-1",
            "type": "tool_call",
        }
    )

    artifact = msg.artifact
    assert isinstance(artifact, TerminatedTrialsResult)
    # Only NCT02628847 remains: NCT00452582 was terminated AFTER the holdout
    # cutoff (_CUTOFF = 2025-01-01), so the holdout scrubber drops it (it was
    # not yet terminated as of the cutoff). The content string notes the drop.
    assert artifact.total_count == 1
    assert len(artifact.trials) == 1
    nct_628847 = next(t for t in artifact.trials if t.nct_id == "NCT02628847")
    assert nct_628847.phase == "Phase 1"
    assert nct_628847.enrollment == 11
    assert nct_628847.why_stopped == "Recruitment was problematic"
    assert _classify_stop_reason(nct_628847.why_stopped) == "enrollment"

    expected_content = (
        "Terminated for sildenafil × stroke: 1 total "
        "(0 safety/efficacy in shown set); dropped 1 post-cutoff "
        "termination(s) (not yet terminated at cutoff)\n"
        "Judge relevance from the drugs (is sildenafil the studied drug?), title, and "
        "summary — is the disease THIS indication, not a distinct subtype?\n"
        "Phase distribution (shown): Phase 1=1\n"
        "Trials shown (all 1 — classify EVERY one):\n"
        "  NCT02628847 | Phase 1          | drugs: sildenafil citrate; Placebo | "
        "stop: enrollment | "
        "Sildenafil and Stroke Recovery | "
        "summary: This is a small, pilot randomized clinical trial of administering "
        "sildenafil citrate to individuals within 10 days of ischemic stroke who have "
        "motor impairment…\n"
        "    why_stopped: Recruitment was problematic"
    )
    assert msg.content == expected_content


# ------------------------------------------------------------------
# Supervisor view — analyze_clinical_trials assembly
# ------------------------------------------------------------------


async def test_supervisor_view_completed_table_sildenafil_hfpef():
    """The supervisor's `analyze_clinical_trials` content string includes
    per-trial start_date / completion_date in ISO format and mesh
    conditions. This test reproduces the supervisor's table-assembly
    logic against a real artifact and asserts the exact rendered string.

    We don't run the full supervisor agent (would require a real LLM);
    we exercise the formatters with the same inputs `analyze_clinical_trials`
    feeds them.

    Borda rank for the 4 sildenafil × HFpEF completed trials orders them:
    NCT00763867 (best enrollment, mid recency) →
    NCT01726049 (mid enrollment, best recency) →
    NCT01046838 (mid both) →
    NCT01156636 (worst both).
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_completed = next(t for t in tools if t.name == "get_completed")
    msg = await get_completed.ainvoke(
        {
            "name": "get_completed",
            "args": {"drug": "sildenafil", "indication": "diastolic heart failure"},
            "id": "test-supervisor-completed",
            "type": "tool_call",
        }
    )
    completed_trials = msg.artifact.trials

    phase_dist = _phase_distribution(completed_trials)
    assert phase_dist == "Phase 4=1, Phase 3=2, Phase 2/Phase 3=1"

    top = _borda_rank_by_enrollment_and_recency(completed_trials, k=10)
    assert [t.nct_id for t in top] == [
        "NCT00763867",
        "NCT01726049",
        "NCT01046838",
        "NCT01156636",
    ]

    table = _format_trial_table(
        top,
        columns=(
            "nct_id",
            "phase",
            "start_date",
            "completion_date",
            "mesh",
            "title",
        ),
        cap=10,
    )
    expected_table = (
        "  NCT00763867 | Phase 3          | start 2008-09 | end 2012-09 | "
        "mesh: Heart Failure; Heart Failure, Diastolic | "
        "Evaluating the Effectiveness of Sildenafil at Improving Health Outcomes "
        "and Exercise Ability in People With Diastolic Heart Failure (The RELAX Study)\n"
        "  NCT01726049 | Phase 3          | start 2011-10 | end 2014-09 | "
        "mesh: Heart Failure, Diastolic; Hypertension, Pulmonary | "
        "Sildenafil in HFpEF (Heart Failure With Preserved Ejection Fraction) and PH\n"
        "  NCT01046838 | Phase 4          | start 2009-12 | end 2012-03 | "
        "mesh: Heart Failure, Diastolic | "
        "SIDAMI - Sildenafil and Diastolic Dysfunction After Acute Myocardial "
        "Infarction (AMI)\n"
        "  NCT01156636 | Phase 2/Phase 3  | start 2006-01 | end 2009-09 | "
        "mesh: Hypertension, Pulmonary; Heart Failure, Diastolic | "
        "Phosphodiesterase-5 (PDE5) Inhibition and Pulmonary Hypertension in "
        "Diastolic Heart Failure"
    )
    assert table == expected_table


async def test_supervisor_view_terminated_table_sildenafil_stroke():
    """Supervisor's terminated table for sildenafil × stroke renders Borda
    rank, classified stop reason, ISO start/end dates (with `end ?` for
    NCT00452582 which has no completion_date), mesh, title, and indented
    why_stopped.

    Only NCT02628847 remains: NCT00452582 was terminated after the holdout
    cutoff (_CUTOFF = 2025-01-01) and is dropped by the holdout scrubber.
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    get_terminated = next(t for t in tools if t.name == "get_terminated")
    msg = await get_terminated.ainvoke(
        {
            "name": "get_terminated",
            "args": {"drug": "sildenafil", "indication": "stroke"},
            "id": "test-supervisor-terminated",
            "type": "tool_call",
        }
    )
    terminated_trials = msg.artifact.trials

    phase_dist = _phase_distribution(terminated_trials)
    assert phase_dist == "Phase 1=1"

    top = _borda_rank_by_enrollment_and_recency(terminated_trials, k=10)
    assert [t.nct_id for t in top] == ["NCT02628847"]

    table = _format_trial_table(
        top,
        columns=(
            "nct_id",
            "phase",
            "stop_reason",
            "start_date",
            "completion_date",
            "mesh",
            "title",
        ),
        cap=10,
        include_why_stopped=True,
        stop_classifier=_classify_stop_reason,
    )
    expected_table = (
        "  NCT02628847 | Phase 1          | stop: enrollment | "
        "start 2012-03 | end 2016-10 | mesh: Stroke | "
        "Sildenafil and Stroke Recovery\n"
        "    why_stopped: Recruitment was problematic"
    )
    assert table == expected_table
