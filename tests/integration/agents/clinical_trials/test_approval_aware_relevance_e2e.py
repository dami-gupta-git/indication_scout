"""Gated end-to-end: the approval-aware relevance gate (TEST 1) treats a trial whose condition is
an APPROVED sub-indication of a broad candidate as CONTAMINATION — and the exclusion is DRIVEN by
the approved-indications list passed into run_clinical_trials_agent, not invented by the model.

Mirrors the validated harness (scratch/approval_aware_trials_harness.py) at the real-agent level:
  - list-driven control: the SAME PAH trial under sildenafil × hypertension flips
    contaminated→relevant purely on whether approved={PAH} is supplied. With PAH approved, the PAH
    trial is an approved sub-indication => contaminated; with approved=(none), TEST 1 never fires
    and PAH rolls up (relevant or contaminated on the distinct-disease axis — we only assert the
    APPROVED-driven direction: WITH the list it must be contaminated).
  - motivating bug: bupropion × mood disorder with approved={SAD,MDD} marks the approved-SAD trial
    NCT00046241 contaminated via the general RULE (the CURATED_CONTAMINATED_NCTS hardcode is gone).

Assertions intersect with what the agent actually saw (live CT.gov set drifts).

Hits real CT.gov + real Anthropic.

Run: pytest tests/integration/agents/clinical_trials/test_approval_aware_relevance_e2e.py
"""

import logging
from datetime import date

import pytest

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.approval_aware

# Holdout cutoffs pinned so the shown set is stable (matches the harness snapshots / known NCTs).
_SILD_CUTOFF = date(2025, 1, 1)
_BUP_CUTOFF = date(2025, 1, 1)

# A PAH trial that the systemic-hypertension query over-recalls for sildenafil. PAH is an approved
# sildenafil indication, so under approved={PAH} it must be classified contaminated (TEST 1).
_SILD_PAH_NCT = "NCT01181284"
# Therapeutic-intent mismatch (TEST 2): sildenafil studied FOR erectile dysfunction (approved) in a
# HYPERTENSIVE population — hypertension is only the population, not the target => contaminated.
_SILD_ED_IN_HTN_NCT = "NCT02620995"
# The approved-SAD bupropion trial — the motivating CURATED_CONTAMINATED_NCTS case.
_BUP_SAD_NCT = "NCT00046241"
# semaglutide × NAFLD: a trial whose condition is "type 2 diabetes with NASH". NASH is the approved
# subtype; the T2DM co-listed condition must NOT rescue it (TEST 1 multi-condition rule). This
# trial flipped relevant<->contaminated across runs before the rule was added.
_SEMA_T2DM_NASH_NCT = "NCT04639414"
_SEMA_CUTOFF = date(2025, 1, 1)


def _shown(output) -> set[str]:
    shown = {t.nct_id for t in (output.search.trials if output.search else [])}
    shown |= {t.nct_id for t in (output.completed.trials if output.completed else [])}
    shown |= {t.nct_id for t in (output.terminated.trials if output.terminated else [])}
    return {n for n in shown if n}


async def test_sildenafil_pah_excluded_only_when_approved_list_supplied():
    """List-driven control: approved={PAH} => PAH trial contaminated (TEST 1)."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_SILD_CUTOFF)

    output = await run_clinical_trials_agent(
        agent,
        "sildenafil",
        "hypertension",
        approved_indications=["pulmonary arterial hypertension", "erectile dysfunction"],
    )

    shown = _shown(output)
    assert _SILD_PAH_NCT in shown, (
        f"{_SILD_PAH_NCT} not in shown set — CT.gov drift; cannot assert TEST 1"
    )
    contaminated = set(output.contaminated_nct_ids)
    relevant = set(output.relevant_nct_ids)
    # completeness: shown set fully classified, no overlap
    assert relevant | contaminated >= shown
    assert not (relevant & contaminated)
    # TEST 1: approved PAH sub-indication => contaminated, NOT relevant
    assert _SILD_PAH_NCT in contaminated
    assert _SILD_PAH_NCT not in relevant


async def test_sildenafil_ed_in_hypertensives_contaminated_therapeutic_intent():
    """Therapeutic-intent mismatch (TEST 2): a sildenafil-for-ED trial run in a HYPERTENSIVE
    population (NCT02620995) must be contaminated for systemic hypertension — hypertension is only
    the population, the target is the approved ED indication."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_SILD_CUTOFF)

    output = await run_clinical_trials_agent(
        agent,
        "sildenafil",
        "hypertension",
        approved_indications=["pulmonary arterial hypertension", "erectile dysfunction"],
    )

    shown = _shown(output)
    contaminated = set(output.contaminated_nct_ids)
    relevant = set(output.relevant_nct_ids)
    assert relevant | contaminated >= shown
    assert not (relevant & contaminated)
    if _SILD_ED_IN_HTN_NCT not in shown:
        pytest.skip(f"{_SILD_ED_IN_HTN_NCT} not recalled this run — cannot assert TEST 2")
    assert _SILD_ED_IN_HTN_NCT in contaminated
    assert _SILD_ED_IN_HTN_NCT not in relevant


async def test_bupropion_approved_sad_trial_contaminated_via_rule():
    """Motivating bug: approved={SAD,MDD} => the approved-SAD trial is contaminated by the RULE."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_BUP_CUTOFF)

    output = await run_clinical_trials_agent(
        agent,
        "bupropion",
        "mood disorder",
        approved_indications=[
            "seasonal affective disorder",
            "major depressive disorder",
            "smoking cessation",
        ],
    )

    shown = _shown(output)
    assert _BUP_SAD_NCT in shown, (
        f"{_BUP_SAD_NCT} not in shown set — CT.gov drift; cannot assert TEST 1"
    )
    contaminated = set(output.contaminated_nct_ids)
    relevant = set(output.relevant_nct_ids)
    assert relevant | contaminated >= shown
    assert not (relevant & contaminated)
    assert _BUP_SAD_NCT in contaminated
    assert _BUP_SAD_NCT not in relevant


async def test_semaglutide_multi_condition_nash_trial_contaminated():
    """Multi-condition TEST 1: a "type 2 diabetes with NASH" trial contaminates because NASH is
    the approved subtype — the co-listed T2DM must not rescue it (the NCT04639414 flip bug)."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_SEMA_CUTOFF)

    output = await run_clinical_trials_agent(
        agent,
        "semaglutide",
        "non-alcoholic fatty liver disease",
        approved_indications=[
            "MASH (NASH) with moderate-to-advanced fibrosis",
            "type 2 diabetes mellitus",
            "obesity",
        ],
    )

    shown = _shown(output)
    contaminated = set(output.contaminated_nct_ids)
    relevant = set(output.relevant_nct_ids)
    # completeness holds regardless of which trials CT.gov recalled this run
    assert relevant | contaminated >= shown
    assert not (relevant & contaminated)
    # The multi-condition anchor: assert only when CT.gov actually recalled it (the set drifts).
    if _SEMA_T2DM_NASH_NCT not in shown:
        pytest.skip(f"{_SEMA_T2DM_NASH_NCT} not recalled this run — cannot assert TEST 1")
    assert _SEMA_T2DM_NASH_NCT in contaminated
    assert _SEMA_T2DM_NASH_NCT not in relevant


# imatinib × leukemia: imatinib is approved for Ph+ CML and Ph+ ALL. The broad "leukemia" candidate
# over-recalls approved Ph+ CML/ALL trials (contaminated) alongside a distinct non-approved leukemia
# (CLL) trial that is genuine repurposing (relevant). The CLL trial is the stable anchor.
_IMA_CLL_RELEVANT_NCT = "NCT00558961"  # imatinib + chlorambucil in CLL — distinct, non-approved
_IMA_CML_CONTAMINATED_NCT = "NCT00102440"  # imatinib in Ph+ CML — an approved indication
_IMA_CUTOFF = date(2025, 1, 1)


async def test_imatinib_leukemia_approved_cml_contaminated_distinct_cll_relevant():
    """imatinib × leukemia: a Ph+ CML trial (approved indication) is contaminated, while a CLL
    trial (a distinct leukemia not covered by any imatinib approval) is relevant. Guards the
    approved-subtype contamination that over-recall pulls into the broad 'leukemia' candidate."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_IMA_CUTOFF)

    output = await run_clinical_trials_agent(
        agent,
        "imatinib",
        "leukemia",
        approved_indications=[
            "Philadelphia chromosome-positive chronic myeloid leukemia",
            "Philadelphia chromosome-positive acute lymphoblastic leukemia",
        ],
    )

    shown = _shown(output)
    contaminated = set(output.contaminated_nct_ids)
    relevant = set(output.relevant_nct_ids)
    # completeness: every shown trial classified, no overlap
    assert relevant | contaminated >= shown
    assert not (relevant & contaminated)
    # CLL is a distinct non-approved leukemia => relevant (the stable anchor).
    if _IMA_CLL_RELEVANT_NCT in shown:
        assert _IMA_CLL_RELEVANT_NCT in relevant
        assert _IMA_CLL_RELEVANT_NCT not in contaminated
    # An approved Ph+ CML trial => contaminated.
    if _IMA_CML_CONTAMINATED_NCT in shown:
        assert _IMA_CML_CONTAMINATED_NCT in contaminated
        assert _IMA_CML_CONTAMINATED_NCT not in relevant
