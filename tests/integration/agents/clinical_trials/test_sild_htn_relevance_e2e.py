"""Gated end-to-end: the clinical_trials agent splits sildenafil × hypertension
relevant vs contaminated correctly, and classifies EVERY shown trial (2a).

The registry over-recalls for this pair: pulmonary hypertension / PAH (a distinct
disease sildenafil is already approved for) and other-drug trials (sitaxsentan,
PF-00489791, sodium nitrite) come back under a systemic-hypertension query. This
test runs the real agent and asserts:
  - completeness (2a): every completed+terminated trial the agent saw got a
    verdict — relevant ∪ contaminated covers the shown set, no overlap.
  - accuracy (2b): for the labeled trials the agent actually saw, relevant
    contains the labeled-R trials and contaminated contains the labeled-C ones.

Oracle: the hand labels in test_relevance_signal_eval.LABELS. Because the live
CT.gov set drifts, assertions intersect LABELS with what the agent saw rather
than pinning an exact set.

Hits real CT.gov + real Anthropic.

Run: pytest tests/integration/agents/clinical_trials/test_sild_htn_relevance_e2e.py
"""

import logging
from datetime import date

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)

from .test_relevance_signal_eval import LABELS

logger = logging.getLogger(__name__)

# Holdout cutoff that matches the harness snapshot (sild_htn_trials.json was
# captured with date_before=2025-01-01), so the shown set overlaps the labels.
_CUTOFF = date(2025, 1, 1)

_LABELED_RELEVANT = {n for n, v in LABELS.items() if v == "R"}
_LABELED_CONTAMINATED = {n for n, v in LABELS.items() if v == "C"}

# Genuinely-ambiguous trials the plan accepts as borderline (PLAN §6 non-goals):
# NCT00150358 is an erectile-dysfunction trial whose MeSH mislabels it
# "Hypertension"; the rich signal can read it either way. The harness documented
# it as the one acceptable false-relevant — we tolerate it rather than chase 100%
# on a borderline case. The test still fails on any OTHER contaminated→relevant
# leak (PAH, sitaxsentan, etc.).
_KNOWN_BORDERLINE = {"NCT00150358"}


async def test_sildenafil_hypertension_relevance_split_e2e():
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(llm, date_before=_CUTOFF)

    output = await run_clinical_trials_agent(agent, "sildenafil", "hypertension")

    relevant = set(output.relevant_nct_ids)
    contaminated = set(output.contaminated_nct_ids)

    # The set the agent was asked to classify: completed + terminated trials shown.
    shown = {t.nct_id for t in (output.completed.trials if output.completed else [])}
    shown |= {t.nct_id for t in (output.terminated.trials if output.terminated else [])}
    assert shown, "agent saw no completed/terminated trials — fixture/holdout drift"

    # --- completeness (2a): every shown trial classified, exactly once ---
    assert relevant | contaminated == shown
    assert not (relevant & contaminated)

    # --- accuracy (2b), scoped to labeled trials the agent actually saw ---
    seen_relevant = _LABELED_RELEVANT & shown
    seen_contaminated = _LABELED_CONTAMINATED & shown
    assert seen_relevant or seen_contaminated, "no labeled trials in shown set"

    misclassified_relevant = seen_relevant - relevant
    misclassified_contaminated = seen_contaminated - contaminated
    logger.info(
        "sild × htn e2e — shown=%d relevant=%d contaminated=%d; labeled seen "
        "R=%d C=%d; misclassified R→C=%s C→R=%s",
        len(shown),
        len(relevant),
        len(contaminated),
        len(seen_relevant),
        len(seen_contaminated),
        sorted(misclassified_relevant),
        sorted(misclassified_contaminated),
    )

    # Labeled-relevant systemic-HTN trials must not be dropped as contaminated.
    assert misclassified_relevant == set()
    # Labeled-contaminated PAH/other-drug trials must not leak in as relevant —
    # except the documented borderline ED-in-hypertensives trial (PLAN §6).
    assert misclassified_contaminated <= _KNOWN_BORDERLINE
