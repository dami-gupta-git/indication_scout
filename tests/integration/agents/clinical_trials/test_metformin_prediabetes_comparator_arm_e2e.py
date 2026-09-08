"""Gated end-to-end: the clinical_trials agent must reject metformin × prediabetes trials
where metformin is only the comparator arm. Both were classified relevant in the 2026-09-08
runs. Metformin IS a listed intervention in both, so only the wrong-drug test catches them.

Hits real CT.gov + real Anthropic.
"""

import logging

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)

logger = logging.getLogger(__name__)

# Metformin is the comparator; the studied agent is dapagliflozin / green tea.
_COMPARATOR_ARM_TRIALS = {"NCT03968224", "NCT06229795"}

# DPP Outcomes + both RISE studies — real metformin trials. Guards against over-rejecting.
_STUDIED_AGENT_TRIALS = {"NCT00038727", "NCT01779362", "NCT01779375"}


async def test_metformin_prediabetes_comparator_arm_trials_are_contaminated():
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_clinical_trials_agent(
        llm, assigned_indication="prediabetes syndrome", target_drug="metformin"
    )

    output = await run_clinical_trials_agent(agent, "metformin", "prediabetes syndrome")

    relevant = set(output.relevant_nct_ids)
    contaminated = set(output.contaminated_nct_ids)

    shown = {t.nct_id for t in (output.completed.trials if output.completed else [])}
    shown |= {t.nct_id for t in (output.terminated.trials if output.terminated else [])}
    shown |= {t.nct_id for t in (output.search.trials if output.search else [])}
    assert shown, "agent saw no trials — CT.gov drift"

    # Every shown trial classified, exactly once.
    assert relevant | contaminated == shown
    assert not (relevant & contaminated)

    seen_comparator = _COMPARATOR_ARM_TRIALS & shown
    seen_studied = _STUDIED_AGENT_TRIALS & shown
    assert (
        seen_comparator
    ), "labeled comparator-arm trials absent from shown set — CT.gov drift"
    assert seen_studied, "labeled metformin trials absent from shown set — CT.gov drift"

    leaked = seen_comparator - contaminated
    dropped = seen_studied - relevant
    logger.info(
        "metformin × prediabetes e2e — shown=%d relevant=%d contaminated=%d; "
        "comparator-arm leaked=%s studied-agent dropped=%s",
        len(shown),
        len(relevant),
        len(contaminated),
        sorted(leaked),
        sorted(dropped),
    )

    assert leaked == set()
    assert dropped == set()
