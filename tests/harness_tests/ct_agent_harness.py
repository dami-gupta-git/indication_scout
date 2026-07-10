"""Standalone harness: send a drug × disease into the REAL clinical_trials agent and
dump (a) the raw message-history response and (b) the assembled ClinicalTrialsOutput —
i.e. the DECISION the agent made (summary verdict + relevance split + derived signals).

Purpose: this is the data we need to carry faithfully up to the supervisor. The harness
runs the agent exactly as the supervisor does (build_clinical_trials_agent +
run_clinical_trials_agent, same ChatAnthropic config as analysis_runner.build_agent),
with no supervisor pipeline in between.

Two REAL agent runs per case: one raw invoke (to print the message history) and one via
run_clinical_trials_agent (to print the typed decision through the production assembly
path). temperature=0 but tool I/O is live, so the two runs can differ slightly.

Run: .venv/bin/python tests/harness_tests/ct_agent_harness.py "metformin" "polycystic ovary syndrome"
     .venv/bin/python tests/harness_tests/ct_agent_harness.py        # runs the built-in A1/A2 cases
"""

import asyncio
import logging
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents._trial_signals import format_derived_signals
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.config import get_settings

logger = logging.getLogger("ct_agent_harness")

# Set FULL=1 in the env to print untruncated message history.
import os  # noqa: E402

_FULL = os.environ.get("FULL") == "1"

# Bug A cases from BUGS_2026-06-14.md, plus first_approval years (generic → no-NDA cue).
CASES: list[tuple[str, str, int | None]] = [
    ("metformin", "polycystic ovary syndrome", 1995),   # A2: 22 completed Phase 3, must be LIVE
    ("metformin", "hepatic steatosis", 1995),           # A1: terminations operational, NOT closed
    ("sildenafil", "covid-19", 1998),                   # B: MeSH mis-resolution (separate bug)
]


def _build_llm() -> ChatAnthropic:
    """Same construction as analysis_runner.build_agent — keep the agent identical."""
    settings = get_settings()
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )


def _dump_messages(messages: list, full: bool = False) -> None:
    """Print the raw agent response (the message history walk run_..._agent harvests).

    full=True prints every message untruncated (complete tool outputs and the full
    finalize input), at the cost of a very long dump.
    """
    print("\n--- RAW MESSAGE HISTORY ---")
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            print(f"[{i}] HUMAN: {m.content}")
        elif isinstance(m, AIMessage):
            calls = ", ".join(tc["name"] for tc in m.tool_calls) or "(final answer)"
            text = m.content if isinstance(m.content, str) else str(m.content)
            print(f"[{i}] AI -> {calls}")
            if text.strip():
                print(f"      text: {text.strip() if full else text.strip()[:600]}")
        elif isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"[{i}] TOOL {m.name}: {content.strip() if full else content.strip()[:300]}")


def _dump_decision(output) -> None:
    """Print the assembled decision — what the supervisor would consume."""
    print("\n--- ASSEMBLED DECISION (ClinicalTrialsOutput) ---")
    print(f"relevant_nct_ids   ({len(output.relevant_nct_ids)}): {output.relevant_nct_ids}")
    print(
        f"contaminated_nct_ids ({len(output.contaminated_nct_ids)}): "
        f"{output.contaminated_nct_ids}"
    )
    print(f"relevance_reasoning: {output.relevance_reasoning}")
    print("\nsignals:")
    if output.signals:
        print(format_derived_signals(output.signals))
    else:
        print("  (None — finalize_analysis did not run; no relevance judgment made)")
    print("\nsummary (the prose closure verdict carried to the supervisor):")
    print(output.summary or "  (empty)")


async def _run_one(drug: str, disease: str, first_approval: int | None) -> None:
    print("=" * 90)
    print(f"CASE: {drug} × {disease}  (first_approval={first_approval})")
    print("=" * 90)
    llm = _build_llm()

    # Run 1: raw invoke to print the message history (the agent's actual response).
    agent = build_clinical_trials_agent(llm=llm)
    task = (
        f"Analyze {drug} in {disease}\n"
        f"DRUG FACT — first_approval (year first approved anywhere): "
        f"{first_approval if first_approval is not None else 'unknown'}"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    _dump_messages(result["messages"], full=_FULL)

    # Run 2: production path — fresh agent, run_clinical_trials_agent assembles the typed
    # decision exactly as the supervisor receives it. Second real run (live tool I/O).
    agent2 = build_clinical_trials_agent(llm=llm)
    output = await run_clinical_trials_agent(
        agent2, drug, disease, first_approval=first_approval
    )
    _dump_decision(output)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    if len(sys.argv) >= 3:
        drug, disease = sys.argv[1], sys.argv[2]
        first_approval = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        await _run_one(drug, disease, first_approval)
    else:
        for drug, disease, first_approval in CASES:
            await _run_one(drug, disease, first_approval)


if __name__ == "__main__":
    asyncio.run(main())
