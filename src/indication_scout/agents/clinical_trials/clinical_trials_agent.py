"""Clinical Trials agent

Uses a gated ReAct loop (see agents/_react_loop.py) that ends as soon as finalize_analysis
succeeds, skipping the discarded trailing model turn. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

import logging
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents._react_loop import (
    _trailing_tool_messages,
    build_gated_react_loop,
)
from indication_scout.agents._trial_signals import derive_trial_signals
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
    FinalizeClinicalTrialsArtifact,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()


def _finalize_done(messages: list) -> bool:
    """End the loop once finalize_analysis has SUCCEEDED this turn.

    A rejected finalize (empty/whitespace summary) returns an empty-string artifact and
    must loop back to the model to retry, so end only on a truthy artifact.
    """
    for m in _trailing_tool_messages(messages):
        if m.name == "finalize_analysis" and m.artifact:
            return True
    return False


def build_clinical_trials_agent(llm, date_before=None):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_clinical_trials_tools(
        date_before=date_before,
    )
    return build_gated_react_loop(llm, tools, SYSTEM_PROMPT, _finalize_done)


async def run_clinical_trials_agent(
    agent,
    drug_name: str,
    disease_name: str,
    first_approval: int | None = None,
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run.

    `first_approval` is the year the drug was first approved anywhere (ChEMBL). It is fed
    to the agent so its closure judgment can tell "old/generic drug, no new NDA expected"
    (no-approval is not failure) from a genuine negative. When None, the literal "unknown"
    is passed — never a default year (CLAUDE.md no-fallback).
    """
    approval_line = (
        f"first_approval (year first approved anywhere): {first_approval}"
        if first_approval is not None
        else "first_approval (year first approved anywhere): unknown"
    )
    task = f"Analyze {drug_name} in {disease_name}\n" f"DRUG FACT — {approval_line}"
    _agent_t0 = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    _agent_elapsed = time.perf_counter() - _agent_t0

    # Per-turn LLM accounting (same as literature/mechanism agents). Each AIMessage
    # is one round-trip; usage_metadata gives context size and output tokens.
    # Logged at WARNING to isolate clinical_trials loop overhead from its (partly
    # uncached) API calls. Read-only on result["messages"].
    ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    total_out = 0
    for i, msg in enumerate(ai_turns):
        usage = msg.usage_metadata or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        # cache_read/cache_write surface whether the prompt-caching breakpoints are
        # hitting; cache_read==0 across turns 2+ means a silent invalidator (e.g. prefix
        # below the model's min cacheable size) is at work. langchain-anthropic reports
        # freshly-written tokens under the TTL-specific ephemeral keys, not cache_creation.
        details = usage.get("input_token_details", {})
        cache_read = details.get("cache_read", 0)
        cache_write = (
            details.get("ephemeral_5m_input_tokens", 0)
            + details.get("ephemeral_1h_input_tokens", 0)
        ) or details.get("cache_creation", 0)
        total_out += out_tok
        called = ", ".join(tc["name"] for tc in msg.tool_calls) or "(final)"
        logger.warning(
            "[LLMTURN] clinical_trials %s turn %d/%d: in=%d out=%d cache_read=%d "
            "cache_write=%d -> %s",
            disease_name,
            i + 1,
            len(ai_turns),
            in_tok,
            out_tok,
            cache_read,
            cache_write,
            called,
        )
    logger.warning(
        "[LLMTURN] clinical_trials %s: %d turns, %d total output tokens, "
        "agent loop %.1fs",
        disease_name,
        len(ai_turns),
        total_out,
        _agent_elapsed,
    )

    artifacts: dict = {
        "search": None,
        "completed": None,
        "terminated": None,
        "landscape": None,
        "approval": None,
        "finalize": None,
    }

    field_map = {
        "search_trials": "search",
        "get_completed": "completed",
        "get_terminated": "terminated",
        "get_landscape": "landscape",
        "check_fda_approval": "approval",
        "finalize_analysis": "finalize",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    tools_called = [k for k, v in artifacts.items() if v is not None]
    logger.warning(
        "clinical_trials_agent: %s × %s — tools called: %s",
        drug_name,
        disease_name,
        tools_called,
    )

    if artifacts["approval"] is None:
        logger.warning(
            "clinical_trials_agent: %s × %s — check_fda_approval was not called "
            "(prompt requires it as step 1)",
            drug_name,
            disease_name,
        )

    # finalize artifact is a FinalizeClinicalTrialsArtifact on a normal end, or "" / None if
    # finalize was never reached (or only rejected). Unpack defensively.
    finalize = artifacts.get("finalize")
    finalized = isinstance(finalize, FinalizeClinicalTrialsArtifact)
    if not finalized:
        logger.warning(
            "clinical_trials_agent: %s × %s — finalize_analysis produced no artifact; "
            "summary, relevance, and signals will be empty",
            drug_name,
            disease_name,
        )
        finalize = FinalizeClinicalTrialsArtifact()

    output = ClinicalTrialsOutput(
        search=artifacts["search"],
        completed=artifacts["completed"],
        terminated=artifacts["terminated"],
        landscape=artifacts["landscape"],
        approval=artifacts["approval"],
        summary=finalize.summary,
        relevant_nct_ids=finalize.relevant_ncts,
        contaminated_nct_ids=finalize.contaminated_ncts,
        relevance_reasoning=finalize.relevance_reasoning,
    )

    # Signals are computed from RELEVANT trials only, so supervisor and human report read
    # identical numbers. Only when the agent actually classified relevance (finalize ran) —
    # otherwise leave signals None so the supervisor knows no relevance judgment was made,
    # rather than silently filtering every trial out with an empty relevant set.
    if finalized:
        output.signals = derive_trial_signals(
            output, relevant_nct_ids=set(output.relevant_nct_ids)
        )
    return output
