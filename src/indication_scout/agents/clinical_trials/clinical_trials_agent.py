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
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.services.clinical_trials_summary import judge_ct_summary
from indication_scout.services.dev_stage import dev_stage_phrase, judge_dev_stage

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()


def _finalize_done(messages: list) -> bool:
    """End the loop once finalize_analysis has SUCCEEDED this turn.

    A rejected finalize (missing/unknown verdicts) returns an empty-string artifact and
    must loop back to the model to retry, so end only on a truthy artifact.
    """
    for m in _trailing_tool_messages(messages):
        if m.name == "finalize_analysis" and m.artifact:
            return True
    return False


def build_clinical_trials_agent(llm, date_before=None, assigned_indication=None):
    """Return a compiled ReAct agent. No graph wiring required.

    `assigned_indication` pins the tools to one indication: a call for any other
    indication is soft-rejected so a drifting agent self-corrects instead of crashing
    at finalize (see build_clinical_trials_tools).
    """
    tools = build_clinical_trials_tools(
        date_before=date_before,
        assigned_indication=assigned_indication,
    )
    return build_gated_react_loop(llm, tools, SYSTEM_PROMPT, _finalize_done)


async def run_clinical_trials_agent(
    agent,
    drug_name: str,
    disease_name: str,
    first_approval: int | None = None,
    approved_indications: list[str] | None = None,
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run.

    `first_approval` is the year the drug was first approved anywhere (ChEMBL). It is fed
    to the agent so its closure judgment can tell "old/generic drug, no new NDA expected"
    (no-approval is not failure) from a genuine negative. When None, the literal "unknown"
    is passed — never a default year (CLAUDE.md no-fallback).

    `approved_indications` is the drug's FDA-approved indications (from the supervisor store).
    Rendered into the task so the relevance gate's TEST 1 can mark a trial whose condition is an
    approved sub-indication of a broad candidate as CONTAMINATION. Empty/None renders "(none)",
    which disables TEST 1 (no behavior change for non-contaminated candidates).
    """
    approval_line = (
        f"first_approval (year first approved anywhere): {first_approval}"
        if first_approval is not None
        else "first_approval (year first approved anywhere): unknown"
    )
    approved_line = (
        "FDA-approved indications of this drug: "
        + (", ".join(approved_indications) if approved_indications else "(none)")
    )
    task = (
        f"Analyze {drug_name} in {disease_name}\n"
        f"DRUG FACT — {approval_line}\n"
        f"DRUG FACT — {approved_line}"
    )
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
        # Include args so repeated search_trials/get_* calls across turns show WHAT each
        # retry is querying (e.g. a reworded disease term), not just that a tool re-ran.
        # Args are rendered compactly and truncated — finalize_analysis carries a per-NCT
        # verdict list (hundreds of entries) that would otherwise flood the log on every turn.
        def _fmt_args(args: dict) -> str:
            rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return rendered if len(rendered) <= 200 else rendered[:200] + "…"

        called = (
            ", ".join(f"{tc['name']}({_fmt_args(tc['args'])})" for tc in msg.tool_calls)
            or "(final)"
        )
        logger.info(
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
    logger.info(
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
            "relevance, signals, and summary will be empty",
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
            output,
            relevant_nct_ids=set(output.relevant_nct_ids),
            contaminated_nct_ids=set(output.contaminated_nct_ids),
        )
        # dev_stage is an LLM judgment (not the deterministic phase-rank, which mis-encoded
        # the Phase-4 trap). Only nct/phase/status of the relevant trials is sent. Cached per
        # trial set. Keeps the deterministic dev_stage already on signals as the fallback when
        # the relevant set is empty.
        #
        # Trial set for the judgment: completed + terminated + search, all filtered by
        # relevant_nct_ids (the agent's relevance split). search-pool trials — where
        # active/recruiting Phase 3s live — are now classified by the relevance gate too (they
        # are added to shown_by_indication in search_trials), so the SAME positive
        # `in relevant_set` filter applies to every scope. This closes the gap where a
        # contaminated active trial (e.g. a PAH trial under a systemic-hypertension query) drove
        # the "Phase 3 active" dev-stage signal because search trials bypassed the gate.
        relevant_set = set(output.relevant_nct_ids)
        seen: set[str] = set()
        relevant_trials = []
        for t in (output.completed.trials if output.completed else []) + (
            output.terminated.trials if output.terminated else []
        ) + (output.search.trials if output.search else []):
            if t.nct_id and t.nct_id in relevant_set and t.nct_id not in seen:
                seen.add(t.nct_id)
                relevant_trials.append(t)
        if relevant_trials:
            judgment = await judge_dev_stage(
                relevant_trials,
                DEFAULT_CACHE_DIR,
                drug=drug_name,
                indication=disease_name,
            )
            output.signals.dev_stage = judgment.tier
            output.signals.active_programs = judgment.active_programs

            # LOAD-BEARING ORDER: judge_dev_stage MUST run before judge_ct_summary. The whole
            # fix is that the trial-section prose is FED the resolved stage so it cannot
            # contradict it (the T1DM "no completed Phase 3" bug). Do NOT reorder these.
            stage_phrase = dev_stage_phrase(output.signals)
            if stage_phrase:
                ct_summary = await judge_ct_summary(
                    relevant_trials,
                    stage=stage_phrase,
                    active_programs=judgment.active_programs,
                    first_approval=first_approval,
                    cache_dir=DEFAULT_CACHE_DIR,
                    drug=drug_name,
                    indication=disease_name,
                )
                if ct_summary is not None:
                    output.summary = ct_summary.prose
                    output.closure = ct_summary.closure
                    output.closure_reason = ct_summary.closure_reason
    return output
