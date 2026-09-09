"""Clinical Trials agent

Gated ReAct loop (agents/_react_loop.py) ending as soon as finalize_analysis succeeds. After
the run, pulls typed artifacts off the ToolMessages into a ClinicalTrialsOutput.
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
    TrialRelevanceCoverage,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
)
from indication_scout.services.clinical_trials_summary import judge_ct_summary
from indication_scout.services.dev_stage import dev_stage_phrase, judge_dev_stage

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()


def _finalize_done(messages: list) -> bool:
    """End the loop once finalize_analysis has SUCCEEDED this turn.

    A rejected finalize returns an empty artifact and must loop back to retry, so end only
    on a truthy one.
    """
    for m in _trailing_tool_messages(messages):
        if m.name == "finalize_analysis" and m.artifact:
            return True
    return False


def build_clinical_trials_agent(
    llm, date_before=None, assigned_indication=None, target_drug=None
):
    """Return a compiled ReAct agent.

    `assigned_indication` pins the tools to one indication; a call for any other is
    soft-rejected so a drifting agent self-corrects instead of crashing at finalize.

    `target_drug` pins the drug the finalize drug-role check is asked about, so the model
    cannot widen it through its own tool arguments.
    """
    tools = build_clinical_trials_tools(
        date_before=date_before,
        assigned_indication=assigned_indication,
        target_drug=target_drug,
    )
    return build_gated_react_loop(llm, tools, SYSTEM_PROMPT, _finalize_done)


def _derive_relevance_coverage(
    result: SearchTrialsResult | CompletedTrialsResult | TerminatedTrialsResult | None,
    *,
    relevant_nct_ids: set[str],
    contaminated_nct_ids: set[str],
) -> TrialRelevanceCoverage | None:
    """Derive query coverage and reviewed relevance counts for one result scope."""
    if result is None:
        return None
    if (
        isinstance(result, SearchTrialsResult)
        and result.resolution_status == "unresolved"
    ):
        return None
    trials_by_id = {trial.nct_id: trial for trial in result.trials if trial.nct_id}
    retrieved_ids = set(trials_by_id)
    relevant_ids = retrieved_ids & relevant_nct_ids
    contaminated_ids = retrieved_ids & contaminated_nct_ids
    classified_ids = relevant_ids | contaminated_ids
    relevant_by_status: dict[str, int] = {}
    for nct_id in relevant_ids:
        status = (trials_by_id[nct_id].overall_status or "").strip().upper()
        if status:
            relevant_by_status[status] = relevant_by_status.get(status, 0) + 1
    return TrialRelevanceCoverage(
        registry_query_matches=result.total_count,
        retrieved_records=len(retrieved_ids),
        classified_records=len(classified_ids),
        relevant_records=len(relevant_ids),
        contaminated_records=len(contaminated_ids),
        unreviewed_records=max(result.total_count - len(classified_ids), 0),
        coverage_complete=(
            result.total_count == len(retrieved_ids)
            and len(classified_ids) == len(retrieved_ids)
        ),
        relevant_by_status=relevant_by_status,
    )


async def run_clinical_trials_agent(
    agent,
    drug_name: str,
    disease_name: str,
    first_approval: int | None = None,
    approved_indications: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run.

    `first_approval` (ChEMBL) lets the closure judgment tell "old generic, no new NDA expected"
    from a genuine negative. None renders "unknown" — never a default year.

    `approved_indications` feeds the relevance gate's approved-subtype test. Empty renders
    "(none)", which disables that test.
    """
    approval_line = (
        f"first_approval (year first approved anywhere): {first_approval}"
        if first_approval is not None
        else "first_approval (year first approved anywhere): unknown"
    )
    approved_line = "FDA-approved indications of this drug: " + (
        ", ".join(approved_indications) if approved_indications else "(none)"
    )
    task = (
        f"Analyze {drug_name} in {disease_name}\n"
        f"DRUG FACT — {approval_line}\n"
        f"DRUG FACT — {approved_line}"
    )
    _agent_t0 = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    _agent_elapsed = time.perf_counter() - _agent_t0

    # Per-turn LLM accounting. Each AIMessage is one round-trip. Read-only.
    ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    total_out = 0
    for i, msg in enumerate(ai_turns):
        usage = msg.usage_metadata or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        # cache_read==0 across turns 2+ means a silent cache invalidator. langchain-anthropic
        # reports fresh writes under the ephemeral keys, not cache_creation.
        details = usage.get("input_token_details", {})
        cache_read = details.get("cache_read", 0)
        cache_write = (
            details.get("ephemeral_5m_input_tokens", 0)
            + details.get("ephemeral_1h_input_tokens", 0)
        ) or details.get("cache_creation", 0)
        total_out += out_tok

        # Args show what each retry queries, not just that a tool re-ran. Truncated —
        # finalize_analysis carries hundreds of per-NCT verdicts.
        def _fmt_args(args: dict) -> str:
            rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return rendered if len(rendered) <= 200 else rendered[:200] + "…"

        called = (
            ", ".join(f"{tc['name']}({_fmt_args(tc['args'])})" for tc in msg.tool_calls)
            or "(final)"
        )
    # logger.warn(
    #     "[LLMTURN] clinical_trials %s: %d turns, %d total output tokens, "
    #     "agent loop %.1fs",
    #     disease_name,
    #     len(ai_turns),
    #     total_out,
    #     _agent_elapsed,
    # )

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
        # A rejected call (wrong indication) returns artifact=None — skip it so it can't
        # overwrite an earlier genuine result for the same tool.
        if (
            isinstance(msg, ToolMessage)
            and msg.name in field_map
            and msg.artifact is not None
        ):
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

    # "" / None when finalize was never reached or only rejected. Unpack defensively.
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

    # Signals come from RELEVANT trials only, so supervisor and report read identical numbers.
    # Left None when finalize never ran, so the supervisor knows no relevance judgment was made
    # rather than seeing everything filtered out by an empty relevant set.
    query_unresolved = (
        output.search is not None and output.search.resolution_status == "unresolved"
    )
    if finalized and not query_unresolved:
        relevant_ids = set(output.relevant_nct_ids)
        contaminated_ids = set(output.contaminated_nct_ids)
        output.search_coverage = _derive_relevance_coverage(
            output.search,
            relevant_nct_ids=relevant_ids,
            contaminated_nct_ids=contaminated_ids,
        )
        output.completed_coverage = _derive_relevance_coverage(
            output.completed,
            relevant_nct_ids=relevant_ids,
            contaminated_nct_ids=contaminated_ids,
        )
        output.terminated_coverage = _derive_relevance_coverage(
            output.terminated,
            relevant_nct_ids=relevant_ids,
            contaminated_nct_ids=contaminated_ids,
        )
        output.signals = derive_trial_signals(
            output,
            relevant_nct_ids=relevant_ids,
            contaminated_nct_ids=contaminated_ids,
        )
        # dev_stage is an LLM judgment, not the deterministic phase-rank (which mis-encoded the
        # Phase-4 trap). Only nct/phase/status is sent; cached per trial set. The deterministic
        # dev_stage stays as the fallback when the relevant set is empty.
        #
        # completed + terminated + search, all filtered by relevant_nct_ids. Search trials —
        # where active Phase 3s live — go through the relevance gate too, so the same filter
        # applies to every scope.
        relevant_set = set(output.relevant_nct_ids)
        seen: set[str] = set()
        relevant_trials = []
        for t in (
            (output.completed.trials if output.completed else [])
            + (output.terminated.trials if output.terminated else [])
            + (output.search.trials if output.search else [])
        ):
            if t.nct_id and t.nct_id in relevant_set and t.nct_id not in seen:
                seen.add(t.nct_id)
                relevant_trials.append(t)
        if relevant_trials:
            judgment = await judge_dev_stage(
                relevant_trials,
                cache_dir,
                drug=drug_name,
                indication=disease_name,
            )
            output.signals.dev_stage = judgment.tier
            output.signals.active_programs = judgment.active_programs

            # LOAD-BEARING ORDER: judge_dev_stage MUST run before judge_ct_summary — the prose
            # is fed the resolved stage so it cannot contradict it. Do NOT reorder.
            stage_phrase = dev_stage_phrase(output.signals)
            if stage_phrase:
                ct_summary = await judge_ct_summary(
                    relevant_trials,
                    stage=stage_phrase,
                    active_programs=judgment.active_programs,
                    coverage=output.search_coverage,
                    first_approval=first_approval,
                    cache_dir=cache_dir,
                    drug=drug_name,
                    indication=disease_name,
                )
                if ct_summary is not None:
                    output.summary = ct_summary.prose
                    output.closure = ct_summary.closure
                    output.closure_reason = ct_summary.closure_reason
    return output
