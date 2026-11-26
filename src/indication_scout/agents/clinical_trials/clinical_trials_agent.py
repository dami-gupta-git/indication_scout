"""Clinical Trials agent

Gated ReAct loop (agents/_react_loop.py) ending as soon as finalize_analysis succeeds. After
the run, pulls typed artifacts off the ToolMessages into a ClinicalTrialsOutput.
"""

import logging
import time
from datetime import date
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from indication_scout.agents._react_loop import (
    _trailing_tool_messages,
    build_gated_react_loop,
)
from indication_scout.agents._trial_signals import (
    derive_trial_signals,
    superseded_registrations,
)
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
from indication_scout.services.approval_check import ApprovalLabel
from indication_scout.services.clinical_trials_summary import judge_ct_summary
from indication_scout.services.dev_stage import dev_stage_phrase, judge_dev_stage
from indication_scout.services.trial_relevance import classify_trials

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()


def _finalize_done(messages: list) -> bool:
    """End the loop once finalize_analysis succeeds; a rejected finalize returns an empty artifact and retries."""
    for m in _trailing_tool_messages(messages):
        if m.name == "finalize_analysis" and m.artifact:
            return True
    return False


def build_clinical_trials_agent(
    llm: BaseChatModel,
    date_before: date | None = None,
    assigned_indication: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    approval_label: ApprovalLabel | None = None,
) -> CompiledStateGraph:
    """Return a compiled ReAct agent.

    `assigned_indication` pins the tools to one indication; calls for any other are soft-rejected.
    `approval_label` is the candidate step's FDA verdict for the pair; required in a live run (see check_fda_approval).
    """
    tools = build_clinical_trials_tools(
        date_before=date_before,
        assigned_indication=assigned_indication,
        cache_dir=cache_dir,
        approval_label=approval_label,
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
    agent: CompiledStateGraph,
    drug_name: str,
    disease_name: str,
    first_approval: int | None = None,
    approved_indications: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run.

    `first_approval` (ChEMBL) lets the closure judgment tell an old generic from a genuine negative; None renders
    "unknown", never a default year. `approved_indications` feeds the per-trial approved-subtype check; empty
    disables it.
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
    # ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    # total_out = 0
    # for i, msg in enumerate(ai_turns):
    #     usage = msg.usage_metadata or {}
    #     in_tok = usage.get("input_tokens", 0)
    #     out_tok = usage.get("output_tokens", 0)
    #     # cache_read==0 across turns 2+ means a silent cache invalidator. langchain-anthropic
    #     # reports fresh writes under the ephemeral keys, not cache_creation.
    #     details = usage.get("input_token_details", {})
    #     cache_read = details.get("cache_read", 0)
    #     cache_write = (
    #         details.get("ephemeral_5m_input_tokens", 0)
    #         + details.get("ephemeral_1h_input_tokens", 0)
    #     ) or details.get("cache_creation", 0)
    #     total_out += out_tok

    #     # Args show what each retry queries, not just that a tool re-ran. Truncated —
    #     # finalize_analysis carries hundreds of per-NCT verdicts.
    #     def _fmt_args(args: dict) -> str:
    #         rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    #         return rendered if len(rendered) <= 200 else rendered[:200] + "…"

    #     called = (
    #         ", ".join(f"{tc['name']}({_fmt_args(tc['args'])})" for tc in msg.tool_calls)
    #         or "(final)"
    #     )
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
        # A rejected call (wrong indication) returns artifact=None; skip it so it can't overwrite an earlier result.
        if (
            isinstance(msg, ToolMessage)
            and msg.name in field_map
            and msg.artifact is not None
        ):
            artifacts[field_map[msg.name]] = msg.artifact

    if artifacts["approval"] is None:
        logger.warning(
            "clinical_trials_agent: %s × %s — check_fda_approval was not called "
            "(prompt requires it as step 1)",
            drug_name,
            disease_name,
        )

    # finalize_analysis only ends the loop; no artifact means the agent never got there.
    finalized = isinstance(artifacts.get("finalize"), FinalizeClinicalTrialsArtifact)
    if not finalized:
        logger.warning(
            "clinical_trials_agent: %s × %s — finalize_analysis produced no artifact; "
            "relevance, signals, and summary will be empty",
            drug_name,
            disease_name,
        )

    output = ClinicalTrialsOutput(
        search=artifacts["search"],
        completed=artifacts["completed"],
        terminated=artifacts["terminated"],
        landscape=artifacts["landscape"],
        approval=artifacts["approval"],
    )

    # Signals come from relevant trials only. Left None when finalize never ran, so a missing relevance judgment
    # is not mistaken for an empty relevant set.
    query_unresolved = (
        output.search is not None and output.search.resolution_status == "unresolved"
    )
    if finalized and not query_unresolved:
        all_trials = [
            t
            for result in (output.completed, output.terminated, output.search)
            for t in (result.trials if result else [])
        ]
        # Every shown trial is classified by four isolated, cached per-trial checks (services.trial_relevance), not by
        # the agent: a batch verdict depended on the other trials in the batch and moved between runs.
        relevance = await classify_trials(
            drug_name, disease_name, all_trials, approved_indications or [], cache_dir
        )
        output.relevant_nct_ids = relevance.ncts("relevant")
        output.contaminated_nct_ids = relevance.ncts("excluded")
        output.combination_nct_ids = relevance.ncts("combination")

        # Collapse sibling registrations of one protocol (EU per-site/per-indication registrations) before anything
        # counts or describes the relevant set: coverage counts, the derived signals, the prose and the report's
        # trial listings all key off relevant_nct_ids.
        superseded = superseded_registrations(
            [t for t in all_trials if t.nct_id in set(output.relevant_nct_ids)]
        )
        if superseded:
            logger.info(
                "[CT] %s x %s: collapsed %d superseded registration(s): %s",
                drug_name,
                disease_name,
                len(superseded),
                superseded,
            )
            output.relevant_nct_ids = [
                n for n in output.relevant_nct_ids if n not in superseded
            ]
        output.relevance_reasoning = (
            f"Relevant trials: {', '.join(output.relevant_nct_ids) or 'none'}. "
            f"Excluded trials: {', '.join(output.contaminated_nct_ids) or 'none'}."
            + (
                f" Combination-only trials: {', '.join(output.combination_nct_ids)}."
                if output.combination_nct_ids
                else ""
            )
        )

        # Combination-only trials count as classified-and-excluded for coverage but stay apart from
        # contaminated_nct_ids, so the report does not call them a different indication.
        combination = set(output.combination_nct_ids)
        relevant_ids = set(output.relevant_nct_ids)
        contaminated_ids = set(output.contaminated_nct_ids) | combination
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
        output.signals.superseded_registrations = superseded
        # dev_stage is an LLM judgment over nct/phase/status, cached per trial set; the deterministic dev_stage is
        # the fallback when the relevant set is empty. Completed, terminated and search trials are all filtered by
        # relevant_nct_ids.
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

            # Order matters: judge_dev_stage must run before judge_ct_summary, which is fed the resolved stage.
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
