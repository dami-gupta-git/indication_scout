"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a MechanismOutput.
"""

import asyncio
import logging
import time
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_candidates import (
    select_top_candidates,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.mechanism.mechanism_row_builder import (
    build_candidate_rows,
)
from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools
from indication_scout.constants import MECHANISM_TOP_CANDIDATES
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import MechanismOfAction
from indication_scout.services.approval_check import (
    get_approved_indications,
    get_fda_approved_disease_mapping,
)

logger = logging.getLogger(__name__)

# Number of top-scored associations per target we pull evidence for. Not the final candidate count —
# select_top_candidates trims to MECHANISM_TOP_CANDIDATES after filtering.
_ASSOCIATIONS_PER_TARGET = 15

SYSTEM_PROMPT = """\
You analyze a drug's molecular targets to surface disease associations for repurposing.

Steps:
1. Call get_drug to fetch the drug's mechanisms of action and targets.
2. From those targets, pick the 3 MOST CONSEQUENTIAL ones — the targets that best
   define the drug's primary pharmacology. When many targets are near-duplicate
   subunits of one complex (e.g. the NDUF* / MT-ND* subunits of mitochondrial
   Complex I), treat the complex as a single target and pick just one representative;
   do not call get_target_associations on every subunit. Call get_target_associations
   on at most 3 targets.
3. Call finalize_analysis with a 3-4 sentence plain-text summary of the findings.

Do not call get_target_associations on more than 3 targets."""


def build_mechanism_agent(llm) -> object:
    """Return a compiled ReAct agent."""
    tools = build_mechanism_tools()
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_mechanism_agent(
    agent, drug_name: str, date_before: date | None = None
) -> MechanismOutput:
    """Invoke the agent and assemble a MechanismOutput from the run.

    `date_before` gates the FDA approval drop-filter so a holdout doesn't use post-cutoff approvals to discard
    candidates that weren't yet approved at the cutoff. OpenTargets associations themselves have no date filter.
    """
    _agent_t0 = time.perf_counter()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze the targets of {drug_name}")]}
    )
    _agent_elapsed = time.perf_counter() - _agent_t0

    # Per-turn LLM accounting (same as the literature agent). Each AIMessage is one
    # round-trip; usage_metadata gives context size and output tokens. Logged at
    # WARNING to isolate whether mechanism's time is round-trips (and how many
    # get_target_associations calls the LLM actually made vs. the 3-target cap) or
    # the post-agent _assemble_candidates fan-out. Read-only on result["messages"].
    ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    total_out = 0
    for i, msg in enumerate(ai_turns):
        usage = msg.usage_metadata or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        total_out += out_tok
        called = ", ".join(tc["name"] for tc in msg.tool_calls) or "(final)"
        logger.warning(
            "[LLMTURN] mechanism %s turn %d/%d: in=%d out=%d -> %s",
            drug_name, i + 1, len(ai_turns), in_tok, out_tok, called,
        )
    logger.warning(
        "[LLMTURN] mechanism %s: %d turns, %d total output tokens, agent loop %.1fs",
        drug_name, len(ai_turns), total_out, _agent_elapsed,
    )

    mechanisms_of_action: list[MechanismOfAction] = []
    associations: dict[str, list] = {}
    summary: str = ""

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "get_drug":
            mechanisms_of_action = msg.artifact or []

        elif msg.name == "get_target_associations":
            associations.update(msg.artifact or {})

        elif msg.name == "finalize_analysis":
            summary = msg.artifact or ""

    # Derive drug_targets from mechanisms_of_action (already fetched by get_drug)
    drug_targets: dict[str, str] = {
        symbol: target_id
        for moa in mechanisms_of_action
        for symbol, target_id in zip(moa.target_symbols, moa.target_ids)
    }

    # Restrict the candidate fan-out to the targets the LLM actually chose
    # (the symbols it called get_target_associations on). The agent is prompted to
    # pick the 3 most consequential targets; honoring that here avoids fetching
    # per-disease evidence for every target — the dominant OT-call cost. Fall back
    # to all targets if the LLM selected none, so a degenerate run still produces
    # candidates.
    chosen_symbols = set(associations.keys())
    assemble_targets = {
        symbol: target_id
        for symbol, target_id in drug_targets.items()
        if symbol in chosen_symbols
    } or drug_targets

    all_mech_diseases = {
        a.disease_name for assoc_list in associations.values() for a in assoc_list
    }
    logger.warning(
        "[MECH] surfaced %d diseases: %s",
        len(all_mech_diseases),
        sorted(all_mech_diseases),
    )

    _asm_t0 = time.perf_counter()
    candidates = await _assemble_candidates(
        drug_name, assemble_targets, mechanisms_of_action, date_before=date_before
    )
    logger.warning(
        "[TIMING] mechanism %s: _assemble_candidates (%d of %d targets) took %.1fs",
        drug_name, len(assemble_targets), len(drug_targets), time.perf_counter() - _asm_t0,
    )
    logger.warning(f"mechanism_agent SUMMARY: {summary}")
    return MechanismOutput(
        drug_targets=drug_targets,
        mechanisms_of_action=mechanisms_of_action,
        candidates=candidates,
        summary=summary,
    )


async def _assemble_candidates(
    drug_name: str,
    drug_targets: dict[str, str],
    mechanisms_of_action: list[MechanismOfAction],
    date_before: date | None = None,
) -> list:
    """Fetch per-target rows, filter approved indications, classify.

    Returns `[]` for any failure path (unresolvable drug, no MoAs, no targets) so the agent keeps
    returning a valid MechanismOutput.
    """
    if not drug_targets or not mechanisms_of_action:
        return []

    # symbol → set of upper-cased action types, drawn from MoA entries.
    symbol_to_actions: dict[str, set[str]] = {}
    for moa in mechanisms_of_action:
        action = (moa.action_type or "").upper()
        if not action:
            continue
        for sym in moa.target_symbols:
            symbol_to_actions.setdefault(sym, set()).add(action)

    async with OpenTargetsClient() as ot_client:
        per_target_rows = await asyncio.gather(
            *[
                build_candidate_rows(
                    ot_client,
                    target_id,
                    symbol_to_actions.get(symbol, set()),
                    _ASSOCIATIONS_PER_TARGET,
                )
                for symbol, target_id in drug_targets.items()
            ],
            return_exceptions=True,
        )

    rows: list[dict] = []
    for symbol, result in zip(drug_targets.keys(), per_target_rows):
        if isinstance(result, Exception):
            logger.warning(
                "_assemble_candidates: row build failed for %s: %s", symbol, result
            )
            continue
        rows.extend(result)

    if not rows:
        return []

    # FDA approval filter. On any failure, fall back to an empty approved set so at least the biology
    # filter runs — better than dropping candidates silently on a chembl / fda hiccup.
    # Holdout swap: when date_before is set, the live openFDA labels would use post-cutoff approvals
    # to drop candidates that weren't yet approved at the cutoff (e.g. dropping imatinib × systemic
    # mastocytosis in a 2002 holdout on the strength of the 2006 approval). Use the hardcoded
    # date-gated approvals table instead, mirroring the supervisor / clinical-trials call sites.
    approved: set[str] = set()
    candidate_names = sorted({r["disease_name"] for r in rows if r.get("disease_name")})
    try:
        if candidate_names:
            if date_before is not None:
                approved = await get_approved_indications(
                    drug_name=drug_name,
                    candidate_diseases=candidate_names,
                    as_of=date_before,
                )
            else:
                mapping = await get_fda_approved_disease_mapping(
                    drug_name=drug_name,
                    candidate_diseases=candidate_names,
                )
                approved = {
                    disease for disease, is_approved in mapping.items() if is_approved
                }
    except Exception as e:
        logger.warning(
            "_assemble_candidates: FDA approval check failed for %r: %s; "
            "proceeding without approval filter",
            drug_name,
            e,
        )

    return select_top_candidates(
        rows, approved_diseases=approved, limit=MECHANISM_TOP_CANDIDATES
    )
