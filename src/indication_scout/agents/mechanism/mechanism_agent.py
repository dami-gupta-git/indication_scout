"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a MechanismOutput.
"""

import asyncio
import logging
from datetime import date

from langchain_core.messages import HumanMessage, ToolMessage
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

SYSTEM_PROMPT = ""


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
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze the targets of {drug_name}")]}
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

    all_mech_diseases = {
        a.disease_name for assoc_list in associations.values() for a in assoc_list
    }
    logger.warning(
        "[MECH] surfaced %d diseases: %s",
        len(all_mech_diseases),
        sorted(all_mech_diseases),
    )

    candidates = await _assemble_candidates(
        drug_name, drug_targets, mechanisms_of_action, date_before=date_before
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
                approved = get_approved_indications(
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
