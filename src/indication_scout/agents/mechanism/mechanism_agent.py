"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a MechanismOutput.
"""
import logging

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

import asyncio

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput, ShapedAssociation
from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import Association, MechanismOfAction
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a drug researcher analysing molecular targets to identify
mechanistic repurposing opportunities.

RULE 0 — GROUNDING (most important):
Every disease, disease_id, pathway, and association score you mention
anywhere in your output MUST come from a tool result in this conversation.
Copy IDs verbatim. Do not introduce any disease, target, or mechanism from
your training knowledge, even if you are certain it is correct.

TOOLS:
- get_drug — fetches the drug's mechanisms of action and target symbols
- get_target_associations — fetches top disease associations for a target, with evidence scores
- finalize_analysis — signals completion; MUST be the final tool call

PROCEDURE:
1. Call get_drug to retrieve mechanisms of action and target symbols.
2. For each unique target symbol in the mechanism results, call get_target_associations.
3. Call finalize_analysis with a plain-text summary referencing the associations returned.

NEGATIVE RESULTS:
If no strong associations are found, say so.
Do not back-fill with general knowledge about the drug.

Your summary may only reference diseases and scores returned by the tools in this run."""


def _compute_shaped_associations(
    mechanisms_of_action: list[MechanismOfAction],
    associations: dict[str, list[Association]],
) -> list[ShapedAssociation]:
    """Deterministically assign mechanistic shapes to each target-disease pair.

    Shape logic (evaluated in order):
    1. confirms_known  — clinical ≥ 0.6 and genetic_association < 0.3
    2. contraindication / hypothesis — derive disease direction from
       datatype_scores, then cross with the drug's action_type:
       - GOF signal (somatic_mutation ≥ 0.4) → disease gains function
       - LOF signal (name contains "deficiency", "loss", or "haploinsufficiency") → disease loses function
       - pathway_overactivity (affected_pathway ≥ 0.4 and somatic_mutation < 0.4) → GOF-like
       - Then: inhibitor+GOF → hypothesis; inhibitor+LOF → contraindication
               activator+LOF → hypothesis;  activator+GOF → contraindication
    3. neutral — insufficient evidence to determine direction
    """
    LOF_TERMS = ("deficiency", " loss", "haploinsufficiency")
    GOF_ACTION_TYPES = frozenset({"AGONIST", "ACTIVATOR", "OPENER", "POSITIVE MODULATOR"})
    INHIBITOR_ACTION_TYPES = frozenset({"INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE MODULATOR"})

    # Build a map from target_symbol → action_types
    symbol_action_types: dict[str, set[str]] = {}
    for moa in mechanisms_of_action:
        at = (moa.action_type or "").upper()
        for sym in moa.target_symbols:
            symbol_action_types.setdefault(sym, set()).add(at)

    shaped: list[ShapedAssociation] = []
    for target_symbol, assoc_list in associations.items():
        action_types = symbol_action_types.get(target_symbol, set())
        for assoc in assoc_list:
            ds = assoc.datatype_scores
            clinical = ds.get("clinical", 0.0) or 0.0
            genetic = ds.get("genetic_association", 0.0) or 0.0
            somatic = ds.get("somatic_mutation", 0.0) or 0.0
            pathway = ds.get("affected_pathway", 0.0) or 0.0
            disease_lower = assoc.disease_name.lower()

            # 1. confirms_known: dominated by clinical evidence, not genetic
            if clinical >= 0.6 and genetic < 0.3:
                shaped.append(ShapedAssociation(
                    target_symbol=target_symbol,
                    disease_name=assoc.disease_name,
                    disease_id=assoc.disease_id,
                    shape="confirms_known",
                    rationale=(
                        f"Clinical evidence dominates (clinical={clinical:.2f}, "
                        f"genetic={genetic:.2f}); association likely reflects known use."
                    ),
                ))
                continue

            # 2. Determine disease direction
            disease_gof = somatic >= 0.4 or (pathway >= 0.4 and somatic < 0.4)
            disease_lof = any(term in disease_lower for term in LOF_TERMS)

            if not disease_gof and not disease_lof:
                shaped.append(ShapedAssociation(
                    target_symbol=target_symbol,
                    disease_name=assoc.disease_name,
                    disease_id=assoc.disease_id,
                    shape="neutral",
                    rationale=(
                        f"Insufficient directional evidence "
                        f"(somatic={somatic:.2f}, pathway={pathway:.2f}, "
                        f"name signals: none)."
                    ),
                ))
                continue

            direction = "GOF" if disease_gof else "LOF"
            is_inhibitor = bool(action_types & INHIBITOR_ACTION_TYPES)
            is_activator = bool(action_types & GOF_ACTION_TYPES)

            if (is_inhibitor and disease_gof) or (is_activator and disease_lof):
                shape = "hypothesis"
            elif (is_inhibitor and disease_lof) or (is_activator and disease_gof):
                shape = "contraindication"
            else:
                shape = "neutral"

            at_str = ", ".join(sorted(action_types)) or "UNKNOWN"
            shaped.append(ShapedAssociation(
                target_symbol=target_symbol,
                disease_name=assoc.disease_name,
                disease_id=assoc.disease_id,
                shape=shape,
                rationale=(
                    f"{target_symbol} action_type={at_str}; disease direction={direction} "
                    f"(somatic={somatic:.2f}, pathway={pathway:.2f}, name='{assoc.disease_name}')."
                ),
            ))

    return shaped


def build_mechanism_agent(llm) -> object:
    """Return a compiled ReAct agent."""
    tools = build_mechanism_tools()
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_mechanism_agent(agent, drug_name: str) -> MechanismOutput:
    """Invoke the agent and assemble a MechanismOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyse the targets of {drug_name}")]}
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

    # Fetch pathways outside the agent loop — no LLM calls needed
    pathways: dict[str, list] = {}
    if drug_targets:
        async with OpenTargetsClient() as client:
            results = await asyncio.gather(
                *[client.get_target_data_pathways(tid) for tid in drug_targets.values()],
                return_exceptions=True,
            )
        for symbol, pathway_result in zip(drug_targets.keys(), results):
            if isinstance(pathway_result, Exception):
                logger.warning("Failed to fetch pathways for %s: %s", symbol, pathway_result)
            else:
                pathways[symbol] = pathway_result

    shaped_associations = _compute_shaped_associations(mechanisms_of_action, associations)

    return MechanismOutput(
        drug_targets=drug_targets,
        mechanisms_of_action=mechanisms_of_action,
        associations=associations,
        shaped_associations=shaped_associations,
        pathways=pathways,
        summary=summary,
    )
