"""Literature agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a LiteratureOutput.
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents._react_loop import (
    cached_system_message,
    history_cache_pre_model_hook,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.literature.literature_tools import build_literature_tools

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

SYSTEM_PROMPT = (_PROMPTS_DIR / "literature.txt").read_text()


def build_literature_agent(
    llm,
    svc,
    db,
    date_before=None,
    approved_indications=None,
):
    """Return a compiled ReAct agent. No graph wiring required.

    `approved_indications` is the drug's FDA-approved indication list, forwarded to the synthesize
    tool so the strength judge can exclude approved-sub-indication papers from a broad candidate.
    """
    tools = build_literature_tools(
        svc,
        db,
        date_before=date_before,
        approved_indications=approved_indications,
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=cached_system_message(SYSTEM_PROMPT),
        pre_model_hook=history_cache_pre_model_hook,
    )


async def run_literature_agent(
    agent, drug_name: str, disease_name: str
) -> LiteratureOutput:
    """Invoke the agent and assemble a LiteratureOutput from the run."""
    logger.debug(
        "Starting literature agent run for drug=%s disease=%s", drug_name, disease_name
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
    )

    # Per-turn LLM accounting. Each AIMessage is one round-trip to the model; its
    # usage_metadata gives input tokens (context re-sent that turn) and output
    # tokens (what it generated). Logged at WARNING so the round-trip breakdown
    # surfaces — isolates whether loop overhead is many turns, large output, or
    # context growth. Pure read of the returned history; no loop logic touched.
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
    #     logger.info(
    #         "[LLMTURN] %s turn %d/%d: in=%d out=%d cache_read=%d cache_write=%d -> %s",
    #         disease_name,
    #         i + 1,
    #         len(ai_turns),
    #         in_tok,
    #         out_tok,
    #         cache_read,
    #         cache_write,
    #         called,
    #     )
    # logger.info(
    #     "[LLMTURN] %s: %d turns, %d total output tokens",
    #     disease_name,
    #     len(ai_turns),
    #     total_out,
    # )

    # Pull each tool's typed artifact off msg.artifact
    artifacts: dict = {
        "queries": [],
        "pmids": [],
        "abstracts": [],
        "evidence": None,
        "summary": "",
    }
    # tool names → keys in the artifacts dict, for mapping to LiteratureOutput
    field_map = {
        "expand_search_terms": "queries",
        "fetch_and_cache": "pmids",
        "semantic_search": "abstracts",
        "synthesize": "evidence",
        "finalize_analysis": "summary",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    if not artifacts["queries"]:
        logger.warning(
            "Literature agent produced no expanded search queries for %s/%s",
            drug_name,
            disease_name,
        )
    if not artifacts["pmids"]:
        logger.warning(
            "Literature agent fetched no PMIDs for %s/%s", drug_name, disease_name
        )
    if artifacts["evidence"] is None:
        logger.warning(
            "Literature agent produced no EvidenceSummary for %s/%s",
            drug_name,
            disease_name,
        )

    return LiteratureOutput(
        search_results=artifacts["queries"],
        pmids=artifacts["pmids"],
        semantic_search_results=artifacts["abstracts"],
        evidence_summary=artifacts["evidence"],
        summary=artifacts["summary"],
    )
