"""Literature agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a LiteratureOutput.
"""

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

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
):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_literature_tools(
        svc,
        db,
        date_before=date_before,
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


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

    # Walk the message history and pull each tool's typed artifact off msg.artifact
    artifacts: dict = {
        "queries": [],
        "pmids": [],
        "abstracts": [],
        "evidence": None,
        "summary": "",
    }
    # maps tool names → keys in the local artifacts dict , used for mapping to LiteratureOutput
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
