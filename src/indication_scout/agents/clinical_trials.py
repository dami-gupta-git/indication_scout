"""Clinical trials agent.

ReAct agent that uses ClinicalTrials.gov to assess whether a drug-disease
pair is being explored, how crowded the space is, and whether prior
attempts failed. Returns a ClinicalTrialsOutput with structured data
and a natural language summary.
"""

import json
import logging
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from indication_scout.agents.base import BaseAgent
from indication_scout.agents.clinical_trials_tools import build_clinical_trials_tools
from indication_scout.agents.model_clinical_trials_agent import ClinicalTrialsOutput
from indication_scout.constants import DEFAULT_LLM_MODEL
from indication_scout.models.model_clinical_trials import (
    IndicationLandscape,
    TerminatedTrial,
    Trial,
    WhitespaceResult,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical trials analyst assessing whether a drug \
could be repurposed for a new indication.

Given a drug and disease, investigate the clinical trial landscape:
1. Start with detect_whitespace to check if trials exist for this pair
2. If trials exist: use search_trials to get details, then get_landscape \
for the competitive picture
3. If whitespace (no trials): use get_terminated to check for prior \
failures, then get_landscape to see what else is being developed
4. If get_terminated finds trials stopped for safety or efficacy, that is \
a strong negative signal — you may skip get_landscape

End with a 2-3 sentence assessment summarizing:
- Whether trials exist for this drug-disease pair
- The competitive landscape (crowded vs. open)
- Any red flags from terminated trials (safety/efficacy failures)
- Your overall assessment of the opportunity

Return your assessment as plain text in your final message."""

RECURSION_LIMIT = 15


class ClinicalTrialsAgent(BaseAgent):
    """Agent that assesses clinical trial evidence for a drug-disease pair."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL) -> None:
        self._model = model

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the clinical trials analysis.

        Args:
            input_data: Must contain 'drug_name' and 'disease_name'.
                Optionally contains 'date_before' (date) for temporal holdout.

        Returns:
            Dict with 'clinical_trials_output' key containing a
            ClinicalTrialsOutput instance.
        """
        drug_name: str = input_data["drug_name"].lower()
        disease_name: str = input_data["disease_name"].lower()
        date_before: date | None = input_data.get("date_before")

        tools = build_clinical_trials_tools(date_before=date_before)

        llm = ChatAnthropic(
            model=self._model,
            temperature=0,
            max_tokens=4096,
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

        user_message = (
            f"Analyze the clinical trial landscape for the drug '{drug_name}' "
            f"in the indication '{disease_name}'."
        )

        logger.info(
            "Running ClinicalTrialsAgent for drug=%s, disease=%s",
            drug_name,
            disease_name,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        output = self._parse_result(result)

        logger.info(
            "ClinicalTrialsAgent complete for drug=%s, disease=%s: "
            "trials=%d, whitespace=%s, terminated=%d",
            drug_name,
            disease_name,
            len(output.trials),
            output.whitespace.is_whitespace if output.whitespace else "N/A",
            len(output.terminated),
        )

        return {"clinical_trials_output": output}

    @staticmethod
    def _parse_result(result: dict[str, Any]) -> ClinicalTrialsOutput:
        """Extract structured data from the agent's message history.

        Walks through all messages looking for tool responses. Parses
        each tool's output back into the corresponding Pydantic model
        and captures the final AI message as the summary.
        """
        trials: list[Trial] = []
        whitespace: WhitespaceResult | None = None
        landscape: IndicationLandscape | None = None
        terminated: list[TerminatedTrial] = []
        summary = ""

        messages = result.get("messages", [])

        for msg in messages:
            # Tool response messages have a .name attribute indicating which tool
            if hasattr(msg, "name") and hasattr(msg, "content"):
                tool_name = msg.name
                content = msg.content

                # LangChain may store tool responses as JSON strings
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        continue

                if tool_name == "detect_whitespace" and isinstance(content, dict):
                    whitespace = WhitespaceResult(**content)

                elif tool_name == "search_trials" and isinstance(content, list):
                    trials = [Trial(**t) for t in content if isinstance(t, dict)]

                elif tool_name == "get_landscape" and isinstance(content, dict):
                    landscape = IndicationLandscape(**content)

                elif tool_name == "get_terminated" and isinstance(content, list):
                    terminated = [
                        TerminatedTrial(**t) for t in content if isinstance(t, dict)
                    ]

        # The last AI message (not a tool message) is the summary.
        # Note: LangChain AIMessage has .name = None, while ToolMessage
        for msg in reversed(messages):
            if hasattr(msg, "content") and not getattr(msg, "name", None):
                # AIMessage — extract text content
                content = msg.content
                if isinstance(content, str) and content.strip():
                    summary = content.strip()
                    break
                # Some AIMessages have content as a list of blocks
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if text_parts:
                        summary = "\n".join(text_parts).strip()
                        break

        return ClinicalTrialsOutput(
            trials=trials,
            whitespace=whitespace,
            landscape=landscape,
            terminated=terminated,
            summary=summary,
        )
