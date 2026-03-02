"""Safety analysis agent."""

from typing import Any

from indication_scout.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    """Agent for analyzing drug safety profiles."""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze safety data for drug-indication combinations."""
        raise NotImplementedError
