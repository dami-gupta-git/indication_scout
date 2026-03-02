"""Orchestrator agent for coordinating other agents."""

from typing import Any

from indication_scout.agents.base import BaseAgent


class Orchestrator(BaseAgent):
    """Coordinates multiple agents to analyze drug indications."""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute orchestrated analysis."""
        raise NotImplementedError
