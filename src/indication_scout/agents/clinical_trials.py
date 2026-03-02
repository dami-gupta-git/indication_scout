"""Clinical trials agent."""

from typing import Any

from indication_scout.agents.base import BaseAgent


class ClinicalTrialsAgent(BaseAgent):
    """Agent for searching and analyzing clinical trials data."""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Search clinical trials for drug-indication signal."""
        raise NotImplementedError
