"""LangChain tool wrappers for ClinicalTrialsClient methods.

Each tool is a thin adapter that makes an existing client method available
to the LLM via LangChain's tool-calling interface. Tools accept primitive
types (strings) and return dicts so the LLM can read the results.
"""

from datetime import date

from langchain_core.tools import tool

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient


def build_clinical_trials_tools(date_before: date | None = None) -> list:
    """Build tool functions with date_before captured via closure.

    Args:
        date_before: Optional temporal holdout cutoff date. Passed through
            to every client call so the agent operates on a consistent
            time window.

    Returns:
        List of LangChain tool functions.
    """

    @tool
    async def detect_whitespace(drug: str, indication: str) -> dict:
        """Check if a drug-indication pair has been explored in clinical trials.

        Returns whitespace signal: whether trials exist for this exact pair,
        how many trials exist for the drug alone and indication alone, and
        (when whitespace exists) other drugs being tested for the same indication.
        This should almost always be the first tool called.
        """
        async with ClinicalTrialsClient() as client:
            result = await client.detect_whitespace(
                drug, indication, date_before=date_before
            )
        return result.model_dump()

    @tool
    async def search_trials(drug: str, indication: str) -> list[dict]:
        """Search for clinical trials matching a drug and indication.

        Returns trial records including phase, status, enrollment, sponsor,
        interventions, and outcomes. Use when detect_whitespace shows trials
        exist and you need details.
        """
        async with ClinicalTrialsClient() as client:
            trials = await client.search_trials(
                drug, indication, date_before=date_before, max_results=50
            )
        return [t.model_dump() for t in trials]

    @tool
    async def get_landscape(indication: str) -> dict:
        """Get the competitive landscape for a indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by
        phase then enrollment, plus phase distribution and recent starts.
        Use to understand how crowded the space is.
        """
        async with ClinicalTrialsClient() as client:
            result = await client.get_landscape(
                indication, date_before=date_before, top_n=10
            )
        return result.model_dump()

    @tool
    async def get_terminated(drug: str, indication: str) -> list[dict]:
        """Get terminated trials for a drug and indication.

        Runs two queries: (1) safety/efficacy terminations for this drug across
        all indications — only stop_category 'safety' or 'efficacy' are returned,
        noise is dropped; (2) all terminations in this indication space.
        Each result includes a stop_category. Use to check for prior failures.
        """
        async with ClinicalTrialsClient() as client:
            results = await client.get_terminated(
                drug, indication, date_before=date_before
            )
        return [t.model_dump() for t in results]

    return [detect_whitespace, search_trials, get_landscape, get_terminated]
