from datetime import date

from langchain_core.tools import tool
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.models.model_clinical_trials import (
    WhitespaceResult,
    IndicationLandscape,
    Trial,
    TerminatedTrial,
)


def build_clinical_trials_tools(
    date_before: date | None = None, max_search_results: int = 50
) -> list:

    @tool(response_format="content_and_artifact")
    async def get_terminated(
        drug: str, indication: str
    ) -> tuple[str, list[TerminatedTrial]]:
        """Get terminated trials for a drug and indication.

        Runs two queries and returns their union:
        (1) Drug-wide safety/efficacy terminations — trials for this drug across
            ALL indications where stop_category is 'safety' or 'efficacy'. This
            count is the same regardless of which indication you pass. It reflects
            the drug's overall safety/efficacy failure history, not failures
            specific to this indication.
        (2) Indication-specific terminations — all terminated trials in this
            indication space, any drug.

        Each result includes a stop_category. When reporting counts, make clear
        that drug-wide safety/efficacy failures apply across all indications, not
        just this one.
        """
        async with ClinicalTrialsClient() as client:
            terminated = await client.get_terminated(
                drug, indication, date_before=date_before, sort="EnrollmentCount:desc"
            )

        drug_wide = [t for t in terminated if t.stop_category in {"safety", "efficacy"}]
        indication_specific = [t for t in terminated if t.stop_category not in {"safety", "efficacy"}]
        content = (
            f"Terminated trials for {drug} × {indication}: "
            f"{len(drug_wide)} drug-wide safety/efficacy failures (across all indications), "
            f"{len(indication_specific)} indication-specific terminations"
        )
        return content, terminated

    @tool(response_format="content_and_artifact")
    async def search_trials(drug: str, indication: str) -> tuple[str, list[Trial]]:
        """Search for clinical trials matching a drug and indication.

        Returns trial records including phase, status, enrollment, sponsor,
        interventions, and outcomes. Use when detect_whitespace shows trials
        exist and you need details.

        Only trials with a start date before the session cutoff are returned.
        """

        async with ClinicalTrialsClient() as client:
            trials = await client.search_trials(
                drug,
                indication,
                date_before=date_before,
                max_results=max_search_results,
                sort="EnrollmentCount:desc",
            )

        return f"Searched on {drug}-{indication} and found {len(trials)} trials", trials

    @tool(response_format="content_and_artifact")
    async def detect_whitespace(
        drug: str, indication: str
    ) -> tuple[str, WhitespaceResult]:
        """Check if a drug-indication pair has been explored in clinical trials.
        Returns whitespace signal: whether trials exist for this exact pair,
        how many trials exist for the drug alone and indication alone, and
        (when whitespace exists) other drugs being tested for the same indication.
        This should almost always be the first tool called.
        """

        async with ClinicalTrialsClient() as client:
            whitespace = await client.detect_whitespace(
                drug, indication, date_before=date_before
            )

        return (
            f"Whitespace check for {drug} × {indication}: {whitespace.is_whitespace}",
            whitespace,
        )

    @tool(response_format="content_and_artifact")
    async def get_landscape(indication: str) -> tuple[str, IndicationLandscape]:
        """Get the competitive landscape for a indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by
        phase then enrollment, plus phase distribution and recent starts.
        Use to understand how crowded the space is.
        """

        async with ClinicalTrialsClient() as client:
            landscape = await client.get_landscape(
                indication, date_before=date_before, top_n=10
            )
        return (
            f"Landscape for {indication}: {len(landscape.competitors)} competitors",
            landscape,
        )

    @tool(response_format="content_and_artifact")
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete.

        Call this as the very last step, passing your 2-3 sentence plain-text
        summary of the findings. This terminates the agent loop.
        """
        return "Analysis complete.", summary

    return [detect_whitespace, search_trials, get_landscape, get_terminated, finalize_analysis]
