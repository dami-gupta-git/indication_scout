"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes a single tool the
supervisor can call. The tool runs the full sub-agent and returns its
typed output as an artifact, plus a short summary string for the LLM.

There's also a find_candidates tool that hits Open Targets directly to
surface disease candidates for a drug.
"""

import logging
import time
from collections import Counter

from langchain_core.tools import tool
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.services.retrieval import RetrievalService


def build_supervisor_tools(
    llm,
    svc: RetrievalService,
    db: Session,
) -> list:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and
    reused across calls — no need to rebuild them per invocation.
    """

    # Build sub-agents once at supervisor construction (except literature — see below)
    ct_agent = build_clinical_trials_agent(llm=llm)
    mech_agent = build_mechanism_agent(llm=llm)

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Uses Open Targets to find diseases where competitor drugs (drugs
        sharing the same molecular targets) are being developed. Returns
        a list of disease names ranked by competitor activity.
        """
        competitors = await svc.get_drug_competitors(drug_name)
        diseases = list(competitors.keys())
        return f"Found {len(diseases)} candidate diseases for {drug_name}", diseases

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks
        abstracts, and produces a structured evidence summary with strength
        rating (none / weak / moderate / strong).
        """
        # Build a fresh agent per call so the closure-scoped store dict in
        # literature_tools is not shared across disease invocations.
        lit_agent = build_literature_agent(llm=llm, svc=svc, db=db)
        t0 = time.perf_counter()
        output = await run_literature_agent(lit_agent, drug_name, disease_name)
        logger.info("analyze_literature took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)
        strength = (
            output.evidence_summary.strength
            if output.evidence_summary
            else "no data"
        )
        summary = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}"
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_clinical_trials(
        drug_name: str, disease_name: str
    ) -> tuple[str, ClinicalTrialsOutput]:
        """Run a full clinical trials analysis for a drug-disease pair.

        Checks ClinicalTrials.gov for existing trials, competitive landscape,
        and terminated trials (safety/efficacy red flags).
        """
        t0 = time.perf_counter()
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)
        logger.info("analyze_clinical_trials took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)
        n_trials = len(output.trials)
        n_terminated = len(output.terminated)
        summary = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_trials} active/completed, {n_terminated} terminated"
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Analyse the molecular targets of a drug to provide mechanistic context.

        Fetches the drug's targets, their top disease associations (with evidence
        scores by type), and their Reactome pathways. Drug-level — call once per
        drug, not once per candidate disease.
        """
        t0 = time.perf_counter()
        output = await run_mechanism_agent(mech_agent, drug_name)
        logger.info("analyze_mechanism took %.2fs for %s", time.perf_counter() - t0, drug_name)
        n_targets = len(output.drug_targets)
        n_top = sum(len(a) for a in output.associations.values())
        shape_counts = Counter(s.shape for s in output.shaped_associations)
        shaped_lines = "\n".join(
            f"  [{s.shape.upper()}] {s.target_symbol} / {s.disease_name} ({s.disease_id}): {s.rationale}"
            for s in output.shaped_associations
        )
        summary = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets, "
            f"top {n_top} disease associations shown (capped per target).\n"
            f"Shaped associations: {dict(shape_counts)}\n"
            f"{shaped_lines}"
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(summary: str) -> tuple[str, str]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step, passing your 4-6 sentence plain-text
        summary of the most promising candidates. This terminates the agent loop.
        """
        return "Supervisor analysis complete.", summary

    return [find_candidates, analyze_mechanism, analyze_literature, analyze_clinical_trials, finalize_supervisor]
