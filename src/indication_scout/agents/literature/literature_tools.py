"""Literature tools — middle-ground version.

Uses content_and_artifact so typed Python objects are preserved on
msg.artifact. Tools share inter-call data via a closure-scoped store dict.
No InjectedState, no LangGraph state machinery.
"""

from datetime import date

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult, RetrievalService


def build_literature_tools(
    svc: RetrievalService,
    db: Session,
    date_before: date | None = None,
    max_search_results: int | None = None,
    num_top_k: int = 5,
) -> list:
    """Build tools that share data via a closure-scoped store dict.

    Tools write to the store themselves as a side effect, so subsequent
    tools can read prior results without the LLM passing them around.
    """

    store: dict = {}

    @tool(response_format="content_and_artifact")
    async def build_drug_profile(drug_name: str) -> tuple[str, DrugProfile]:
        """Fetch pharmacological profile (synonyms, gene targets, mechanisms)
        for a drug. Call before expand_search_terms for richer queries."""
        profile = await svc.build_drug_profile(drug_name)
        store["drug_profile"] = profile
        return f"Profile for {drug_name}: {len(profile.synonyms)} synonyms", profile

    @tool(response_format="content_and_artifact")
    async def expand_search_terms(
        drug_name: str, disease_name: str
    ) -> tuple[str, list[str]]:
        """Generate diverse PubMed keyword queries. Uses the drug profile
        if available, otherwise builds one on the fly."""
        profile = store.get("drug_profile") or await svc.build_drug_profile(drug_name)
        queries = await svc.expand_search_terms(drug_name, disease_name, profile)
        store["queries"] = queries
        return f"Generated {len(queries)} queries", queries

    @tool(response_format="content_and_artifact")
    async def fetch_and_cache(drug_name: str) -> tuple[str, list[str]]:
        """Run PubMed queries, fetch abstracts, embed, cache in pgvector."""
        queries = store.get("queries", [])
        if not queries:
            return "No queries — call expand_search_terms first.", []
        pmids = await svc.fetch_and_cache(
            queries, db, date_before=date_before, max_results=max_search_results
        )
        store["pmids"] = pmids
        return f"Fetched {len(pmids)} PMIDs", pmids

    @tool(response_format="content_and_artifact")
    async def semantic_search(
        drug_name: str, disease_name: str
    ) -> tuple[str, list[AbstractResult]]:
        """Re-rank cached abstracts by similarity to the drug-disease pair."""
        pmids = store.get("pmids", [])
        if not pmids:
            return "No PMIDs — call fetch_and_cache first.", []
        results = await svc.semantic_search(
            disease_name, drug_name, pmids, db, top_k=num_top_k
        )
        store["abstracts"] = results
        top = results[0].similarity if results else 0.0
        return f"Found {len(results)} abstracts (top sim: {top:.2f})", results

    @tool(response_format="content_and_artifact")
    async def synthesize(
        drug_name: str, disease_name: str
    ) -> tuple[str, EvidenceSummary]:
        """Synthesize abstracts into a structured evidence summary."""
        abstracts = store.get("abstracts", [])
        evidence = await svc.synthesize(drug_name, disease_name, abstracts)
        return f"Evidence strength: {evidence.strength}", evidence

    @tool(response_format="content_and_artifact")
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete.

        Call this as the very last step, passing your 3-4 sentence plain-text
        summary of the findings. This terminates the agent loop.
        """
        store["final_summary"] = summary
        return "Analysis complete.", summary

    return [
        build_drug_profile,
        expand_search_terms,
        fetch_and_cache,
        semantic_search,
        synthesize,
        finalize_analysis,
    ]
