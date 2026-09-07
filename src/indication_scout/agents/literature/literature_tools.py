"""Literature tools — middle-ground version.

Uses content_and_artifact so typed Python objects are preserved on msg.artifact. Tools share
inter-call data via a closure-scoped store dict. No InjectedState, no LangGraph state machinery.
"""

import asyncio
import logging
import time
from datetime import date

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.config import get_settings
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.models.model_drug_profile import DrugProfile

_settings = get_settings()
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult, RetrievalService

logger = logging.getLogger(__name__)


def build_literature_tools(
    svc: RetrievalService,
    db: Session,
    date_before: date | None = None,
    approved_indications: list[str] | None = None,
    drug_profile: DrugProfile | None = None,
) -> list:
    """Build tools that share data via a closure-scoped store dict.

    Tools write to the store themselves as a side effect, so subsequent tools can read prior results
    without the LLM passing them around.

    `approved_indications` is the drug's FDA-approved indication list, threaded into synthesize so
    the strength judge can exclude papers about an approved sub-indication of a broad candidate.
    `drug_profile` seeds this candidate's local store when the supervisor already built it.
    """

    store: dict = {}
    if drug_profile is not None:
        store["drug_profile"] = drug_profile
        if drug_profile.chembl_id:
            store["chembl_id"] = drug_profile.chembl_id

    async def _get_chembl(drug_name: str) -> str:
        chembl_id = store.get("chembl_id")
        if chembl_id is None:
            chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
            store["chembl_id"] = chembl_id
        return chembl_id

    @tool(response_format="content_and_artifact")
    async def build_drug_profile(drug_name: str) -> tuple[str, DrugProfile]:
        """Fetch pharmacological profile (gene targets, mechanisms, ATC codes) for a drug. Call
        before expand_search_terms for richer queries."""
        _t0 = time.perf_counter()
        profile = store.get("drug_profile")
        if profile is None:
            chembl_id = await _get_chembl(drug_name)
            profile = await svc.build_drug_profile(chembl_id)
            store["drug_profile"] = profile
        else:
            chembl_id = profile.chembl_id
        logger.info(
            "[TIMING] build_drug_profile %s: %.1fs",
            drug_name,
            time.perf_counter() - _t0,
        )
        return (
            f"Profile for {drug_name} ({chembl_id}): "
            f"{len(profile.target_gene_symbols)} targets, "
            f"{len(profile.mechanisms_of_action)} mechanisms",
            profile,
        )

    @tool(response_format="content_and_artifact")
    async def expand_search_terms(
        drug_name: str, disease_name: str
    ) -> tuple[str, list[str]]:
        """Generate diverse PubMed keyword queries. Uses the drug profile if available, otherwise
        builds one on the fly."""
        _t0 = time.perf_counter()
        chembl_id = await _get_chembl(drug_name)
        profile = store.get("drug_profile")
        if profile is None:
            profile = await svc.build_drug_profile(chembl_id)
            store["drug_profile"] = profile
        queries = await svc.expand_search_terms(chembl_id, disease_name, profile)
        store["queries"] = queries
        # logger.warning(
        #     "[TIMING] expand_search_terms %s: %.1fs",
        #     disease_name,
        #     time.perf_counter() - _t0,
        # )
        return f"Generated {len(queries)} queries", queries

    @tool(response_format="content_and_artifact")
    async def fetch_and_cache(drug_name: str) -> tuple[str, list[str]]:
        """Run PubMed queries, fetch abstracts, embed, cache in pgvector."""
        queries = store.get("queries", [])
        if not queries:
            return "No queries — call expand_search_terms first.", []
        _t0 = time.perf_counter()
        # logger.warning(
        #     "[INVEST] fetch_and_cache %s: %d queries -> PubMed", drug_name, len(queries)
        # )
        pmids = await svc.fetch_and_cache(queries, db, date_before=date_before)
        store["pmids"] = pmids
        # logger.warning(
        #     "[TIMING] fetch_and_cache %s: %.1fs", drug_name, time.perf_counter() - _t0
        # )
        return f"Fetched {len(pmids)} PMIDs", pmids

    @tool(response_format="content_and_artifact")
    async def semantic_search(
        drug_name: str, disease_name: str
    ) -> tuple[str, list[AbstractResult]]:
        """Re-rank cached abstracts by similarity to the drug-disease pair."""
        pmids = store.get("pmids", [])
        if not pmids:
            return "No PMIDs — call fetch_and_cache first.", []
        _t0 = time.perf_counter()
        chembl_id = await _get_chembl(drug_name)
        results = await svc.semantic_search(
            disease_name, chembl_id, pmids, db, date_before=date_before
        )
        store["abstracts"] = results
        # logger.warning(
        #     "[TIMING] semantic_search %s: %.1fs",
        #     disease_name,
        #     time.perf_counter() - _t0,
        # )
        top = results[0].similarity if results else 0.0
        return f"Found {len(results)} abstracts (top sim: {top:.2f})", results

    @tool(response_format="content_and_artifact")
    async def safety_search(
        drug_name: str, disease_name: str
    ) -> tuple[str, EvidenceSummary]:
        """Summarize the drug's safety signal from its adverse-event literature (PubMed, ranked by
        citation count), anchored on OpenTargets warnings/adverse-events as ground truth in
        production. Uses the drug profile (builds one if needed). REQUIRED step — call before
        synthesize."""
        chembl_id = await _get_chembl(drug_name)
        drug_profile = store.get("drug_profile") or await svc.build_drug_profile(
            chembl_id
        )
        store["drug_profile"] = drug_profile
        safety_results = await svc.safety_search(
            chembl_id, date_before=date_before, disease=disease_name
        )
        safety = await svc.summarize_safety(
            chembl_id,
            disease_name,
            drug_profile,
            safety_results.combined,
            date_before=date_before,
        )
        # Disease-specific: does the safety literature report a harm for THIS indication?
        harm, harm_summary, harm_pmids = await svc.classify_indication_harm(
            chembl_id, disease_name, safety_results.disease_scoped
        )
        store["safety_summary"] = safety.safety_summary
        store["regulatory_safety_summary"] = safety.regulatory_summary
        store["regulatory_safety_full_labels"] = safety.regulatory_full_labels
        store["pharmacovigilance_summary"] = safety.pharmacovigilance_summary
        store["literature_safety_summary"] = safety.literature_summary
        store["label_safety_available"] = safety.label_data_available
        store["safety_pmids"] = safety.safety_pmids
        store["safety_severity"] = safety.safety_severity
        store["indication_harm"] = harm
        store["indication_harm_summary"] = harm_summary
        store["indication_harm_pmids"] = harm_pmids
        content = (
            f"Safety evidence ({safety.safety_severity or 'severity unavailable'}): "
            f"{safety.safety_summary}"
            if safety.safety_summary
            else "No drug-wide safety evidence available from the permitted sources."
        )
        if harm:
            content += f" | Indication-specific harm reported: {harm_summary}"
        return content, EvidenceSummary(
            safety_summary=safety.safety_summary,
            regulatory_safety_summary=safety.regulatory_summary,
            regulatory_safety_full_labels=safety.regulatory_full_labels,
            pharmacovigilance_summary=safety.pharmacovigilance_summary,
            literature_safety_summary=safety.literature_summary,
            label_safety_available=safety.label_data_available,
            safety_pmids=safety.safety_pmids,
            safety_severity=safety.safety_severity,
            indication_harm=harm,
            indication_harm_summary=harm_summary,
            indication_harm_pmids=harm_pmids,
        )

    @tool(response_format="content_and_artifact")
    async def synthesize(
        drug_name: str, disease_name: str
    ) -> tuple[str, EvidenceSummary]:
        """Synthesize abstracts into a structured evidence summary."""
        abstracts = store.get("abstracts", [])
        _t0 = time.perf_counter()
        chembl_id = await _get_chembl(drug_name)
        evidence = await svc.synthesize(
            chembl_id,
            disease_name,
            abstracts,
            approved_indications=approved_indications,
        )
        # Merge in the safety pass's result (see safety_search) — ONLY if safety_search has already
        # run and populated the store. safety_search is REQUIRED-before-synthesize by the prompt,
        # but the LLM controls ordering and could batch synthesize first; overwriting with defaults
        # in that case would silently drop a real withdrawn/harm signal that safety_search populates
        # afterward. "safety_summary" in store is the presence marker (set by safety_search even
        # when the value is "" for a no-signal drug). EvidenceSummary's own defaults stand otherwise.
        if "safety_summary" in store:
            evidence.safety_summary = store["safety_summary"]
            evidence.regulatory_safety_summary = store["regulatory_safety_summary"]
            evidence.regulatory_safety_full_labels = store[
                "regulatory_safety_full_labels"
            ]
            evidence.pharmacovigilance_summary = store["pharmacovigilance_summary"]
            evidence.literature_safety_summary = store["literature_safety_summary"]
            evidence.label_safety_available = store["label_safety_available"]
            evidence.safety_pmids = store["safety_pmids"]
            evidence.safety_severity = store["safety_severity"]
            evidence.indication_harm = store["indication_harm"]
            evidence.indication_harm_summary = store["indication_harm_summary"]
            evidence.indication_harm_pmids = store["indication_harm_pmids"]
        # logger.warning(
        #     "[TIMING] synthesize %s: %.1fs", disease_name, time.perf_counter() - _t0
        # )
        # logger.warning(
        #     f"literature agent evidence: {evidence}")
        return (
            f"Evidence strength: {evidence.strength}, direction: {evidence.direction}",
            evidence,
        )

    @tool(response_format="content_and_artifact", return_direct=True)
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete. Pass the narrative summary as the argument;
        it is returned as the artifact for downstream assembly into LiteratureOutput.
        """
        return "Analysis complete.", summary

    return [
        build_drug_profile,
        expand_search_terms,
        fetch_and_cache,
        semantic_search,
        safety_search,
        synthesize,
        finalize_analysis,
    ]
