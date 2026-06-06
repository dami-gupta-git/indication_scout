"""On-demand drill-down routes (PLAN_react.md §2.4b, T2.6).

Thin wrappers over existing data-source clients, fetched when the user clicks into something.
INTEGRITY: these only enrich entities the analysis already surfaced — they introduce no new
candidate diseases/targets/edges.
"""

import logging

from fastapi import APIRouter, HTTPException

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.models.model_open_targets import TargetData
from indication_scout.models.model_pubmed_abstract import PubmedAbstract

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["drilldown"])


@router.get("/trials/{nct_id}")
async def get_trial(nct_id: str) -> Trial:
    """Full trial detail for a clinical-trials row drill-down."""
    try:
        async with ClinicalTrialsClient() as client:
            return await client.get_trial(nct_id)
    except DataSourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/pubmed/{pmid}")
async def get_pubmed_abstract(pmid: str) -> PubmedAbstract:
    """Abstract preview for a PMID hover/expand."""
    try:
        async with PubMedClient() as client:
            abstracts = await client.fetch_abstracts([pmid])
    except DataSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not abstracts:
        raise HTTPException(status_code=404, detail=f"No abstract for PMID '{pmid}'")
    return abstracts[0]


@router.get("/targets/{target_id}")
async def get_target(target_id: str) -> TargetData:
    """Target detail (function, tractability) for a network-graph node. Expects an Ensembl ID."""
    try:
        async with OpenTargetsClient() as client:
            return await client.get_target_data(target_id)
    except DataSourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
