"""Adverse-event literature search for a drug, holdout-clean. Both modes rank by Europe PMC citation count.

  - DRUG-LEVEL (disease=None): the drug's MeSH "adverse effects" subheading as a major topic ([Majr]), falling back
    to a title/abstract query when MeSH indexing is sparse. Answers "is this drug unsafe at all?"
  - DISEASE-SCOPED (disease set): drug AND adverse-event vocabulary AND disease. Answers "is this drug unsafe FOR
    this candidate indication?"

Citation ranking replaces PubMed's relevance sort, which buries landmark safety papers (APPROVe ranks outside the
top 300). Queries use only drug/disease names and generic MeSH tags, so they are safe under date_before.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from indication_scout.constants import (
    AE_DISEASE_AE_LEG,
    AE_DISEASE_DISEASE_LEG,
    AE_DISEASE_DRUG_LEG,
    AE_FALLBACK_MIN_HITS,
    AE_FALLBACK_QUERY_TEMPLATE,
    AE_MAJR_QUERY_TEMPLATE,
    AE_SEARCH_MAX_RESULTS,
    AE_TOP_CITED,
    DEFAULT_CACHE_DIR,
)
from indication_scout.data_sources.europe_pmc import EuropePMCClient
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_pubmed_abstract import PubmedAbstract

logger = logging.getLogger(__name__)


def _disease_scoped_query(
    drug: str, disease: str, disease_aliases: list[str] | None = None
) -> str:
    """Drug leg AND adverse-event vocabulary leg AND disease leg — the indication-specific query."""
    drug_leg = AE_DISEASE_DRUG_LEG.format(drug=drug)
    disease_terms = list(
        dict.fromkeys(
            term.strip().lower()
            for term in [disease, *(disease_aliases or [])]
            if term.strip()
        )
    )
    disease_legs = [
        AE_DISEASE_DISEASE_LEG.format(disease=term) for term in disease_terms
    ]
    disease_leg = (
        disease_legs[0] if len(disease_legs) == 1 else f"({' OR '.join(disease_legs)})"
    )
    return f"({drug_leg} AND {AE_DISEASE_AE_LEG}) AND {disease_leg}"


async def search_adverse_events(
    drug_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    date_before: date | None = None,
    top_cited: int = AE_TOP_CITED,
    disease: str | None = None,
    disease_aliases: list[str] | None = None,
) -> list[PubmedAbstract]:
    """Return top adverse-event abstracts, ranked by Europe PMC citation count.

    Args:
        drug_name: preferred drug name (e.g. "cerivastatin").
        cache_dir: cache directory for the PubMed client.
        date_before: optional holdout cutoff (articles strictly before this date).
        top_cited: how many top-cited abstracts to return.
        disease: when set, runs the disease-scoped query; when None, the drug-level [Majr] query.
        disease_aliases: equivalent disease names to OR into the disease-scoped query.

    Returns:
        Up to `top_cited` PubmedAbstract objects, most-cited first; empty when nothing matches.
    """
    drug = drug_name.strip().lower()

    async with PubMedClient(cache_dir=cache_dir) as pubmed:
        if disease:
            pmids = await pubmed.search(
                _disease_scoped_query(drug, disease, disease_aliases),
                max_results=AE_SEARCH_MAX_RESULTS,
                date_before=date_before,
            )
        else:
            pmids = await pubmed.search(
                AE_MAJR_QUERY_TEMPLATE.format(drug=drug),
                max_results=AE_SEARCH_MAX_RESULTS,
                date_before=date_before,
            )
            # Sparse MeSH indexing (recent / less-studied drugs) → title/abstract fallback.
            if len(pmids) < AE_FALLBACK_MIN_HITS:
                logger.info(
                    "pubmed_ae: [Majr] returned %d hits for %r; using title/abstract fallback",
                    len(pmids),
                    drug,
                )
                pmids = await pubmed.search(
                    AE_FALLBACK_QUERY_TEMPLATE.format(drug=drug),
                    max_results=AE_SEARCH_MAX_RESULTS,
                    date_before=date_before,
                )

        # Holdout post-guard: PubMed's maxdate trusts [pdat], which is unreliable when registration and publish
        # years differ, so re-verify each PMID's sortpubdate and drop post-cutoff ones (as fetch_and_cache does).
        if date_before is not None and pmids:
            pmids = await pubmed._filter_pmids_by_date(pmids, date_before)

        if not pmids:
            return []

        async with EuropePMCClient() as epmc:
            counts = await epmc.fetch_citation_counts(pmids)

        # Rank by citation count desc; PMIDs Europe PMC didn't return sort last (missing == 0).
        ranked_pmids = sorted(pmids, key=lambda p: counts.get(p, 0), reverse=True)[
            :top_cited
        ]

        abstracts = await pubmed.fetch_abstracts(ranked_pmids)

    # Preserve citation rank order (fetch_abstracts may reorder by PMID-batch).
    by_pmid = {a.pmid: a for a in abstracts}
    return [by_pmid[p] for p in ranked_pmids if p in by_pmid]
