"""Adverse-event literature search for a drug — holdout-clean.

Two query modes, both citation-ranked:
  - DRUG-LEVEL (disease=None): the drug's MeSH "adverse effects" SUBHEADING as a MAJOR topic
    ([Majr]) — PubMed's curated "this paper is primarily about this drug's harms" tag. Precise (no
    500-cap saturation), catches specific-named signals (agranulocytosis, bladder cancer) without
    knowing the event name. Falls back to a title/abstract query when MeSH indexing is sparse.
    Answers "is this drug unsafe at all?"
  - DISEASE-SCOPED (disease set): drug leg AND an adverse-event vocabulary leg AND a disease leg —
    recovers indication-specific safety papers the drug-level pool misses (e.g. rofecoxib×colorectal
    → APPROVe). Answers "is this drug unsafe FOR this candidate indication?"

Both rank by Europe PMC citation count, NOT PubMed's term-frequency relevance sort — the latter
buries landmark safety papers (APPROVe ranks outside PubMed's top 300 for a toxicity query despite
1,600+ citations). Both use only the drug/disease names + generic MeSH tags (no future knowledge),
so they are safe in a temporal-holdout (date_before) context.
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


def _disease_scoped_query(drug: str, disease: str) -> str:
    """Drug leg AND adverse-event vocabulary leg AND disease leg — the indication-specific query."""
    drug_leg = AE_DISEASE_DRUG_LEG.format(drug=drug)
    disease_leg = AE_DISEASE_DISEASE_LEG.format(disease=disease.strip().lower())
    return f"({drug_leg} AND {AE_DISEASE_AE_LEG}) AND {disease_leg}"


async def search_adverse_events(
    drug_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    date_before: date | None = None,
    top_cited: int = AE_TOP_CITED,
    disease: str | None = None,
) -> list[PubmedAbstract]:
    """Return top adverse-event abstracts, ranked by Europe PMC citation count.

    Args:
        drug_name: preferred drug name (e.g. "cerivastatin").
        cache_dir: cache directory for the PubMed client.
        date_before: optional temporal-holdout cutoff (articles strictly before this date).
        top_cited: how many top-cited abstracts to return.
        disease: when set, runs the DISEASE-SCOPED query (indication-specific safety); when None,
            the DRUG-LEVEL [Majr] query (drug-wide safety).

    Returns:
        Up to `top_cited` PubmedAbstract objects, most-cited first. Empty when there is no matching
        adverse-event literature (no fabrication).
    """
    drug = drug_name.strip().lower()

    async with PubMedClient(cache_dir=cache_dir) as pubmed:
        if disease:
            pmids = await pubmed.search(
                _disease_scoped_query(drug, disease),
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

        # Holdout post-guard: PubMed's maxdate filter trusts the [pdat] field, which is UNRELIABLE
        # for records whose original registration year differs from the publish year (e.g. a
        # Cochrane review with pdat=1994 but sortpubdate=2004). esearch lets those leak past the
        # cutoff; re-verify each PMID's sortpubdate and drop post-cutoff ones. Same guard the
        # efficacy path (fetch_and_cache) already applies.
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
