"""Europe PMC API client — citation counts for PubMed articles.

Used to rank adverse-event literature by citation count rather than PubMed's term-frequency
relevance sort, which buries landmark safety papers (see AE_TOP_CITED / pubmed_ae.py).
"""

from __future__ import annotations

import logging

from indication_scout.constants import (
    EUROPE_PMC_CITATION_BATCH,
    EUROPE_PMC_SEARCH_URL,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError

logger = logging.getLogger(__name__)


class EuropePMCClient(BaseClient):
    """Client for Europe PMC's REST search API."""

    SEARCH_URL = EUROPE_PMC_SEARCH_URL

    @property
    def _source_name(self) -> str:
        return "europepmc"

    async def fetch_citation_counts(
        self, pmids: list[str], batch_size: int = EUROPE_PMC_CITATION_BATCH
    ) -> dict[str, int]:
        """Return {pmid: citedByCount} for the given PMIDs.

        Europe PMC's ``citedByCount`` draws on a broader citation graph than NCBI esummary's
        pmcrefcount (which is blank for exactly the high-impact papers we want ranked highly).
        A PMID absent from the result maps to nothing (caller treats missing as 0); ranking is a
        best-effort enhancement, so a failed batch is logged and skipped, never raised.
        """
        if not pmids:
            return {}

        counts: dict[str, int] = {}
        for i in range(0, len(pmids), batch_size):
            chunk = pmids[i : i + batch_size]
            query = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED"
            params = {"query": query, "format": "json", "pageSize": batch_size}
            try:
                data = await self._rest_get(self.SEARCH_URL, params)
            except DataSourceError as e:
                logger.warning(
                    "europepmc: citation batch failed (%s); those PMIDs fall back to 0", e
                )
                continue
            for rec in (data.get("resultList", {}) or {}).get("result", []) or []:
                pmid = rec.get("pmid")
                if pmid:
                    counts[pmid] = int(rec.get("citedByCount", 0) or 0)
        return counts
