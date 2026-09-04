"""Europe PMC API client.

Two uses: citation counts for ranking adverse-event literature (PubMed's term-frequency relevance
sort buries landmark safety papers — see AE_TOP_CITED / pubmed_ae.py), and drug-scoped literature
retrieval for candidate-indication sourcing (see design_europe_pmc.md).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    EUROPE_PMC_CITATION_BATCH,
    EUROPE_PMC_CITATION_NS,
    EUROPE_PMC_CURSOR_PARAM,
    EUROPE_PMC_CURSOR_START,
    EUROPE_PMC_DRUG_QUERY,
    EUROPE_PMC_PAGE_SIZE,
    EUROPE_PMC_RESULT_TYPE,
    EUROPE_PMC_SEARCH_NS,
    EUROPE_PMC_SEARCH_URL,
    EUROPE_PMC_YEAR_CLAUSE,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_europe_pmc import EuropePMCArticle
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


class EuropePMCClient(BaseClient):
    """Client for Europe PMC's REST search API."""

    SEARCH_URL = EUROPE_PMC_SEARCH_URL

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    @property
    def _source_name(self) -> str:
        return "europepmc"

    def _build_drug_query(self, drug_name: str, date_before: date | None) -> str:
        """Compose the drug-scoped search query, with an optional publication-date upper bound.

        ``date_before`` is exclusive, matching `PubMedClient.search`: a holdout cut at an approval
        date must not see papers published on that date. Europe PMC's FIRST_PDATE range is
        inclusive at both ends, so the bound sent is the preceding day.
        """
        year_clause = ""
        if date_before:
            last_visible_day = date_before - timedelta(days=1)
            year_clause = EUROPE_PMC_YEAR_CLAUSE.format(
                date_before=last_visible_day.strftime("%Y-%m-%d")
            )
        return EUROPE_PMC_DRUG_QUERY.format(drug=drug_name, year_clause=year_clause)

    async def search_by_drug(
        self, drug_name: str, date_before: date | None = None
    ) -> list[EuropePMCArticle]:
        """Return every Europe PMC article mentioning the drug in its title or abstract.

        Paginated by cursor until exhausted — no result cap, no relevance filtering. A failure at
        any page raises rather than returning what was collected so far: a partial pool understates
        the literature and is indistinguishable downstream from a drug with little written about it.
        """
        query = self._build_drug_query(drug_name, date_before)
        cache_params = {"query": query}
        cached = cache_get(EUROPE_PMC_SEARCH_NS, cache_params, self.cache_dir)
        if cached is not None:
            return [EuropePMCArticle(**rec) for rec in cached]

        articles: list[EuropePMCArticle] = []
        cursor = EUROPE_PMC_CURSOR_START
        seen_cursors: set[str] = set()

        while True:
            params: dict[str, Any] = {
                "query": query,
                "format": "json",
                "resultType": EUROPE_PMC_RESULT_TYPE,
                "pageSize": EUROPE_PMC_PAGE_SIZE,
                EUROPE_PMC_CURSOR_PARAM: cursor,
            }
            data = await self._rest_get(self.SEARCH_URL, params)

            page = (data.get("resultList") or {}).get("result") or []
            for raw in page:
                try:
                    articles.append(EuropePMCArticle.from_search_result(raw))
                except (KeyError, TypeError, ValueError) as e:
                    raise DataSourceError(
                        self._source_name,
                        f"Unparseable search record {raw.get('source')}:{raw.get('id')}: {e}",
                    ) from e

            next_cursor = data.get("nextCursorMark")
            # Europe PMC repeats the cursor on the final page rather than omitting it; without this
            # check the loop would re-request the last page forever.
            if not next_cursor or next_cursor == cursor or not page:
                break
            if next_cursor in seen_cursors:
                raise DataSourceError(
                    self._source_name, f"Cursor loop detected at {next_cursor}"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        logger.info(
            "europepmc: retrieved %d articles for %r%s",
            len(articles),
            drug_name,
            f" (before {date_before})" if date_before else "",
        )
        cache_set(
            EUROPE_PMC_SEARCH_NS,
            cache_params,
            [a.model_dump() for a in articles],
            self.cache_dir,
        )
        return articles

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

        # Per-PMID cache. A PMID Europe PMC has no record for is cached as {"count": None} rather
        # than left absent, so those PMIDs aren't re-queried every run — otherwise a single
        # unindexed PMID keeps the whole batch request alive and re-pays its timeout cost.
        counts: dict[str, int] = {}
        missing: list[str] = []
        for pmid in pmids:
            cached = cache_get(EUROPE_PMC_CITATION_NS, {"pmid": pmid}, self.cache_dir)
            if cached is None:
                missing.append(pmid)
            elif cached["count"] is not None:
                counts[pmid] = cached["count"]

        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            query = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED"
            params = {"query": query, "format": "json", "pageSize": batch_size}
            try:
                data = await self._rest_get(self.SEARCH_URL, params)
            except DataSourceError as e:
                logger.warning(
                    "europepmc: citation batch failed (%s); those PMIDs fall back to 0",
                    e,
                )
                continue
            fetched: dict[str, int] = {}
            for rec in (data.get("resultList", {}) or {}).get("result", []) or []:
                pmid = rec.get("pmid")
                if pmid:
                    fetched[pmid] = int(rec.get("citedByCount", 0) or 0)
            counts.update(fetched)
            for pmid in chunk:
                cache_set(
                    EUROPE_PMC_CITATION_NS,
                    {"pmid": pmid},
                    {"count": fetched.get(pmid)},
                    self.cache_dir,
                )
        return counts
