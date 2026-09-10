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
    EUROPE_PMC_ANNOTATIONS_BATCH,
    EUROPE_PMC_ANNOTATIONS_URL,
    EUROPE_PMC_CITATION_BATCH,
    EUROPE_PMC_CITATION_NS,
    EUROPE_PMC_CURSOR_PARAM,
    EUROPE_PMC_CURSOR_START,
    EUROPE_PMC_DRUG_QUERY,
    EUROPE_PMC_DRUG_TITLE_ABS_QUERY,
    EUROPE_PMC_PAGE_SIZE,
    EUROPE_PMC_RESULT_TYPE,
    EUROPE_PMC_SEARCH_NS,
    EUROPE_PMC_SEARCH_PAGE_SIZE,
    EUROPE_PMC_SEARCH_URL,
    EUROPE_PMC_YEAR_CLAUSE,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_europe_pmc import (
    ArticleAnnotations,
    EuropePMCArticle,
    TextMinedAnnotation,
)
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Europe PMC SciLite annotation types this client fetches in one request. "Pathway" is a legacy
# annotation type Europe PMC's current text-mining pipeline no longer produces — omitted.
_ANNOTATION_TYPES = "Diseases,Chemicals,Gene_Proteins"


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

    async def search_filtered_by_drug(
        self,
        drug: str,
        pub_types: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sort: str | None = None,
        page_size: int = EUROPE_PMC_SEARCH_PAGE_SIZE,
    ) -> list[EuropePMCArticle]:
        """Return articles where ``drug`` appears in the title or abstract.

        ``pub_types`` are OR-ed together and AND-ed onto the query (Europe PMC ``PUB_TYPE`` values,
        e.g. "Case Reports", "Randomized Controlled Trial", "Review"). ``year_from``/``year_to``
        bound ``PUB_YEAR``; either end may be given alone, and the open end becomes a wildcard.
        ``sort`` takes ``EUROPE_PMC_SORT_CITED`` or ``EUROPE_PMC_SORT_RECENT``; leaving it None
        gives Europe PMC's relevance order (the API rejects a literal "RELEVANCE" value).

        Returns at most ``page_size`` articles — this is a single page, not a full crawl of the
        result set. Unlike the citation/annotation lookups, a failure here is raised, not
        swallowed: the caller asked for this pool and an empty list would be indistinguishable
        from a drug with no literature.
        """
        if not drug:
            return []

        query = EUROPE_PMC_DRUG_TITLE_ABS_QUERY.format(drug=drug)

        if pub_types:
            clause = " OR ".join(f'PUB_TYPE:"{p}"' for p in pub_types)
            query += f" AND ({clause})"

        if year_from is not None or year_to is not None:
            lo = year_from if year_from is not None else "*"
            hi = year_to if year_to is not None else "*"
            query += f" AND PUB_YEAR:[{lo} TO {hi}]"

        params: dict[str, str | int] = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
        }
        if sort:
            params["sort"] = sort

        data = await self._rest_get(self.SEARCH_URL, params)

        results = (data.get("resultList", {}) or {}).get("result", []) or []
        articles = [
            EuropePMCArticle.from_search_result(result)
            for result in results
            if result.get("pmid")
        ]
        logger.info(
            "europepmc: drug=%s hits=%s returned=%s query=%s",
            drug,
            data.get("hitCount"),
            len(articles),
            query,
        )
        return articles

    async def fetch_disease_annotations(
        self, pmids: list[str], batch_size: int = EUROPE_PMC_ANNOTATIONS_BATCH
    ) -> dict[str, ArticleAnnotations]:
        """Return {pmid: ArticleAnnotations} of pre-tagged disease, chemical, and gene/protein
        mentions per article.

        Uses Europe PMC's Annotations API (SciLite text-mining), not an LLM — every entity name
        returned was tagged by Europe PMC's own pipeline against a specific article section, and
        linked to its ontology (UMLS for diseases, CHEBI for chemicals, UniProt for gene/proteins).
        Only articles Europe PMC has text-mined are covered; a PMID with no mined terms of any of
        these types (or not indexed at all) is simply absent from the result. A failed batch is
        logged and skipped, never raised — this is a best-effort literature signal, not a hard
        dependency.
        """
        if not pmids:
            return {}

        results: dict[str, ArticleAnnotations] = {}
        for i in range(0, len(pmids), batch_size):
            chunk = pmids[i : i + batch_size]
            params = {
                "articleIds": ",".join(f"MED:{p}" for p in chunk),
                "type": _ANNOTATION_TYPES,
                "format": "JSON",
            }
            try:
                data = await self._rest_get(EUROPE_PMC_ANNOTATIONS_URL, params)
            except DataSourceError as e:
                logger.warning(
                    "europepmc: annotations batch failed (%s); those PMIDs are omitted",
                    e,
                )
                continue
            for article in data or []:
                pmid = article.get("extId")
                if not pmid:
                    continue
                by_type: dict[str, list[TextMinedAnnotation]] = {
                    "Diseases": [],
                    "Chemicals": [],
                    "Gene_Proteins": [],
                }
                for ann in article.get("annotations", []) or []:
                    ann_type = ann.get("type")
                    if ann_type not in by_type:
                        continue
                    by_type[ann_type].append(
                        TextMinedAnnotation(
                            exact=ann.get("exact") or "",
                            concept_name=(ann.get("tags") or [{}])[0].get("name") or "",
                            concept_uri=(ann.get("tags") or [{}])[0].get("uri") or "",
                            section=ann.get("section") or "",
                        )
                    )
                results[pmid] = ArticleAnnotations(
                    pmid=pmid,
                    diseases=by_type["Diseases"],
                    chemicals=by_type["Chemicals"],
                    gene_proteins=by_type["Gene_Proteins"],
                )
        return results
