"""
PubMed API client.

Three methods:
  1. search         — Find PMIDs matching a query (cached)
  2. fetch_articles — Fetch article content for given PMIDs
  3. get_count      — Quick count of results without fetching
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from indication_scout.config import get_settings
from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    PUBMED_FETCH_URL,
    PUBMED_MAX_CONCURRENT_REQUESTS,
    PUBMED_PUBDATE_TTL,
    PUBMED_SEARCH_SLEEP_SECONDS,
    PUBMED_SEARCH_URL,
    PUBMED_SUMMARY_URL,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.utils.cache import cache_get, cache_set
from indication_scout.models.model_pubmed_abstract import PubmedAbstract


class PubMedClient(BaseClient):
    """Client for querying PubMed/NCBI APIs."""

    # NCBI eutils is a hard dependency for literature analysis. If retries
    # exhaust, exit the program (logged to data_source_failures.log) instead
    # of raising a DataSourceError that downstream code may swallow into
    # silently degraded results.
    exit_on_retry_exhausted = True

    SEARCH_URL = PUBMED_SEARCH_URL
    FETCH_URL = PUBMED_FETCH_URL
    SUMMARY_URL = PUBMED_SUMMARY_URL

    # Process-wide concurrency cap on NCBI eutils requests. Class-level so the
    # bound holds across the multiple PubMedClient instances the supervisor
    # spins up in parallel (each RetrievalService call creates a new client).
    # Created lazily and rebound when the running event loop changes — an
    # asyncio.Semaphore binds to the loop it is created on, so a cached one
    # would raise "bound to a different event loop" on a second asyncio.run().
    _request_semaphore: asyncio.Semaphore | None = None
    _semaphore_loop: asyncio.AbstractEventLoop | None = None

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key: str = get_settings().ncbi_api_key

    @property
    def _source_name(self) -> str:
        return "pubmed"

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if cls._request_semaphore is None or cls._semaphore_loop is not loop:
            cls._request_semaphore = asyncio.Semaphore(PUBMED_MAX_CONCURRENT_REQUESTS)
            cls._semaphore_loop = loop
        return cls._request_semaphore

    def _inject_api_key(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add the NCBI api_key to request params if configured."""
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _rest_get_json_tolerant(self, url: str, params: dict[str, Any]) -> Any:
        """GET JSON, tolerating raw control characters in the response body.

        NCBI eutils occasionally echoes user-supplied query terms back inside
        ``querytranslation`` / ``translationstack`` with stray control bytes,
        which strict ``json.loads`` rejects. Fetch as text and parse with
        ``strict=False`` so those responses don't crash the supervisor.
        """
        text = await self._request("GET", url, params=params, as_text=True)
        return json.loads(text, strict=False)

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        date_before: date | None = None,
    ) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        if max_results is None:
            max_results = get_settings().pubmed_search_default_max_results
        effective_maxdate = (date_before - timedelta(days=1)) if date_before else None
        cache_params: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "date_before": effective_maxdate,
        }
        cached = cache_get("pubmed_search", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if effective_maxdate:
            params["datetype"] = "pdat"
            params["mindate"] = "1900/01/01"
            params["maxdate"] = effective_maxdate.strftime("%Y/%m/%d")

        async with self._get_semaphore():
            await asyncio.sleep(PUBMED_SEARCH_SLEEP_SECONDS)
            data = await self._rest_get_json_tolerant(
                self.SEARCH_URL, self._inject_api_key(params)
            )
        pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            logger.warning("PubMed search returned 0 PMIDs for query: %r", query)

        # NOTE: the post-search esummary date guard used to live here. It
        # was moved into RetrievalService.fetch_and_cache so it can read
        # pub_date from pgvector for known PMIDs and only call esummary
        # for unknowns — a major NCBI traffic reduction on re-runs.
        # Callers that need a standalone date filter can still invoke
        # PubMedClient._filter_pmids_by_date directly (used by some tests).

        cache_set("pubmed_search", cache_params, pmids, self.cache_dir)
        return pmids

    async def get_count(self, query: str, date_before: date | None = None) -> int:
        """Quick count of results without fetching full records."""
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retmode": "json",
        }
        if date_before:
            params["datetype"] = "pdat"
            params["mindate"] = "1900/01/01"
            params["maxdate"] = (date_before - timedelta(days=1)).strftime("%Y/%m/%d")

        async with self._get_semaphore():
            data = await self._rest_get_json_tolerant(
                self.SEARCH_URL, self._inject_api_key(params)
            )

        count_str = data.get("esearchresult", {}).get("count", "0")
        return int(count_str)

    async def _filter_pmids_by_date(
        self, pmids: list[str], date_before: date, batch_size: int | None = None
    ) -> list[str]:
        """Remove PMIDs whose sortpubdate is on or after date_before.

        Uses esummary to fetch publication dates cheaply (no abstract text).
        PubMed's maxdate filter is not strictly respected, so this is a
        post-search guard.
        """
        if batch_size is None:
            batch_size = get_settings().pubmed_esummary_batch_size

        # Per-PMID sortpubdate cache. A real date is immutable; an empty string
        # (abstract-less / unindexed PMID) is persisted too so those PMIDs aren't
        # re-fetched from esummary every run — the residual holdout date cost.
        dates: dict[str, str] = {}
        missing: list[str] = []
        for pmid in pmids:
            cached = cache_get("pubmed_pubdate", {"pmid": pmid}, self.cache_dir)
            if cached is not None:
                dates[pmid] = cached
            else:
                missing.append(pmid)

        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "json",
            }
            async with self._get_semaphore():
                data = await self._rest_get_json_tolerant(
                    self.SUMMARY_URL, self._inject_api_key(params)
                )
            result = data.get("result", {})
            for pmid in batch:
                sortpubdate: str = result.get(pmid, {}).get("sortpubdate", "")
                dates[pmid] = sortpubdate
                cache_set(
                    "pubmed_pubdate",
                    {"pmid": pmid},
                    sortpubdate,
                    self.cache_dir,
                    ttl=PUBMED_PUBDATE_TTL,
                )

        kept: list[str] = []
        for pmid in pmids:
            sortpubdate = dates.get(pmid, "")
            if not sortpubdate:
                kept.append(pmid)
                continue
            try:
                pub_date = date.fromisoformat(sortpubdate[:10].replace("/", "-"))
            except ValueError:
                kept.append(pmid)
                continue
            if pub_date < date_before:
                kept.append(pmid)
        return kept

    async def fetch_pubtypes(
        self, pmids: list[str], batch_size: int | None = None
    ) -> dict[str, list[str]]:
        """Fetch publication types from esummary for the given PMIDs.

        Returns a dict mapping pmid → list of pubtype strings (e.g.
        ["Journal Article", "Randomized Controlled Trial"]). PMIDs with
        no esummary record or no pubtype field are absent from the result.

        Per-PMID file cache so re-runs across different candidate sets
        reuse prior fetches. Cache misses are batched into single
        esummary requests (pubmed_esummary_batch_size, default 200).
        """
        if not pmids:
            return {}

        if batch_size is None:
            batch_size = get_settings().pubmed_esummary_batch_size

        result: dict[str, list[str]] = {}
        missing: list[str] = []
        for pmid in pmids:
            cached = cache_get("pubmed_pubtypes", {"pmid": pmid}, self.cache_dir)
            if cached is not None:
                if cached:
                    result[pmid] = cached
            else:
                missing.append(pmid)

        if not missing:
            return result

        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "json",
            }
            async with self._get_semaphore():
                data = await self._rest_get_json_tolerant(
                    self.SUMMARY_URL, self._inject_api_key(params)
                )
            summaries = data.get("result", {})
            for pmid in batch:
                summary = summaries.get(pmid, {})
                pubtypes = summary.get("pubtype", []) or []
                # Persist even empty lists so we don't refetch records
                # that genuinely have no pubtype field.
                cache_set("pubmed_pubtypes", {"pmid": pmid}, pubtypes, self.cache_dir)
                if pubtypes:
                    result[pmid] = pubtypes

        return result

    async def fetch_abstracts(
        self, pmids: list[str], batch_size: int | None = None
    ) -> list[PubmedAbstract]:
        """Fetch article content for given PMIDs."""
        if not pmids:
            return []

        if batch_size is None:
            batch_size = get_settings().pubmed_efetch_batch_size
        all_articles: list[PubmedAbstract] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }

            async with self._get_semaphore():
                xml_text = await self._rest_get_xml(
                    self.FETCH_URL, self._inject_api_key(params)
                )

            articles = self._parse_pubmed_xml(xml_text)
            all_articles.extend(articles)

        return all_articles

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubmedAbstract]:
        """Parse PubMed XML response into PubmedAbstract objects."""
        articles = []

        # Strip control characters that are invalid in XML 1.0 (U+0000–U+001F,
        # excluding the three characters XML permits: tab, newline, carriage-return).
        xml_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", xml_text)

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise DataSourceError(self._source_name, f"Failed to parse XML: {e}")

        for article_elem in root.findall(".//PubmedArticle"):
            pmid = self._xml_text(article_elem, ".//PMID")
            if not pmid:
                continue

            title = self._xml_text(article_elem, ".//ArticleTitle")

            # Abstract - may have multiple sections
            abstract_parts = []
            for abs_elem in article_elem.findall(".//AbstractText"):
                label = abs_elem.get("Label", "")
                text = "".join(abs_elem.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last_name = self._xml_text(author, "LastName")
                fore_name = self._xml_text(author, "ForeName")
                if last_name:
                    name = f"{last_name}, {fore_name}" if fore_name else last_name
                    authors.append(name)

            journal = self._xml_text(article_elem, ".//Journal/Title")

            # Publication date
            pub_date = None
            pub_date_elem = article_elem.find(".//PubDate")
            if pub_date_elem is not None:
                year = self._xml_text(pub_date_elem, "Year")
                month = self._xml_text(pub_date_elem, "Month")
                day = self._xml_text(pub_date_elem, "Day")
                if year:
                    pub_date = year
                    if month:
                        pub_date += f"-{month}"
                        if day:
                            pub_date += f"-{day}"

            mesh_terms = [
                self._xml_text(mesh, "DescriptorName")
                for mesh in article_elem.findall(".//MeshHeading")
                if self._xml_text(mesh, "DescriptorName")
            ]

            keywords = [kw.text for kw in article_elem.findall(".//Keyword") if kw.text]

            articles.append(
                PubmedAbstract(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    pub_date=pub_date,
                    mesh_terms=mesh_terms,
                    keywords=keywords,
                )
            )

        for book_elem in root.findall(".//PubmedBookArticle"):
            doc = book_elem.find("BookDocument")
            if doc is None:
                continue

            pmid = self._xml_text(doc, ".//PMID")
            if not pmid:
                continue

            title = self._xml_text(doc, ".//ArticleTitle")

            # Abstract - may have multiple labelled sections
            abstract_parts = []
            for abs_elem in doc.findall(".//AbstractText"):
                label = abs_elem.get("Label", "")
                text = "".join(abs_elem.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

            # Authors only — exclude editors (AuthorList Type="editors")
            authors = []
            for author_list in doc.findall("AuthorList"):
                if author_list.get("Type") == "editors":
                    continue
                for author in author_list.findall("Author"):
                    last_name = self._xml_text(author, "LastName")
                    fore_name = self._xml_text(author, "ForeName")
                    if last_name:
                        name = f"{last_name}, {fore_name}" if fore_name else last_name
                        authors.append(name)

            journal = self._xml_text(doc, ".//Book/BookTitle")

            # Publication date
            pub_date = None
            pub_date_elem = doc.find(".//PubDate")
            if pub_date_elem is not None:
                year = self._xml_text(pub_date_elem, "Year")
                month = self._xml_text(pub_date_elem, "Month")
                day = self._xml_text(pub_date_elem, "Day")
                if year:
                    pub_date = year
                    if month:
                        pub_date += f"-{month}"
                        if day:
                            pub_date += f"-{day}"

            keywords = [kw.text for kw in doc.findall(".//Keyword") if kw.text]

            articles.append(
                PubmedAbstract(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    pub_date=pub_date,
                    mesh_terms=[],
                    keywords=keywords,
                )
            )

        return articles

    @staticmethod
    def _xml_text(elem: ET.Element, path: str) -> str | None:
        """Safely extract text from an XML element."""
        found = elem.find(path)
        return found.text if found is not None and found.text else None
