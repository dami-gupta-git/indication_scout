"""
PubMed API client.

Three methods:
  1. search         — Find PMIDs matching a query (cached)
  2. fetch_articles — Fetch article content for given PMIDs
  3. get_count      — Quick count of results without fetching
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

from indication_scout.constants import DEFAULT_CACHE_DIR, PUBMED_FETCH_URL, PUBMED_SEARCH_URL
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.utils.cache import cache_get, cache_set
from indication_scout.models.model_pubmed_abstract import PubmedAbstract


class PubMedClient(BaseClient):
    """Client for querying PubMed/NCBI APIs."""

    SEARCH_URL = PUBMED_SEARCH_URL
    FETCH_URL = PUBMED_FETCH_URL

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _source_name(self) -> str:
        return "pubmed"

    async def search(
        self, query: str, max_results: int = 50, date_before: date | None = None
    ) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        cache_params: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "date_before": date_before,
        }
        cached = cache_get("pubmed_search", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        if date_before:
            params["datetype_maxdate"] = date_before.strftime("%Y-%m-%d")

        data = await self._rest_get(self.SEARCH_URL, params)
        pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])

        cache_set("pubmed_search", cache_params, pmids, self.cache_dir)
        return pmids

    async def get_count(self, query: str, date_before: date | None = None) -> int:
        """Quick count of results without fetching full records."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retmode": "json",
        }
        if date_before:
            params["datetype_maxdate"] = date_before.strftime("%Y-%m-%d")

        data = await self._rest_get(self.SEARCH_URL, params)

        count_str = data.get("esearchresult", {}).get("count", "0")
        return int(count_str)

    async def fetch_abstracts(
        self, pmids: list[str], batch_size: int = 100
    ) -> list[PubmedAbstract]:
        """Fetch article content for given PMIDs."""
        if not pmids:
            return []

        all_articles: list[PubmedAbstract] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }

            xml_text = await self._rest_get_xml(self.FETCH_URL, params)

            articles = self._parse_pubmed_xml(xml_text)
            all_articles.extend(articles)

        return all_articles

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubmedAbstract]:
        """Parse PubMed XML response into PubmedAbstract objects."""
        articles = []

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
