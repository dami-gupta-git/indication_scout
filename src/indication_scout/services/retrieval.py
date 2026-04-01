"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import asyncio
import json
import logging
from pathlib import Path
from typing import TypedDict

import wandb

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from indication_scout.constants import CACHE_TTL, PUBMED_MAX_RESULTS
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.open_targets import (
    CompetitorRawData,
    OpenTargetsClient,
)
from indication_scout.services.disease_helper import (
    llm_normalize_disease_batch,
    merge_duplicate_diseases,
)
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.sqlalchemy.pubmed_abstracts import PubmedAbstracts
from indication_scout.services.embeddings import embed_async
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.llm import parse_llm_response, query_llm, query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class AbstractResult(TypedDict):
    pmid: str
    title: str
    abstract: str
    similarity: float


class RetrievalService:
    """Stateful retrieval service bound to a specific cache directory.

    Instantiate with the appropriate cache_dir for the calling context
    (e.g. DEFAULT_CACHE_DIR for production, TEST_CACHE_DIR for tests).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    async def _normalize_disease_groups(
        self, diseases: dict[str, set[str]]
    ) -> dict[str, set[str]]:
        """Normalize disease names via LLM and merge groups that collapse to the same key.

        Args:
            diseases: Dict mapping disease name to set of competitor drug names.

        Returns:
            Dict with normalized disease names as keys and unioned drug sets.
        """
        original_names = list(diseases.keys())
        norm_map = await llm_normalize_disease_batch(original_names)

        merged: dict[str, set[str]] = {}
        for original in original_names:
            normalized = norm_map[original]
            key = normalized.split(" OR ")[0].strip().lower()
            if key in merged:
                merged[key] |= diseases[original]
            else:
                merged[key] = set(diseases[original])
        return merged

    async def get_drug_competitors(self, drug_name: str) -> dict[str, set[str]]:
        """Fetch top disease indications and their competitor drugs from Open Targets.

        Fetches raw competitor data from the client, then uses an LLM to merge
        duplicate disease names and remove overly broad terms before returning.

        Args:
            drug_name: Common drug name (e.g. "empagliflozin").

        Returns:
            Dict mapping disease name to set of competitor drug names.
        """
        cache_params = {"drug_name": drug_name}
        cached = cache_get("drug_competitors", cache_params, self.cache_dir)
        if cached is not None and len(cached) > 0:
            return {disease: set(drugs) for disease, drugs in cached.items()}

        async with OpenTargetsClient(cache_dir=self.cache_dir) as client:
            raw: CompetitorRawData = await client.get_drug_competitors(drug_name)

        top_40 = await self._normalize_disease_groups(raw["diseases"])
        drug_indications = raw["drug_indications"]
        disease_names = list(top_40.keys())
        merge_result = await merge_duplicate_diseases(disease_names, drug_indications)

        for disease in merge_result["remove"]:
            if disease.lower() in top_40:
                del top_40[disease.lower()]

        removed = {n.lower() for n in merge_result["remove"]}
        for canonical, aliases in merge_result["merge"].items():
            canonical_lower = canonical.lower()
            aliases_lower = [a.lower() for a in aliases]
            all_names = [canonical_lower] + aliases_lower

            if canonical_lower in removed:
                surviving = [n for n in aliases_lower if n not in removed]
                if not surviving:
                    continue
                canonical_lower = surviving[0]

            combined: set[str] = set()
            for disease in all_names:
                if disease in removed:
                    continue
                if disease in top_40:
                    combined |= top_40[disease]
                    if disease != canonical_lower:
                        del top_40[disease]
            if combined:
                top_40[canonical_lower] = combined

        sorted_data = dict(
            sorted(top_40.items(), key=lambda item: len(item[1]), reverse=True)
        )
        top_15 = dict(list(sorted_data.items())[:10])

        cache_set(
            "drug_competitors",
            cache_params,
            {disease: list(drugs) for disease, drugs in top_15.items()},
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return top_15

    async def build_drug_profile(self, drug_name: str) -> DrugProfile:
        """Fetch drug + target data from Open Targets, enrich with ATC descriptions from ChEMBL,
        and return a DrugProfile ready for use in search term expansion.

        Args:
            drug_name: Common drug name (e.g. "metformin").

        Returns:
            DrugProfile with all fields populated. atc_descriptions will be [] if the drug
            has no ATC classifications.
        """
        async with OpenTargetsClient(cache_dir=self.cache_dir) as open_targets_client:
            rich = await open_targets_client.get_rich_drug_data(drug_name)

        atc_descriptions = []
        if rich.drug.atc_classifications:
            async with ChEMBLClient() as chembl_client:
                for code in rich.drug.atc_classifications:
                    atc_descriptions.append(
                        await chembl_client.get_atc_description(code)
                    )

        return DrugProfile.from_rich_drug_data(rich, atc_descriptions)

    def get_stored_pmids(self, pmids: list[str], db: Session) -> set[str]:
        """Return the subset of the given PMIDs that already exist in pubmed_abstracts.

        Used by fetch_and_cache to avoid re-fetching and re-embedding abstracts that
        are already in pgvector. A single bulk SELECT is used rather than one query
        per PMID to keep DB round-trips to a minimum.

        Args:
            pmids: Candidate PMIDs to check.
            db: Active SQLAlchemy session.

        Returns:
            Set of PMIDs that are present in the pubmed_abstracts table.
        """
        if not pmids:
            return set()

        rows = db.execute(
            text("SELECT pmid FROM pubmed_abstracts WHERE pmid = ANY(:pmids)"),
            {"pmids": pmids},
        ).fetchall()

        return {row[0] for row in rows}

    async def fetch_new_abstracts(
        self, all_pmids: list[str], stored_pmids: set[str], client: PubMedClient
    ) -> list[PubmedAbstract]:
        """Fetch PubMed abstracts for PMIDs not already in the database.

        Computes the set difference between all_pmids and stored_pmids, then
        calls client.fetch_abstracts on only the new ones. If there are
        no new PMIDs the network call is skipped entirely.

        Args:
            all_pmids: Full list of PMIDs from a PubMed search result.
            stored_pmids: PMIDs already present in pubmed_abstracts (from get_stored_pmids).
            client: Open PubMedClient session to reuse (owned by the caller).

        Returns:
            List of PubmedAbstract objects for each newly fetched PMID.
            Empty list if all PMIDs were already stored.
        """
        new_pmids = [p for p in all_pmids if p not in stored_pmids]
        if not new_pmids:
            logger.debug("All %d PMIDs already stored; skipping fetch", len(all_pmids))
            return []

        logger.debug("Fetching %d new abstracts from PubMed", len(new_pmids))
        return await client.fetch_abstracts(new_pmids)

    async def embed_abstracts(
        self,
        abstracts: list[PubmedAbstract],
    ) -> list[tuple[PubmedAbstract, list[float]]]:
        """Embed a list of PubMed abstracts using BioLORD-2023.

        Builds embed text as "<title>. <abstract>" for each abstract and calls
        embed_async() in a single batch. Returns (abstract, vector) pairs aligned by index.

        Args:
            abstracts: Abstracts to embed.

        Returns:
            List of (PubmedAbstract, embedding vector) pairs in the same order as input.
            Empty list if abstracts is empty (embed() is not called).
        """
        if not abstracts:
            return []

        texts = [f"{a.title}. {a.abstract or ''}" for a in abstracts]
        vectors = await embed_async(texts)
        return list(zip(abstracts, vectors))

    def insert_abstracts(
        self,
        pairs: list[tuple[PubmedAbstract, list[float]]],
        db: Session,
    ) -> None:
        """Bulk-insert (abstract, embedding) pairs into pubmed_abstracts.

        Uses INSERT ... ON CONFLICT DO NOTHING so re-running with already-stored
        PMIDs is safe and idempotent. Does nothing when pairs is empty.

        Args:
            pairs: Output of embed_abstracts — (PubmedAbstract, vector) tuples.
            db: Active SQLAlchemy session.
        """
        if not pairs:
            return

        rows = [
            {
                "pmid": abstract.pmid,
                "title": abstract.title,
                "abstract": abstract.abstract,
                "authors": abstract.authors or [],
                "journal": abstract.journal,
                "pub_date": abstract.pub_date,
                "mesh_terms": abstract.mesh_terms or [],
                "embedding": vector,
            }
            for abstract, vector in pairs
        ]

        stmt = (
            insert(PubmedAbstracts)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["pmid"])
        )
        db.execute(stmt)
        db.commit()
        logger.debug("Inserted %d abstracts into pubmed_abstracts", len(rows))

    async def fetch_and_cache(self, queries: list[str], db: Session) -> list[str]:
        """Hit PubMed for all queries concurrently, fetch new abstracts, embed in one batch, cache in pgvector.

        Steps:
          1. Search PubMed concurrently for all queries → deduplicated PMIDs
          2. Single bulk check against pgvector for already-stored PMIDs
          3. Single fetch for all new abstracts
          4. Single embed call with BioLORD-2023
          5. Single bulk INSERT into pgvector (ON CONFLICT DO NOTHING)

        Returns the deduplicated union of all PMIDs across all queries.

        Args:
            queries: PubMed keyword queries (e.g. from expand_search_terms).
            db: Active SQLAlchemy session.

        Returns:
            Deduplicated list of all PMIDs returned by PubMed search across all queries.
            Note: not every returned PMID has a row in pubmed_abstracts — articles without
            an abstract (letters, editorials) are excluded from the vector store. Callers
            that pass this list to semantic_search will see those PMIDs silently skipped
            by the WHERE pmid = ANY(:pmids) clause, which is intentional and correct.
        """
        async with PubMedClient(cache_dir=self.cache_dir) as client:
            # 1. Search all queries concurrently
            search_results = await asyncio.gather(
                *[
                    client.search(query, max_results=PUBMED_MAX_RESULTS)
                    for query in queries
                ]
            )

            # Flatten and deduplicate while preserving first-seen order
            all_pmids: list[str] = list(
                dict.fromkeys(pmid for pmids in search_results for pmid in pmids)
            )

            # 2. Single bulk check against pgvector
            stored = self.get_stored_pmids(all_pmids, db)

            # 3. Single fetch for all new abstracts
            new_abstracts = await self.fetch_new_abstracts(all_pmids, stored, client)

        # Articles with no abstract (letters, editorials) are excluded —
        # they have no text to embed meaningfully.
        abstracts_with_text = [a for a in new_abstracts if a.abstract]

        # 4. Single embed call for the entire batch
        pairs = await self.embed_abstracts(abstracts_with_text)

        # 5. Single bulk insert
        self.insert_abstracts(pairs, db)

        return all_pmids

    async def semantic_search(
        self, disease: str, drug: str, pmids: list[str], db: Session, top_k: int = 5
    ) -> list[AbstractResult]:
        """For a given drug, disease, and list of PMIDs, return top-k most similar abstracts from pgvector

        Constructs a natural-language query from drug and disease (e.g. "Evidence for metformin
        as a treatment for colorectal cancer..."), embeds it with BioLORD-2023, then runs a
        cosine similarity search restricted to the given PMIDs.

        Args:
            disease: e.g. "colorectal cancer"
            drug: e.g. "metformin"
            pmids: e.g. ["29734553", "31245678", "30198432"]
            top_k: Maximum number of results to return (default 5).

        Returns:
            List of dicts ranked by descending similarity, e.g.:
            [{"pmid": "29734553", "title": "Metformin suppresses colorectal...", "abstract": "...", "similarity": 0.89}, ...]
        """
        query_string = (
            f"Evidence for {drug} as a treatment for {disease}, "
            "including clinical trials, efficacy data, mechanism of action, "
            "and preclinical studies"
        )
        query_vector = (await embed_async([query_string]))[0]

        rows = db.execute(
            text("""
                SELECT pmid, title, abstract, similarity
                FROM (
                    SELECT pmid, title, abstract,
                           1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                    FROM pubmed_abstracts
                    WHERE pmid = ANY(:pmids)
                ) sub
                ORDER BY similarity DESC
                LIMIT :top_k
            """),
            {
                "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
                "pmids": pmids,
                "top_k": top_k,
            },
        ).fetchall()

        results = [
            {
                "pmid": row[0],
                "title": row[1],
                "abstract": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]
        avg_similarity = (
            sum(r["similarity"] for r in results) / len(results) if results else 0.0
        )

        if wandb.run:
            pass
            # disease_key = disease.replace(" ", "_")
            # table = wandb.Table(columns=["pmid", "title", "similarity"])
            # for r in results:
            #     table.add_data(r["pmid"], r["title"], r["similarity"])
            # wandb.log({
            #     f"semantic_search/{disease_key}/query": query_string,
            #     f"semantic_search/{disease_key}/candidate_pmids": len(pmids),
            #     f"semantic_search/{disease_key}/results_returned": len(results),
            #     f"semantic_search/{disease_key}/top_similarity": results[0]["similarity"] if results else None,
            #     f"semantic_search/{disease_key}/results_table": table,
            # })
            # wandb.log({
            #     "disease": disease,
            #     "drug": drug,
            #     "avg_similarity_score": avg_similarity,
            # })
            # table = wandb.Table(columns=["drug", "disease", "avg_similarity"])
            # table.add_data(drug, disease, avg_similarity)
            # wandb.log({"searches": table})

        return results

    async def synthesize(
        self, drug: str, disease: str, top_abstracts: list[AbstractResult]
    ) -> EvidenceSummary:
        """Summarize PubMed evidence for a drug-disease pair using an LLM.

        Formats the top abstracts from semantic_search into a prompt, calls the LLM,
        and parses the JSON response into an EvidenceSummary.

        Args:
            drug: Drug name (e.g. "metformin").
            disease: Candidate disease (e.g. "colorectal cancer").
            top_abstracts: Output of semantic_search — list of dicts with keys
                "pmid", "title", "abstract", "similarity".

        Returns:
            EvidenceSummary with all fields populated from the LLM response.
        """
        formatted = "\n\n".join(
            f"PMID: {r['pmid']}\nTitle: {r['title']}\nAbstract: {r['abstract']}"
            for r in top_abstracts
        )

        template = (_PROMPTS_DIR / "synthesize.txt").read_text()
        prompt = template.format(
            drug_name=drug, disease_name=disease, abstracts=formatted
        )

        response = await query_llm(prompt)
        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        stripped = response.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```", 2)[1]
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
            stripped = stripped.rsplit("```", 1)[0].strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error(
                "synthesize: failed to parse LLM response: %s\nResponse was: %s",
                e,
                response,
            )
            raise
        return EvidenceSummary(**data)

    async def extract_organ_term(self, disease_name: str) -> str:
        """Return the primary organ or tissue for a disease name via a small LLM call."""
        cached = cache_get("organ_term", {"disease_name": disease_name}, self.cache_dir)
        if cached is not None:
            logger.debug("Cache hit for organ_term: %s", disease_name)
            return cached

        template = (_PROMPTS_DIR / "extract_organ_term.txt").read_text()
        prompt = template.format(disease_name=disease_name)
        result = await query_small_llm(prompt)
        organ_term = result.strip()

        cache_set(
            "organ_term",
            {"disease_name": disease_name},
            organ_term,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        logger.debug(
            "Extracted organ term '%s' for disease '%s'", organ_term, disease_name
        )
        return organ_term

    async def expand_search_terms(
        self, drug_name: str, disease_name: str, drug_profile: DrugProfile
    ) -> list[str]:
        """Use LLM to generate diverse PubMed search queries from a drug-disease pair.

        Combines the drug's synonyms, gene targets, mechanisms of action, and ATC
        classifications with the organ term extracted from the disease name to produce
        a broad set of complementary PubMed queries. Results are cached by drug/disease
        pair and deduplicated (case-insensitive) before return.

        Args:
            drug_name: Common drug name (e.g. "metformin").
            disease_name: Target indication (e.g. "colorectal cancer").
            drug_profile: DrugProfile built from Open Targets + ChEMBL data.

        Returns:
            Deduplicated list of PubMed keyword queries ready to pass to fetch_and_cache.

        Examples:
            >>> # metformin × colorectal cancer might return:
            >>> [
            ...     "metformin colorectal cancer",
            ...     "metformin colon tumor",
            ...     "AMPK colorectal cancer",
            ...     "biguanide colon neoplasm",
            ...     "metformin PRKAB1 cancer",
            ... ]

            >>> # semaglutide × non-alcoholic steatohepatitis might return:
            >>> [
            ...     "semaglutide NASH",
            ...     "GLP-1 receptor agonist liver fibrosis",
            ...     "semaglutide non-alcoholic fatty liver disease",
            ...     "ozempic hepatic steatosis",
            ...     "GLP1R liver inflammation",
            ... ]
        """
        cached = cache_get(
            "expand_search_terms",
            {"drug_name": drug_name, "disease_name": disease_name},
            self.cache_dir,
        )
        if cached is not None:
            logger.debug(
                "Cache hit for expand_search_terms: %s / %s", drug_name, disease_name
            )
            return cached

        organ_term = await self.extract_organ_term(disease_name)

        template = (_PROMPTS_DIR / "expand_search_terms.txt").read_text()
        prompt = template.format(
            drug_name=drug_name,
            disease_name=disease_name,
            organ_term=organ_term,
            synonyms=", ".join(drug_profile.synonyms),
            target_gene_symbols=", ".join(drug_profile.target_gene_symbols),
            mechanisms_of_action=", ".join(drug_profile.mechanisms_of_action),
            atc_codes=", ".join(drug_profile.atc_codes),
            atc_descriptions=", ".join(drug_profile.atc_descriptions),
            drug_type=drug_profile.drug_type,
        )

        llm_output = await query_small_llm(prompt)
        raw: list[str] = parse_llm_response(llm_output)

        # Case-normalised dedup: lowercase+strip as key, preserve original casing
        seen: dict[str, str] = {}
        for term in raw:
            key = term.lower().strip()
            if key not in seen:
                seen[key] = term
        deduped = list(seen.values())

        cache_set(
            "expand_search_terms",
            {"drug_name": drug_name, "disease_name": disease_name},
            deduped,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        logger.debug(
            "expand_search_terms returned %d queries for %s / %s",
            len(deduped),
            drug_name,
            disease_name,
        )
        return deduped

    async def get_disease_synonyms(self, disease: str) -> list[str]:
        """Return alternate names for a disease via LLM.

        Args:
            disease: Disease name (e.g. "colorectal cancer").

        Returns:
            List of synonyms (e.g. ["colon cancer", "bowel cancer", "CRC"]).
        """
        template = (_PROMPTS_DIR / "disease_synonyms.txt").read_text()
        prompt = template.format(disease=disease)

        llm_output = await query_small_llm(prompt)
        return parse_llm_response(llm_output)
