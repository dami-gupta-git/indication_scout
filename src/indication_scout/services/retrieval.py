"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import asyncio
import calendar
import json
import logging
from datetime import date
from pathlib import Path

import wandb
from pydantic import BaseModel

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from typing_extensions import deprecated

from indication_scout.config import get_settings
from indication_scout.constants import CACHE_TTL
from indication_scout.data_sources.chembl import ChEMBLClient, get_all_drug_names
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
from indication_scout.services.llm import (
    parse_llm_response,
    query_llm,
    query_small_llm,
    strip_markdown_fences,
)
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Pubtype multiplicative boosts applied to semantic similarity to surface
# primary clinical evidence (RCTs, phase trials) over reviews/commentary
# in the final top-k. Types not listed get a neutral 1.0 boost; the
# per-record boost is max() across the record's pubtype list, so a paper
# tagged both "Journal Article" and "Randomized Controlled Trial" gets
# the RCT boost.
PUBTYPE_BOOSTS: dict[str, float] = {
    "Randomized Controlled Trial": 2.0,
    "Clinical Trial, Phase III": 1.8,
    "Clinical Trial, Phase II": 1.6,
    "Clinical Trial": 1.5,
    "Meta-Analysis": 1.3,
    "Systematic Review": 1.2,
    "Review": 0.6,
    "Comment": 0.3,
    "Editorial": 0.3,
    "Letter": 0.3,
}
PUBTYPE_BOOST_DEFAULT: float = 1.0


class AbstractResult(BaseModel):
    pmid: str
    title: str
    abstract: str
    similarity: float
    pubtype: list[str] = []


class RetrievalService:
    """Stateful retrieval service bound to a specific cache directory.

    Instantiate with the appropriate cache_dir for the calling context
    (e.g. DEFAULT_CACHE_DIR for production, TEST_CACHE_DIR for tests).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    # @deprecated
    # async def _normalize_disease_groups(
    #     self, diseases: dict[str, set[str]]
    # ) -> dict[str, set[str]]:
    #     """Normalize disease names via LLM and merge groups that collapse to the same key.
    #
    #     Args:
    #         diseases: Dict mapping disease name to set of competitor drug names.
    #
    #     Returns:
    #         Dict with normalized disease names as keys and unioned drug sets.
    #     """
    #     original_names = list(diseases.keys())
    #     norm_map = await llm_normalize_disease_batch(original_names)
    #
    #     merged: dict[str, set[str]] = {}
    #     for original in original_names:
    #         normalized = norm_map[original]
    #         key = normalized.split(" OR ")[0].strip().lower()
    #         if key in merged:
    #             merged[key] |= diseases[original]
    #         else:
    #             merged[key] = set(diseases[original])
    #     return merged

    async def get_drug_competitors(
        self, chembl_id: str, date_before: date | None = None
    ) -> dict[str, set[str]]:
        """Fetch top disease indications and their competitor drugs from Open Targets.

        Fetches raw competitor data from the client, then uses an LLM to merge
        duplicate disease names and remove overly broad terms before returning.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            date_before: Optional temporal holdout cutoff. Forwarded to the
                OT client to suppress its current-state approved-indications
                strip; cache key is keyed on the cutoff so cutoff and no-cutoff
                runs do not share cached competitor lists.

        Returns:
            Dict mapping disease name to set of competitor drug names.
        """
        cache_params = {
            "chembl_id": chembl_id,
            "date_before": date_before.isoformat() if date_before else None,
            "top_k": _settings.literature_top_k,
        }
        cached = cache_get("competitors_merged", cache_params, self.cache_dir)
        if cached is not None and len(cached) > 0:
            # logger.warning("[COMP] cache HIT for %r, %d diseases: %s",
            #                chembl_id, len(cached), list(cached.keys()))
            return {disease: set(drugs) for disease, drugs in cached.items()}

        async with OpenTargetsClient(cache_dir=self.cache_dir) as client:
            raw: CompetitorRawData = await client.get_drug_competitors(
                chembl_id, date_before=date_before
            )
            # logger.warning("[COMP] raw from OT: %d diseases: %s",
            #                len(raw["diseases"]), list(raw["diseases"].keys()))

        top_40 = raw["diseases"]
        logger.warning(
            "[COMP] after normalize: %d diseases: %s", len(top_40), list(top_40.keys())
        )

        drug_indications = raw["drug_indications"]
        disease_names = list(top_40.keys())
        merge_result = await merge_duplicate_diseases(disease_names, drug_indications)
        logger.warning(
            "[COMP] merge_result: merge=%s remove=%s",
            merge_result["merge"],
            merge_result["remove"],
        )

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
        top_15 = dict(list(sorted_data.items())[: _settings.literature_top_k])
        logger.warning("[COMP] final top_15: %s", list(top_15.keys()))

        cache_set(
            "competitors_merged",
            cache_params,
            {disease: list(drugs) for disease, drugs in top_15.items()},
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return top_15

    async def build_drug_profile(self, chembl_id: str) -> DrugProfile:
        """Fetch drug + target data from Open Targets, enrich with ATC descriptions from ChEMBL,
        and return a DrugProfile ready for use in search term expansion.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").

        Returns:
            DrugProfile with all fields populated. atc_descriptions will be [] if the drug
            has no ATC classifications.
        """
        async with OpenTargetsClient(cache_dir=self.cache_dir) as open_targets_client:
            rich = await open_targets_client.get_rich_drug_data(chembl_id)

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

    @staticmethod
    def _parse_pub_date_conservative(raw: str | None) -> date | None:
        """Parse a PubMed pub_date string into a conservative date.

        PubMed publication dates come in mixed formats from _parse_pubmed_xml:
            "2023"            → year only
            "2023-Mar"        → year + 3-letter month
            "2023-03"         → year + numeric month
            "2023-03-15"      → full ISO date
        For partial dates we use the LAST day of the partial range so the
        cutoff comparison errs toward "later" (a paper dated "2023-Mar"
        becomes 2023-03-31, which is correctly excluded by a 2023-04-01
        cutoff but not by a 2023-04-30 one). Returns None for missing or
        unparseable input — caller decides what to do with that.
        """
        if not raw:
            return None
        raw = raw.strip()
        if not raw:
            return None

        # Full ISO date
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass

        parts = raw.split("-")
        try:
            year = int(parts[0])
        except (ValueError, IndexError):
            return None

        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        if len(parts) == 1:
            # "YYYY" → Dec 31 of that year
            return date(year, 12, 31)

        month_raw = parts[1].lower()[:3]
        month = month_map.get(month_raw)
        if month is None:
            try:
                month = int(parts[1])
            except ValueError:
                return None
        if not 1 <= month <= 12:
            return None

        if len(parts) == 2:
            # "YYYY-Mon" or "YYYY-MM" → last day of that month
            last_day = calendar.monthrange(year, month)[1]
            return date(year, month, last_day)

        try:
            day = int(parts[2])
            return date(year, month, day)
        except ValueError:
            return None

    def _read_pub_dates_from_db(
        self, pmids: list[str], db: Session
    ) -> dict[str, str | None]:
        """Bulk-read (pmid → raw pub_date string) from pubmed_abstracts.

        Returns a dict containing only the PMIDs found in the DB. Values
        may be None (the column is nullable). PMIDs not in the DB are
        absent from the result.
        """
        if not pmids:
            return {}
        rows = db.execute(
            text(
                "SELECT pmid, pub_date FROM pubmed_abstracts WHERE pmid = ANY(:pmids)"
            ),
            {"pmids": pmids},
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    async def _filter_pmids_by_date(
        self,
        pmids: list[str],
        date_before: date,
        db: Session,
        client: PubMedClient,
    ) -> list[str]:
        """Drop PMIDs whose publication date is on/after `date_before`.

        Reads pub_date from pgvector for already-stored PMIDs (no HTTP),
        falls back to esummary for the unknowns. The fallback uses the
        existing client._filter_pmids_by_date so the esummary parsing
        logic stays in one place. Missing or unparseable dates are KEPT
        — same policy as client._filter_pmids_by_date.
        """
        if not pmids:
            return []

        known = self._read_pub_dates_from_db(pmids, db)
        from_db_kept: list[str] = []
        unknown: list[str] = []
        for pmid in pmids:
            if pmid not in known:
                unknown.append(pmid)
                continue
            parsed = self._parse_pub_date_conservative(known[pmid])
            if parsed is None:
                # Stored row but no usable date — match the production
                # policy of keeping the PMID rather than dropping it.
                from_db_kept.append(pmid)
                continue
            if parsed < date_before:
                from_db_kept.append(pmid)

        logger.debug(
            "_filter_pmids_by_date: %d total, %d known in DB, %d unknown → "
            "esummary; %d kept from DB",
            len(pmids),
            len(known),
            len(unknown),
            len(from_db_kept),
        )

        if not unknown:
            return from_db_kept

        from_esummary_kept = await client._filter_pmids_by_date(unknown, date_before)
        # Preserve original input order
        kept_set = set(from_db_kept) | set(from_esummary_kept)
        return [p for p in pmids if p in kept_set]

    async def fetch_and_cache(
        self,
        queries: list[str],
        db: Session,
        date_before: date | None = None,
    ) -> list[str]:
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
            date_before: Optional temporal holdout cutoff; only articles published
                before this date are returned by PubMed search.

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
                    client.search(
                        query,
                        max_results=_settings.pubmed_max_results,
                        date_before=date_before,
                    )
                    for query in queries
                ]
            )

            # Flatten and deduplicate while preserving first-seen order
            all_pmids: list[str] = list(
                dict.fromkeys(pmid for pmids in search_results for pmid in pmids)
            )

            # 1.5 Cutoff post-guard. PubMed's eutils maxdate filter is not
            # strictly respected, so we re-verify each PMID's publication
            # date. Reads pub_date from pgvector for already-stored PMIDs
            # (no HTTP) and only falls back to esummary for unknowns.
            # Massively reduces NCBI traffic on re-runs.
            if date_before is not None:
                all_pmids = await self._filter_pmids_by_date(
                    all_pmids, date_before, db, client
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
        self,
        disease: str,
        chembl_id: str,
        pmids: list[str],
        db: Session,
        date_before: date | None = None,
    ) -> list[AbstractResult]:
        """For a given drug, disease, and list of PMIDs, return top-k most similar abstracts from pgvector

        Constructs a natural-language query from drug and disease (e.g. "Evidence for metformin
        as a treatment for colorectal cancer..."), embeds it with BioLORD-2023, then runs a
        cosine similarity search restricted to the given PMIDs.

        Args:
            disease: e.g. "colorectal cancer"
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            pmids: e.g. ["29734553", "31245678", "30198432"]
            date_before: Optional temporal holdout cutoff. pgvector is a
                shared cache that may hold abstracts fetched in a prior
                non-holdout run, so re-apply the same date post-guard here
                before re-ranking.

        Returns:
            List of dicts ranked by descending similarity, e.g.:
            [{"pmid": "29734553", "title": "Metformin suppresses colorectal...", "abstract": "...", "similarity": 0.89}, ...]
        """
        # Holdout post-guard: drop PMIDs published on/after the cutoff that
        # may have leaked in via the shared pgvector cache. Reuses the same
        # filter as fetch_and_cache so the date policy stays in one place.
        if date_before is not None:
            async with PubMedClient(cache_dir=self.cache_dir) as client:
                pmids = await self._filter_pmids_by_date(
                    pmids, date_before, db, client
                )
            if not pmids:
                return []

        pref_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]
        query_string = (
            f"Evidence for {pref_name} as a treatment for {disease}, "
            "including clinical trials, efficacy data, mechanism of action, "
            "and preclinical studies"
        )
        query_vector = (await embed_async([query_string]))[0]

        # Over-fetch cap: pull top-N by similarity from pgvector, then rerank
        # by pubtype in Python. Cap gives the boost headroom to reorder
        # (a 2x RCT boost at rank 50 can beat a review at rank 5) without
        # an unbounded scan if the candidate pool grows.
        top_k = _settings.semantic_search_top_k
        rerank_cap = max(top_k * 10, 100)

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
                LIMIT :rerank_cap
            """),
            {
                "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
                "pmids": pmids,
                "rerank_cap": rerank_cap,
            },
        ).fetchall()

        if not rows:
            return []

        candidate_pmids = [row[0] for row in rows]
        async with PubMedClient(cache_dir=self.cache_dir) as client:
            pubtypes_map = await client.fetch_pubtypes(candidate_pmids)

        if not pubtypes_map:
            logger.warning(
                "semantic_search: fetch_pubtypes returned empty for all %d "
                "candidates (%s / %s); pubtype boost is a no-op for this call",
                len(candidate_pmids), chembl_id, disease,
            )

        scored: list[tuple[AbstractResult, float, float]] = []
        for row in rows:
            pmid, title, abstract, similarity = row[0], row[1], row[2], float(row[3])
            pubtypes = pubtypes_map.get(pmid, [])
            boost = max(
                (PUBTYPE_BOOSTS.get(pt, PUBTYPE_BOOST_DEFAULT) for pt in pubtypes),
                default=PUBTYPE_BOOST_DEFAULT,
            )
            final_score = similarity * boost
            result = AbstractResult(
                pmid=pmid,
                title=title,
                abstract=abstract,
                similarity=similarity,
                pubtype=pubtypes,
            )
            scored.append((result, boost, final_score))

        scored.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            "semantic_search rerank top-20 for %s / %s (%d candidates, cap=%d):",
            chembl_id, disease, len(scored), rerank_cap,
        )
        # for result, boost, final_score in scored[:20]:
        #     logger.info(
        #         "  pmid=%s title=%r sim=%.4f pubtype=%s boost=%.2f final=%.4f",
        #         result.pmid,
        #         result.title[:60],
        #         result.similarity,
        #         result.pubtype,
        #         boost,
        #         final_score,
        #     )

        return [item[0] for item in scored[:top_k]]

    async def synthesize(
        self,
        chembl_id: str,
        disease: str,
        top_abstracts: list[AbstractResult],
        holdout_mode: bool = False,
    ) -> EvidenceSummary:
        """Summarize PubMed evidence for a drug-disease pair using an LLM.

        Formats the top abstracts from semantic_search into a prompt, calls the LLM,
        and parses the JSON response into an EvidenceSummary.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            disease: Candidate disease (e.g. "colorectal cancer").
            top_abstracts: Output of semantic_search — list of dicts with keys
                "pmid", "title", "abstract", "similarity".
            holdout_mode: When True, swap to synthesize_holdout.txt — a relaxed
                rubric that allows class-level evidence (other drugs in the
                same mechanism class) to score weak/moderate when the cutoff
                predates drug-specific publications. Used during temporal
                holdout runs only.

        Returns:
            EvidenceSummary with all fields populated from the LLM response.
        """
        # Cache key uses sorted PMIDs so two abstract orderings that contain the
        # same evidence collapse to one cache entry. Holdout mode uses a different
        # prompt template, so it must be part of the key.
        cache_params = {
            "chembl_id": chembl_id,
            "disease": disease,
            "pmids": sorted(r.pmid for r in top_abstracts),
            "holdout_mode": holdout_mode,
            "llm_model": _settings.llm_model,
        }
        cached = cache_get("synthesize", cache_params, self.cache_dir)
        if cached is not None:
            # logger.debug(
            #     "Cache hit for synthesize: %s / %s (%d pmids)",
            #     chembl_id, disease, len(cache_params["pmids"]),
            # )
            return EvidenceSummary(**cached)

        pref_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]
        formatted = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract}"
            for r in top_abstracts
        )

        prompt_file = "synthesize_holdout.txt" if holdout_mode else "synthesize.txt"
        logger.debug("synthesize prompt: %s", prompt_file)
        template = (_PROMPTS_DIR / prompt_file).read_text()
        prompt = template.format(
            drug_name=pref_name, disease_name=disease, abstracts=formatted
        )

        response = await query_llm(prompt)
        stripped = strip_markdown_fences(response)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error(
                "synthesize: failed to parse LLM response: %s\nResponse was: %s",
                e,
                response,
            )
            raise
        summary = EvidenceSummary(**data)
        cache_set(
            "synthesize",
            cache_params,
            summary.model_dump(mode="json"),
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return summary

    async def extract_organ_term(self, disease_name: str) -> str:
        """Return the primary organ or tissue for a disease name via a small LLM call."""
        cache_params = {
            "disease_name": disease_name,
            "small_llm_model": _settings.small_llm_model,
        }
        cached = cache_get("organ_term", cache_params, self.cache_dir)
        if cached is not None:
            # logger.debug("Cache hit for organ_term: %s", disease_name)
            return cached

        template = (_PROMPTS_DIR / "extract_organ_term.txt").read_text()
        prompt = template.format(disease_name=disease_name)
        result = await query_small_llm(prompt)
        organ_term = result.strip()

        cache_set(
            "organ_term",
            cache_params,
            organ_term,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        logger.debug(
            "Extracted organ term '%s' for disease '%s'", organ_term, disease_name
        )
        return organ_term

    async def expand_search_terms(
        self, chembl_id: str, disease_name: str, drug_profile: DrugProfile
    ) -> list[str]:
        """Use LLM to generate diverse PubMed search queries from a drug-disease pair.

        Combines the drug's synonyms, gene targets, mechanisms of action, and ATC
        classifications with the organ term extracted from the disease name to produce
        a broad set of complementary PubMed queries. Results are cached by drug/disease
        pair and deduplicated (case-insensitive) before return.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
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
        cache_params = {
            "chembl_id": chembl_id,
            "disease_name": disease_name,
            "small_llm_model": _settings.small_llm_model,
        }
        cached = cache_get(
            "expand_search_terms",
            cache_params,
            self.cache_dir,
        )
        if cached is not None:
            # logger.debug(
            #     "Cache hit for expand_search_terms: %s / %s", chembl_id, disease_name
            # )
            return cached

        all_names = await get_all_drug_names(chembl_id, self.cache_dir)
        pref_name = all_names[0]
        synonyms = all_names[1:]
        organ_term = await self.extract_organ_term(disease_name)

        # Resolve the disease string to its canonical MeSH preferred term and
        # pass that into the prompt instead of the raw disease name. PubMed's
        # auto-term-mapping breaks when bare multi-word phrases appear on
        # either side of AND, so the prompt template instructs the LLM to wrap
        # the disease term in double quotes for reliable parsing. If the
        # MeSH lookup misses, fall back to the raw disease name.
        from indication_scout.services.disease_helper import resolve_mesh_id
        mesh_result = await resolve_mesh_id(disease_name)
        disease_term = mesh_result[1] if mesh_result else disease_name
        if mesh_result is None:
            logger.warning(
                "expand_search_terms: MeSH resolution missed for %r; using raw disease name",
                disease_name,
            )

        template = (_PROMPTS_DIR / "expand_search_terms.txt").read_text()
        prompt = template.format(
            drug_name=pref_name,
            disease_name=disease_term,
            organ_term=organ_term,
            synonyms=", ".join(synonyms),
            target_gene_symbols=", ".join(drug_profile.target_gene_symbols),
            mechanisms_of_action=", ".join(drug_profile.mechanisms_of_action),
            atc_codes=", ".join(drug_profile.atc_codes),
            atc_descriptions=", ".join(drug_profile.atc_descriptions),
            drug_type=drug_profile.drug_type,
        )

        llm_output = await query_small_llm(prompt)
        try:
            raw: list[str] = parse_llm_response(llm_output)
        except json.JSONDecodeError:
            logger.error(
                "expand_search_terms: failed to parse LLM output for chembl_id=%s "
                "disease_name=%r. Raw output:\n%s",
                chembl_id,
                disease_name,
                llm_output,
            )
            raise

        # Substitute the <DISEASE> placeholder with the quoted MeSH preferred term.
        # The prompt instructs the LLM to emit `<DISEASE>` instead of writing the
        # disease name directly, so the JSON array never contains embedded quotes.
        quoted_disease = f'"{disease_term}"'
        substituted = [q.replace("<DISEASE>", quoted_disease) for q in raw]

        # Case-normalised dedup: lowercase+strip as key, preserve original casing
        seen: dict[str, str] = {}
        for term in substituted:
            key = term.lower().strip()
            if key not in seen:
                seen[key] = term
        deduped = list(seen.values())

        cache_set(
            "expand_search_terms",
            cache_params,
            deduped,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        logger.debug(
            "expand_search_terms returned %d queries for %s / %s",
            len(deduped),
            chembl_id,
            disease_name,
        )
        return deduped
