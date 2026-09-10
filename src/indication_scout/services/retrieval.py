"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import asyncio
import calendar
import json
import logging
import re
import time
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from typing_extensions import deprecated

from indication_scout.agents.literature.pubmed_ae import search_adverse_events
from indication_scout.config import get_settings
from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    CACHE_TTL,
    LITERATURE_TRIAL_REFERENCE_RESERVE,
    PUBMED_NCT_QUERY_BATCH_SIZE,
    SAFETY_QUOTE_MAX_WORDS,
    SAFETY_TOP_ADVERSE_EVENTS,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient, get_all_drug_names
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import (
    CompetitorRawData,
    OpenTargetsClient,
)
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import (
    EvidenceDirectionJudgment,
    EvidenceSummary,
    PmidJudgment,
)
from indication_scout.models.model_fda import FDALabelSafetyRecord
from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.models.model_safety import (
    DrugSafetyAssessment,
    SafetyPaperVerdict,
)
from indication_scout.services.citation_guard import (
    strip_findings_with_unknown_pmids,
    strip_sentences_with_unknown_pmids,
    unknown_pmids,
)
from indication_scout.services.disease_helper import (
    llm_normalize_disease_batch,
    merge_duplicate_diseases,
    resolve_mesh_id,
)
from indication_scout.services.embeddings import embed_async
from indication_scout.services.llm import (
    parse_last_json_object,
    parse_llm_response,
    query_llm,
    query_small_llm,
    strip_markdown_fences,
)
from indication_scout.services.progress import PHASE_LITERATURE, emit_progress
from indication_scout.sqlalchemy.pubmed_abstracts import PubmedAbstracts
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _filter_overly_broad_candidates(
    candidates: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Remove candidates whose words are all generic blocklisted terms."""
    return {
        disease: drugs
        for disease, drugs in candidates.items()
        if not {word.lower() for word in disease.split()} <= BROADENING_BLOCKLIST
    }


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


_PMID_DIRECTION_PROMPT = (_PROMPTS_DIR / "pmid_direction.txt").read_text()
_EVIDENCE_DIRECTION_PROMPT = (_PROMPTS_DIR / "judge_evidence_direction.txt").read_text()
_PMID_DRUG_IDENTITY_PROMPT = (_PROMPTS_DIR / "pmid_drug_identity.txt").read_text()
# "class_level" = the abstract's result belongs to the drug's mechanistic class, not to the drug
# alone (a pooled sildenafil+tadalafil meta-analysis, a review of PDE5 inhibitors). It is judged
# PER PAPER because evidence_basis is one verdict for the whole candidate: without a per-paper
# value a single drug-specific animal study made the candidate "drug_specific" and the class-level
# papers then counted at full weight in the supporting list, which is what the strength cap exists
# to prevent (sildenafil x ischemic stroke read "moderate, supports" on rodent data plus two
# class-level papers).
_PMID_DRUG_IDENTITY_VERDICTS = {"studied", "class_level", "not_studied"}
_PMID_TREATS_DISEASE_PROMPT = (_PROMPTS_DIR / "pmid_treats_disease.txt").read_text()
_PMID_TREATS_DISEASE_VERDICTS = {"treats", "not_treats"}
# Appended on the retry after a prose field cited a PMID that was not in the supplied abstracts.
_PMID_RETRY_NOTE = """

CORRECTION: your previous answer cited a PMID that is not among the abstracts above. Every PMID you
write must be copied digit by digit from those abstracts. Do not cite any other paper.
"""
_CONTROLLED_DESIGN_PATTERN = re.compile(
    r"\brandomi[sz](?:ed|ation)\b|"
    r"\bplacebo[- ]controlled\b|"
    r"\b(?:double|single)[- ]blind\b|"
    r"\bcross[- ]?over\b|"
    r"\bcontrolled (?:clinical )?(?:trial|study)\b|"
    r"\bcompared (?:with|to) placebo\b|"
    r"\brct\b",
    re.IGNORECASE,
)
_CONTROLLED_PUBTYPES = {"Randomized Controlled Trial", "Controlled Clinical Trial"}


def _mentions_exact_drug(drug_names: list[str], result: "AbstractResult") -> bool:
    """Return whether the title or abstract contains a supplied name for the exact drug."""

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

    text = f" {normalize(f'{result.title} {result.abstract}')} "
    return any(
        normalized_name and f" {normalized_name} " in text
        for name in drug_names
        if (normalized_name := normalize(name))
    )


async def _judge_pmid_drug_identity(
    chembl_id: str,
    drug_names: list[str],
    abstracts: list["AbstractResult"],
    cache_dir: Path,
) -> dict[str, str]:
    """Return each PMID's drug-role verdict: "studied", "class_level" or "not_studied".

    Absence of every accepted drug name is a deterministic rejection. Remaining abstracts are judged
    in isolated calls and cached by ChEMBL ID and PMID, so changing the surrounding batch cannot
    affect a decision. The question does not depend on the candidate disease, so one entry serves
    every candidate in a run. Missing or invalid responses fail closed as "not_studied" and are not
    cached.
    """
    if not abstracts:
        return {}

    names = list(dict.fromkeys(name.strip() for name in drug_names if name.strip()))
    formatted_names = "\n".join(f"- {name}" for name in names)
    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(result: "AbstractResult") -> tuple[str, str]:
        if not _mentions_exact_drug(names, result):
            # Logged because this branch excludes a paper WITHOUT an LLM call: a name the corpus
            # uses but ChEMBL does not carry would otherwise drop real evidence with no trace.
            logger.info(
                "pmid_drug_identity: no accepted name for %s in PMID %s; excluding paper",
                chembl_id,
                result.pmid,
            )
            return result.pmid, "not_studied"

        cache_params = {
            "chembl_id": chembl_id,
            "pmid": result.pmid,
            "drug_names": sorted(names),
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "drug_identity_v2",
        }
        cached = cache_get("pmid_drug_identity", cache_params, cache_dir)
        if isinstance(cached, str) and cached in _PMID_DRUG_IDENTITY_VERDICTS:
            return result.pmid, cached

        prompt = _PMID_DRUG_IDENTITY_PROMPT.format(
            drug=names[0],
            drug_names=formatted_names,
            pmid=result.pmid,
            title=result.title,
            abstract=result.abstract,
        )
        async with semaphore:
            try:
                response = await query_small_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "pmid_drug_identity: LLM call failed for %s / PMID %s; excluding paper: %s",
                    chembl_id,
                    result.pmid,
                    exc,
                )
                return result.pmid, "not_studied"
        data = parse_last_json_object(response)
        verdict = (
            str(data.get("verdict", "")).strip().lower()
            if isinstance(data, dict)
            else ""
        )
        if verdict not in _PMID_DRUG_IDENTITY_VERDICTS:
            logger.warning(
                "pmid_drug_identity: unusable verdict for %s / PMID %s; excluding paper. "
                "Response was: %s",
                chembl_id,
                result.pmid,
                response,
            )
            return result.pmid, "not_studied"

        cache_set(
            "pmid_drug_identity",
            cache_params,
            verdict,
            cache_dir,
            ttl=CACHE_TTL,
        )
        return result.pmid, verdict

    decisions = await asyncio.gather(*(judge_one(result) for result in abstracts))
    return dict(decisions)


async def _judge_pmid_treats_disease(
    chembl_id: str,
    drug: str,
    disease: str,
    abstracts: list["AbstractResult"],
    cache_dir: Path,
) -> dict[str, bool]:
    """Return whether each PMID gave the drug IN ORDER TO treat this disease.

    Isolated one-question-per-abstract calls, cached by ChEMBL ID, PMID and disease, so changing the
    surrounding batch cannot affect a decision. This is the therapeutic-target gate the combined
    synthesize prompt still owned; kept there, a related but distinct condition could grade as the
    candidate disease (sildenafil for neonatal hypoxic-ischemic brain injury read as evidence for
    ischemic stroke). Missing or invalid responses fail closed as ``False`` and are not cached.
    """
    if not abstracts:
        return {}

    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(result: "AbstractResult") -> tuple[str, bool]:
        cache_params = {
            "chembl_id": chembl_id,
            "pmid": result.pmid,
            "disease": disease,
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "treats_disease_v5",
        }
        cached = cache_get("pmid_treats_disease", cache_params, cache_dir)
        if isinstance(cached, str) and cached in _PMID_TREATS_DISEASE_VERDICTS:
            return result.pmid, cached == "treats"

        prompt = _PMID_TREATS_DISEASE_PROMPT.format(
            drug=drug,
            disease=disease,
            pmid=result.pmid,
            title=result.title,
            abstract=result.abstract,
        )
        async with semaphore:
            try:
                response = await query_small_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "pmid_treats_disease: LLM call failed for %s / %s / PMID %s; excluding "
                    "paper: %s",
                    chembl_id,
                    disease,
                    result.pmid,
                    exc,
                )
                return result.pmid, False
        data = parse_last_json_object(response)
        verdict = (
            str(data.get("verdict", "")).strip().lower()
            if isinstance(data, dict)
            else ""
        )
        if verdict not in _PMID_TREATS_DISEASE_VERDICTS:
            logger.warning(
                "pmid_treats_disease: unusable verdict for %s / %s / PMID %s; excluding paper. "
                "Response was: %s",
                chembl_id,
                disease,
                result.pmid,
                response,
            )
            return result.pmid, False

        cache_set(
            "pmid_treats_disease",
            cache_params,
            verdict,
            cache_dir,
            ttl=CACHE_TTL,
        )
        return result.pmid, verdict == "treats"

    decisions = await asyncio.gather(*(judge_one(result) for result in abstracts))
    return dict(decisions)


async def _judge_pmid_directions(
    drug: str, disease: str, relevant_abstracts: list["AbstractResult"]
) -> dict[str, PmidJudgment]:
    """Per-PMID direction (supporting | contradicting | mixed | neutral) AND study design over the RELEVANT abstracts, via
    a small isolated sub-call. This is the AUTHORITATIVE direction for each relevant abstract — it replaces both the
    in-prompt verdict-direction and the old regex guards, which could not reliably attribute a benefit to the right
    drug-arm (metformin × hepatic steatosis: a benefit that belonged to the comparator, or a metabolic-marker improvement,
    read as "supporting" for metformin). The narrow one-question-per-abstract framing handles attribution that phrase
    matching cannot. The same call answers whether each study is in humans and whether it is itself controlled, which
    phrase matching over the whole abstract could not (sildenafil × ischemic stroke: an uncontrolled human safety study
    whose background described placebo-controlled RAT experiments certified the pair as RCT-backed). Returns a
    {pmid: judgment} map; a PMID the sub-call omits or labels unrecognizably is left out (the caller keeps the synthesize
    verdict for it)."""
    if not relevant_abstracts:
        return {}
    formatted = "\n\n".join(
        f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract}"
        for r in relevant_abstracts
    )
    prompt = _PMID_DIRECTION_PROMPT.format(
        drug=drug, disease=disease, abstracts=formatted
    )
    response = await query_small_llm(prompt)
    data = parse_last_json_object(response)
    if not isinstance(data, dict):
        logger.warning(
            "pmid_direction: unparseable sub-call response for %s / %s: %s",
            drug,
            disease,
            response,
        )
        return {}
    out: dict[str, PmidJudgment] = {}
    valid_pmids = {r.pmid for r in relevant_abstracts}
    for pmid, payload in data.items():
        if str(pmid) not in valid_pmids or not isinstance(payload, dict):
            continue
        try:
            out[str(pmid)] = PmidJudgment(**payload)
        except ValidationError:
            # An unrecognized verdict drops the whole abstract, as before: the caller then keeps the synthesize verdict
            # for it, and it can never certify the pair as controlled human evidence.
            logger.warning(
                "pmid_direction: unusable judgment for PMID %s (%s / %s): %s",
                pmid,
                drug,
                disease,
                payload,
            )
    return out


async def _judge_overall_evidence_direction(
    drug: str,
    disease: str,
    abstracts: list["AbstractResult"],
    verdict_of: dict[str, str],
) -> EvidenceDirectionJudgment | None:
    """Weigh already-relevant papers when their efficacy verdicts conflict."""
    if not abstracts:
        return None

    formatted = "\n\n".join(
        f"PMID: {result.pmid}\n"
        f"Paper-level verdict: {verdict_of[result.pmid]}\n"
        f"Title: {result.title}\n"
        f"Abstract: {result.abstract}"
        for result in abstracts
    )
    prompt = _EVIDENCE_DIRECTION_PROMPT.format(
        drug=drug,
        disease=disease,
        abstracts=formatted,
    )
    allowed_pmids = {result.pmid for result in abstracts}
    context = f"{drug} / {disease} evidence direction"
    judgment: EvidenceDirectionJudgment | None = None
    for attempt in (1, 2):
        response = await query_llm(
            prompt if attempt == 1 else prompt + _PMID_RETRY_NOTE
        )
        data = parse_last_json_object(response)
        if not isinstance(data, dict):
            logger.warning(
                "evidence_direction: unparseable response for %s / %s: %s",
                drug,
                disease,
                response,
            )
            return None

        try:
            judgment = EvidenceDirectionJudgment(**data)
        except ValidationError as exc:
            logger.warning(
                "evidence_direction: invalid response for %s / %s: %s",
                drug,
                disease,
                exc,
            )
            return None
        if not judgment.summary.strip():
            logger.warning(
                "evidence_direction: incomplete response for %s / %s: %s",
                drug,
                disease,
                response,
            )
            return None
        bad = unknown_pmids(judgment.summary, allowed_pmids)
        for finding in judgment.key_findings:
            bad.extend(unknown_pmids(finding, allowed_pmids))
        if not bad:
            return judgment
        logger.error(
            "evidence_direction: cited PMID(s) %s not among the abstracts supplied for %s "
            "(attempt %d)",
            ", ".join(bad),
            context,
            attempt,
        )

    judgment.summary = strip_sentences_with_unknown_pmids(
        judgment.summary, allowed_pmids, context=context
    )
    judgment.key_findings = strip_findings_with_unknown_pmids(
        judgment.key_findings, allowed_pmids, context=context
    )
    if not judgment.summary.strip():
        logger.error(
            "evidence_direction: every sentence of the summary for %s cited an unknown PMID; "
            "keeping the synthesis summary instead",
            context,
        )
        return None
    return judgment


class AbstractResult(BaseModel):
    pmid: str
    title: str
    abstract: str
    similarity: float
    pubtype: list[str] = []


class SafetySearchResult(BaseModel):
    """Adverse-event abstracts with retrieval provenance preserved."""

    drug_level: list[AbstractResult]
    disease_scoped: list[AbstractResult]

    @property
    def combined(self) -> list[AbstractResult]:
        """Return an order-preserving union for drug-wide safety synthesis."""
        by_pmid: dict[str, AbstractResult] = {}
        for abstract in self.drug_level + self.disease_scoped:
            by_pmid.setdefault(abstract.pmid, abstract)
        return list(by_pmid.values())


class RetrievalService:
    """Stateful retrieval service bound to a specific cache directory.

    Instantiate with the appropriate cache_dir for the calling context
    (e.g. DEFAULT_CACHE_DIR for production, TEST_CACHE_DIR for tests).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._label_safety_tasks: dict[
            str, asyncio.Task[list[FDALabelSafetyRecord]]
        ] = {}

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
            "logic_version": "cache_disease_aliases_v1",
        }
        cached = cache_get("competitors_merged", cache_params, self.cache_dir)
        if cached is not None:
            # logger.warning("[COMP] cache HIT for %r, %d diseases: %s",
            #                chembl_id, len(cached), list(cached.keys()))
            cached_candidates = {
                disease: set(drugs) for disease, drugs in cached.items()
            }
            return _filter_overly_broad_candidates(cached_candidates)

        async with OpenTargetsClient(cache_dir=self.cache_dir) as client:
            raw: CompetitorRawData = await client.get_drug_competitors(
                chembl_id, date_before=date_before
            )
            # logger.warning("[COMP] raw from OT: %d diseases: %s",
            #                len(raw["diseases"]), list(raw["diseases"].keys()))

        top_40 = raw["diseases"]

        drug_indications = raw["drug_indications"]
        disease_names = list(top_40.keys())
        merge_result = await merge_duplicate_diseases(disease_names, drug_indications)
        # logger.warning(
        #     "[COMP] merge_result: merge=%s remove=%s",
        #     merge_result["merge"],
        #     merge_result["remove"],
        # )

        for disease in merge_result["remove"]:
            if disease.lower() in top_40:
                del top_40[disease.lower()]

        removed = {n.lower() for n in merge_result["remove"]}
        aliases_by_disease: dict[str, list[str]] = {}
        for canonical, aliases in merge_result["merge"].items():
            canonical_lower = canonical.lower()
            aliases_lower = [a.lower() for a in aliases]
            all_names = [canonical_lower] + aliases_lower

            if canonical_lower in removed:
                surviving = [n for n in aliases_lower if n not in removed]
                if not surviving:
                    continue
                canonical_lower = surviving[0]

            source_names = [
                disease
                for disease in all_names
                if disease not in removed and disease in top_40
            ]
            combined: set[str] = set()
            source_present = False
            for disease in all_names:
                if disease in removed:
                    continue
                if disease in top_40:
                    source_present = True
                    combined |= top_40[disease]
                    if disease != canonical_lower:
                        del top_40[disease]
            if source_present:
                top_40[canonical_lower] = combined
                aliases_by_disease[canonical_lower] = [
                    disease for disease in source_names if disease != canonical_lower
                ]

        top_40 = _filter_overly_broad_candidates(top_40)
        sorted_data = dict(
            sorted(top_40.items(), key=lambda item: len(item[1]), reverse=True)
        )
        top_15 = dict(list(sorted_data.items())[: _settings.literature_top_k])
        for disease in top_15:
            cache_set(
                "disease_aliases",
                {"chembl_id": chembl_id, "disease": disease},
                aliases_by_disease.get(disease, []),
                self.cache_dir,
                ttl=CACHE_TTL,
            )
        # logger.warning("[COMP] final top_15: %s", list(top_15.keys()))

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

    async def cache_trial_reference_abstracts(
        self,
        pmids: list[str],
        db: Session,
        date_before: date | None,
    ) -> list[str]:
        """Fetch and embed trial-linked PubMed records independently of query retrieval."""
        linked_pmids = list(dict.fromkeys(str(pmid) for pmid in pmids if str(pmid)))
        if not linked_pmids:
            return []
        async with PubMedClient(cache_dir=self.cache_dir) as client:
            if date_before is not None:
                linked_pmids = await self._filter_pmids_by_date(
                    linked_pmids, date_before, db, client
                )
            stored = self.get_stored_pmids(linked_pmids, db)
            db.rollback()
            new_abstracts = await self.fetch_new_abstracts(
                linked_pmids, stored, client
            )
        abstracts_with_text = [abstract for abstract in new_abstracts if abstract.abstract]
        pairs = await self.embed_abstracts(abstracts_with_text)
        self.insert_abstracts(pairs, db)
        return linked_pmids

    async def find_trial_linked_pmids(
        self,
        drug: str,
        disease: str,
        date_before: date | None,
    ) -> list[str]:
        """Find publications tagged in PubMed with NCT IDs from the pair registry lookup."""
        resolved = await resolve_mesh_id(disease)
        if resolved is None:
            return []
        _, mesh_term = resolved
        async with ClinicalTrialsClient(cache_dir=self.cache_dir) as trials_client:
            trial_result = await trials_client.search_trials(
                drug, mesh_term, date_before=date_before
            )
        nct_ids = list(
            dict.fromkeys(trial.nct_id for trial in trial_result.trials if trial.nct_id)
        )
        if not nct_ids:
            return []

        max_results = _settings.pubmed_max_results
        pmids: list[str] = []
        async with PubMedClient(cache_dir=self.cache_dir) as pubmed_client:
            for start in range(0, len(nct_ids), PUBMED_NCT_QUERY_BATCH_SIZE):
                remaining = max_results - len(pmids)
                if remaining <= 0:
                    break
                batch = nct_ids[start : start + PUBMED_NCT_QUERY_BATCH_SIZE]
                query = " OR ".join(f"{nct_id}[si]" for nct_id in batch)
                found = await pubmed_client.search(
                    query,
                    max_results=remaining,
                    date_before=date_before,
                )
                pmids.extend(pmid for pmid in found if pmid not in pmids)
        return pmids

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

        emit_progress(PHASE_LITERATURE, f"Embedding {len(abstracts)} new abstracts")
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
        # The SELECT starts a transaction. End it before the possible PubMed
        # request below so the connection returns to the pool while awaiting I/O.
        db.rollback()
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
        direct_query: str | None = None,
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
            direct_query: Deterministic drug-disease query to retrieve completely.
                Other queries retain the configured relevance-result cap.

        Returns:
            Deduplicated list of all PMIDs returned by PubMed search across all queries.
            Note: not every returned PMID has a row in pubmed_abstracts — articles without
            an abstract (letters, editorials) are excluded from the vector store. Callers
            that pass this list to semantic_search will see those PMIDs silently skipped
            by the WHERE pmid = ANY(:pmids) clause, which is intentional and correct.
        """
        _t_search = time.perf_counter()
        async with PubMedClient(cache_dir=self.cache_dir) as client:
            # 1. Search all queries concurrently
            search_results = await asyncio.gather(
                *[
                    (
                        client.search_complete(
                            query,
                            page_size=_settings.pubmed_max_results,
                            date_before=date_before,
                        )
                        if query == direct_query
                        else client.search(
                            query,
                            max_results=_settings.pubmed_max_results,
                            date_before=date_before,
                        )
                    )
                    for query in queries
                ]
            )

            # Flatten and deduplicate while preserving first-seen order
            all_pmids: list[str] = list(
                dict.fromkeys(pmid for pmids in search_results for pmid in pmids)
            )
            _dt_search = time.perf_counter() - _t_search

            # Per-query attribution: how many PMIDs each query returned (PMIDs themselves
            # omitted to keep logs readable).
            for _q, _pmids in zip(queries, search_results):
                # logger.warning("[QUERYMAP] query=%r returned %d pmids", _q, len(_pmids))
                pass

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
            # Release the read transaction before fetching abstracts or embedding
            # them. The later insert starts and commits its own transaction.
            db.rollback()

            # 3. Single fetch for all new abstracts
            _t_fetch = time.perf_counter()
            new_abstracts = await self.fetch_new_abstracts(all_pmids, stored, client)
            _dt_fetch = time.perf_counter() - _t_fetch

        # Articles with no abstract (letters, editorials) are excluded —
        # they have no text to embed meaningfully.
        abstracts_with_text = [a for a in new_abstracts if a.abstract]

        # 4. Single embed call for the entire batch (embed_async chunks internally and releases
        # the lock between chunks so concurrent candidates' query-embeds stay responsive).
        _t_embed = time.perf_counter()
        pairs = await self.embed_abstracts(abstracts_with_text)
        _dt_embed = time.perf_counter() - _t_embed

        # 5. Single bulk insert
        _t_insert = time.perf_counter()
        self.insert_abstracts(pairs, db)
        _dt_insert = time.perf_counter() - _t_insert

        # logger.warning(
        #     "[TIMING] fetch_and_cache breakdown: search=%.1fs fetch_abstracts=%.1fs "
        #     "embed=%.1fs(%d new) insert=%.1fs | %d total pmids, %d stored",
        #     _dt_search,
        #     _dt_fetch,
        #     _dt_embed,
        #     len(abstracts_with_text),
        #     _dt_insert,
        #     len(all_pmids),
        #     len(stored),
        # )

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
            date_before: Optional temporal leak-free cutoff. pgvector is a
                shared cache that may hold abstracts fetched in a prior
                run with no cutoff, so re-apply the same date post-guard here
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
                pmids = await self._filter_pmids_by_date(pmids, date_before, db, client)

        drug_names = await get_all_drug_names(chembl_id, self.cache_dir)
        pref_name = drug_names[0]
        linked_pmids = await self.find_trial_linked_pmids(
            pref_name, disease, date_before
        )
        linked_pmids = await self.cache_trial_reference_abstracts(
            linked_pmids, db, date_before
        )
        query_string = (
            f"Evidence for {pref_name} as a treatment for {disease}, "
            "including clinical trials, efficacy data, mechanism of action, "
            "and preclinical studies"
        )
        _t_embed = time.perf_counter()
        query_vector = (await embed_async([query_string]))[0]
        # logger.warning(
        #     "[TIMING] semantic_search %s embed_query: %.1fs",
        #     disease,
        #     time.perf_counter() - _t_embed,
        # )

        # Over-fetch cap: pull top-N by similarity from pgvector, then rerank
        # by pubtype in Python. Cap gives the boost headroom to reorder
        # (a 2x RCT boost at rank 50 can beat a review at rank 5) without
        # an unbounded scan if the candidate pool grows.
        top_k = _settings.semantic_search_top_k
        rerank_cap = max(top_k * 10, 100)

        # Relevance audit: how many of the fetched PMIDs actually have an embedding
        # row (i.e. truly competed in the rerank), independent of the rerank_cap
        # LIMIT below. Compares against len(pmids) to expose over-fetch — abstracts
        # fetched + embedded but never surfaced in the top-k. Read-only COUNT.
        _embedded_count = db.execute(
            text("SELECT count(*) FROM pubmed_abstracts WHERE pmid = ANY(:pmids)"),
            {"pmids": pmids},
        ).scalar()
        # logger.warning(
        #     "[RELEVANCE] semantic_search %s: %d pmids fetched, %d had embeddings, "
        #     "rerank_cap=%d, top_k kept=%d",
        #     disease,
        #     len(pmids),
        #     _embedded_count or 0,
        #     rerank_cap,
        #     top_k,
        # )

        _t_scan = time.perf_counter()
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
        linked_rows = []
        if linked_pmids:
            linked_rows = db.execute(
                text("""
                    SELECT pmid, title, abstract,
                           1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                    FROM pubmed_abstracts
                    WHERE pmid = ANY(:pmids)
                    ORDER BY similarity DESC
                """),
                {
                    "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
                    "pmids": linked_pmids,
                },
            ).fetchall()
        baseline_pmids = {str(row[0]) for row in rows}
        rows_by_pmid = {str(row[0]): row for row in rows}
        for row in linked_rows:
            rows_by_pmid.setdefault(str(row[0]), row)
        all_rows = list(rows_by_pmid.values())
        # Rows are fully materialized, so the read transaction is no longer
        # needed while PubMed and the literature agent perform awaited work.
        db.rollback()
        # logger.warning(
        #     "[TIMING] semantic_search %s pgvector_scan: %.1fs (%d pmids in)",
        #     disease,
        #     time.perf_counter() - _t_scan,
        #     len(pmids),
        # )

        if not all_rows:
            return []

        candidate_pmids = [row[0] for row in all_rows]
        _t_pt = time.perf_counter()
        async with PubMedClient(cache_dir=self.cache_dir) as client:
            pubtypes_map = await client.fetch_pubtypes(candidate_pmids)
        # logger.warning(
        #     "[TIMING] semantic_search %s fetch_pubtypes: %.1fs (%d candidates)",
        #     disease,
        #     time.perf_counter() - _t_pt,
        #     len(candidate_pmids),
        # )

        if not pubtypes_map:
            logger.warning(
                "semantic_search: fetch_pubtypes returned empty for all %d "
                "candidates (%s / %s); pubtype boost is a no-op for this call",
                len(candidate_pmids),
                chembl_id,
                disease,
            )

        scored: list[tuple[AbstractResult, float, float]] = []
        for row in all_rows:
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
        baseline_scored = [item for item in scored if item[0].pmid in baseline_pmids]

        if linked_pmids:
            trial_reference_set = set(linked_pmids)
            linked = [
                result
                for result, _, _ in scored
                if result.pmid in trial_reference_set
            ]
            drug_identity = await _judge_pmid_drug_identity(
                chembl_id, drug_names, linked, self.cache_dir
            )
            exact_drug = [
                result
                for result in linked
                if drug_identity.get(result.pmid) == "studied"
            ]
            on_topic = await _judge_pmid_treats_disease(
                chembl_id, pref_name, disease, exact_drug, self.cache_dir
            )
            eligible_pmids = {
                result.pmid
                for result in exact_drug
                if on_topic.get(result.pmid, False)
            }
            reserved = [
                item
                for item in scored
                if item[0].pmid in eligible_pmids
            ][:LITERATURE_TRIAL_REFERENCE_RESERVE]
            reserved_pmids = {item[0].pmid for item in reserved}
            if reserved_pmids and not reserved_pmids.intersection(
                item[0].pmid for item in baseline_scored[:top_k]
            ):
                baseline_scored = baseline_scored[: top_k - len(reserved)] + reserved
                baseline_scored.sort(key=lambda item: item[2], reverse=True)

        # logger.info(
        #     "semantic_search rerank top-20 for %s / %s (%d candidates, cap=%d):",
        #     chembl_id,
        #     disease,
        #     len(scored),
        #     rerank_cap,
        # )

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

        return [item[0] for item in baseline_scored[:top_k]]

    async def synthesize(
        self,
        chembl_id: str,
        disease: str,
        top_abstracts: list[AbstractResult],
        approved_indications: list[str] | None = None,
    ) -> EvidenceSummary:
        """Summarize PubMed evidence for a drug-disease pair using an LLM.

        Formats the top abstracts from semantic_search into a prompt, calls the LLM,
        and parses the JSON response into an EvidenceSummary.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            disease: Candidate disease (e.g. "colorectal cancer").
            top_abstracts: Output of semantic_search — list of dicts with keys
                "pmid", "title", "abstract", "similarity".

        Returns:
            EvidenceSummary with all fields populated from the LLM response.
        """
        # Cache key uses sorted PMIDs so two abstract orderings that contain the
        # same evidence collapse to one cache entry. The combined prompt now
        # owns the approved-indication exclusion, so the sorted approved set MUST be
        # in the key — the same PMIDs grade differently under different approved lists.
        approved = sorted(
            {i.strip() for i in (approved_indications or []) if i.strip()}
        )
        cache_params = {
            "chembl_id": chembl_id,
            "disease": disease,
            "pmids": sorted(r.pmid for r in top_abstracts),
            "approved_indications": approved,
            "llm_model": _settings.llm_model,
            # Bump when the relevance prompt or DERIVED fields (direction rollup, strength cap)
            # change, so stale judgments cannot preserve behavior that the new rules reject.
            "logic_version": "per_pmid_class_and_target_gates_v5_per_paper_design",
        }
        cached = cache_get("synthesize", cache_params, self.cache_dir)
        if cached is not None:
            # logger.debug(
            #     "Cache hit for synthesize: %s / %s (%d pmids)",
            #     chembl_id, disease, len(cache_params["pmids"]),
            # )
            return EvidenceSummary(**cached)

        drug_names = await get_all_drug_names(chembl_id, self.cache_dir)
        pref_name = drug_names[0]
        # Two isolated per-paper gates run BEFORE the combined prompt, so the prompt only ever sees
        # abstracts already established as this drug alone, given to treat this disease. Both are
        # cached per paper, so a paper's fate no longer moves when the surrounding batch changes.
        drug_identity = await _judge_pmid_drug_identity(
            chembl_id, drug_names, top_abstracts, self.cache_dir
        )
        on_topic = await _judge_pmid_treats_disease(
            chembl_id,
            pref_name,
            disease,
            [
                result
                for result in top_abstracts
                if drug_identity.get(result.pmid) in ("studied", "class_level")
            ],
            self.cache_dir,
        )
        synthesis_abstracts = [
            result
            for result in top_abstracts
            if drug_identity.get(result.pmid) == "studied"
            and on_topic.get(result.pmid, False)
        ]
        # Class-level papers never enter the supporting/contradicting lists. Their only role is to
        # distinguish "no evidence at all" from "evidence exists, but only for the class".
        class_level_pmids = [
            result.pmid
            for result in top_abstracts
            if drug_identity.get(result.pmid) == "class_level"
            and on_topic.get(result.pmid, False)
        ]
        if not synthesis_abstracts:
            summary = EvidenceSummary(
                summary="",
                study_count=0,
                strength="none",
                direction="none",
                evidence_basis="class_level" if class_level_pmids else "none",
                is_observational=None,
                is_animal_only=None,
                key_findings=[],
                supporting_pmids=[],
                contradicting_pmids=[],
                relevant_pmids=[],
                contaminated_pmids=[result.pmid for result in top_abstracts],
                neutral_pmids=[],
            )
            cache_set(
                "synthesize",
                cache_params,
                summary.model_dump(mode="json"),
                self.cache_dir,
                ttl=CACHE_TTL,
            )
            return summary

        formatted = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract}"
            for r in synthesis_abstracts
        )

        template = (_PROMPTS_DIR / "synthesize.txt").read_text()
        prompt = template.format(
            drug_name=pref_name,
            disease_name=disease,
            abstracts=formatted,
            approved_indications=", ".join(approved) if approved else "(none)",
        )

        # The identifier lists below are rebuilt in code, so they cannot name an abstract that was
        # not supplied. The prose is the exception — the model types those digits itself — so a
        # cited PMID that was never supplied costs a retry, and then the offending sentence.
        allowed_pmids = {r.pmid for r in synthesis_abstracts}
        prose_context = f"{pref_name} / {disease} synthesis"
        data = None
        for attempt in (1, 2):
            response = await query_llm(
                prompt if attempt == 1 else prompt + _PMID_RETRY_NOTE
            )
            # Tolerant parse: the merged prompt is long, so the model sometimes emits prose before
            # the JSON or an empty/overflowed response. parse_last_json_object scans for the last
            # balanced {...} block (same tolerance the retired judge had). On a genuine parse
            # failure, DEGRADE to a safe floor (basis=none, strength none, all abstracts
            # contaminated) rather than `raise` — one bad LLM response must not crash the whole
            # analysis pipeline.
            data = parse_last_json_object(response)
            if data is None:
                break
            bad = unknown_pmids(str(data.get("summary") or ""), allowed_pmids)
            for finding in data.get("key_findings") or []:
                bad.extend(unknown_pmids(str(finding), allowed_pmids))
            if not bad:
                break
            logger.error(
                "synthesize: cited PMID(s) %s not among the abstracts supplied for %s "
                "(attempt %d)",
                ", ".join(bad),
                prose_context,
                attempt,
            )
        else:
            data["summary"] = strip_sentences_with_unknown_pmids(
                str(data.get("summary") or ""), allowed_pmids, context=prose_context
            )
            data["key_findings"] = strip_findings_with_unknown_pmids(
                [str(f) for f in (data.get("key_findings") or [])],
                allowed_pmids,
                context=prose_context,
            )

        if data is None:
            logger.error(
                "synthesize: could not parse a JSON object for %s / %s; returning a safe "
                "untested floor. Response was: %s",
                chembl_id,
                disease,
                response,
            )
            data = {
                "summary": "",
                "study_count": 0,
                "strength": "none",
                "direction": "none",
                "evidence_basis": "none",
                "is_observational": None,
                "is_animal_only": None,
                "key_findings": [],
                "supporting_pmids": [],
                "contradicting_pmids": [],
                "verdicts": {},
            }

        # Per-PMID DIRECTIONAL verdict over the INPUT PMID set. The prompt labels every abstract
        # one of "contaminated"/"supporting"/"contradicting"/"mixed"; any input PMID the model
        # OMITS (or labels unrecognizably) is treated as contaminated (conservative — an
        # unclassified abstract never counts as evidence). The supporting/contradicting/relevant
        # lists are BUILT HERE from this map, NOT emitted by the LLM — this removes the loose
        # second-pass bucketing that mis-placed a positive trial as contradicting (BRAVE-I).
        verdicts = data.get("verdicts")
        input_pmids = [r.pmid for r in top_abstracts]
        synthesis_pmids = {r.pmid for r in synthesis_abstracts}
        # Relevant = anything not contaminated. "neutral" (PK/safety/mechanism, set by the direction
        # sub-call) is relevant — it counts toward study_count — but is excluded from supporting AND
        # contradicting below. synthesize itself never emits "neutral", so including it here only
        # affects post-sub-call membership.
        _RELEVANT_VERDICTS = {"supporting", "contradicting", "mixed", "neutral"}
        if isinstance(verdicts, dict):
            verdict_of = {
                p: (
                    str(verdicts.get(p, "")).strip().lower()
                    if p in synthesis_pmids
                    else "contaminated"
                )
                for p in input_pmids
            }
        else:
            logger.warning(
                "synthesize: no usable 'verdicts' for %s / %s; treating all abstracts as "
                "contaminated. Response was: %s",
                chembl_id,
                disease,
                response,
            )
            verdict_of = {p: "contaminated" for p in input_pmids}

        # AUTHORITATIVE per-PMID DIRECTION via an isolated sub-call over the relevant abstracts.
        # synthesize's verdict map decides relevant-vs-contaminated; this sub-call decides the
        # DIRECTION (supporting/contradicting/mixed) of each relevant abstract. It replaces the old
        # phrase-matching guards, which could not attribute a benefit to the right drug-arm
        # (metformin × hepatic steatosis: comparator's benefit / metabolic-marker improvement read
        # as supporting for metformin). The sub-call's verdict OVERRIDES the synthesize direction
        # for any relevant PMID it returns; PMIDs it omits keep the synthesize direction.
        relevant_for_direction = [
            r for r in synthesis_abstracts if verdict_of[r.pmid] in _RELEVANT_VERDICTS
        ]
        pmid_judgments = await _judge_pmid_directions(
            pref_name, disease, relevant_for_direction
        )
        for p, pmid_judgment in pmid_judgments.items():
            if verdict_of.get(p) != pmid_judgment.verdict:
                logger.info(
                    "synthesize: pmid_direction set PMID %s %s->%s for %s / %s",
                    p,
                    verdict_of.get(p),
                    pmid_judgment.verdict,
                    chembl_id,
                    disease,
                )
                verdict_of[p] = pmid_judgment.verdict

        relevant_pmids = [p for p in input_pmids if verdict_of[p] in _RELEVANT_VERDICTS]
        contaminated_pmids = [
            p for p in input_pmids if verdict_of[p] not in _RELEVANT_VERDICTS
        ]
        # supporting = supporting + mixed; contradicting = contradicting + mixed (a mixed abstract
        # appears in both — it carries evidence in each direction).
        supporting_pmids = [
            p for p in input_pmids if verdict_of[p] in ("supporting", "mixed")
        ]
        contradicting_pmids = [
            p for p in input_pmids if verdict_of[p] in ("contradicting", "mixed")
        ]
        # neutral = relevant but non-efficacy (PK/safety/mechanism): in neither directional list,
        # surfaced separately so a cited-as-context PMID does not look dropped.
        neutral_pmids = [p for p in input_pmids if verdict_of[p] == "neutral"]

        data["relevant_pmids"] = relevant_pmids
        data["contaminated_pmids"] = contaminated_pmids
        data["supporting_pmids"] = supporting_pmids
        data["contradicting_pmids"] = contradicting_pmids
        data["neutral_pmids"] = neutral_pmids
        data["study_count"] = len(relevant_pmids)
        summary = EvidenceSummary(**data)

        # The synthesis model sometimes emits is_observational=False even while describing every relevant human study as
        # uncontrolled. The card renders False as "RCT-backed / controlled", so require a per-abstract controlled-design
        # signal before allowing that claim. Only a paper carrying an EFFICACY verdict may certify the pair: a neutral
        # (PK/safety/mechanism) paper is excluded from the grade, so it must not set the design word either. The judgment
        # is per-abstract because a document-wide phrase match cannot tell a study's own design from one it cites
        # (sildenafil × ischemic stroke: two animal studies graded the pair, and an uncontrolled 12-patient safety study
        # whose background described placebo-controlled RAT experiments supplied the "RCT-backed" claim). The guard is
        # one-way: it withdraws an unsupported controlled claim without upgrading any study to one. With no judgments at
        # all (sub-call failure) there is no evidence either way, so the synthesis value stands.
        directional_judgments = [
            pmid_judgment
            for pmid, pmid_judgment in pmid_judgments.items()
            if verdict_of[pmid] in ("supporting", "contradicting", "mixed")
        ]
        if (
            summary.evidence_basis == "drug_specific"
            and summary.is_observational is False
            and directional_judgments
            and not any(
                pmid_judgment.is_human and pmid_judgment.is_controlled
                for pmid_judgment in directional_judgments
            )
        ):
            summary.is_observational = None

        # Preserve the synthesis model's evidence-weighted overall direction when individual papers
        # disagree. A presence-only rollup made any positive case report cancel controlled negative
        # evidence. The one-sided and no-efficacy cases remain deterministic guardrails.
        has_support = bool(supporting_pmids)
        has_against = any(
            verdict_of[p] in ("contradicting", "mixed") for p in input_pmids
        )
        if not relevant_pmids:
            summary.direction = "none"
        elif has_support and has_against and summary.evidence_basis == "drug_specific":
            directional_abstracts = [
                result
                for result in relevant_for_direction
                if verdict_of[result.pmid] != "neutral"
            ]
            judgment = await _judge_overall_evidence_direction(
                pref_name,
                disease,
                directional_abstracts,
                verdict_of,
            )
            if judgment is not None:
                summary.direction = judgment.direction
                summary.summary = judgment.summary
                summary.key_findings = judgment.key_findings
        elif has_support:
            summary.direction = "supports"
        elif has_against:
            summary.direction = "contradicts"
        else:
            # Every relevant abstract is "neutral" (PK / safety-only / mechanism) — there is no
            # efficacy result in either direction to grade. Without this branch the LLM's own
            # direction survived unchecked: sildenafil x astrocytoma read "weak, supports" on four
            # neutral abstracts and an empty supporting list, and the non-zero study_count then
            # carried it past the supervisor's zero-evidence gate.
            summary.direction = "none"

        # DETERMINISTIC strength cap — the one clinical-accuracy invariant NOT trusted to the
        # prompt. strength/direction grade DRUG-SPECIFIC evidence only; whenever evidence_basis is
        # not "drug_specific" (class_level / approved / none) they MUST be "none", else a
        # class-level RCT body could inflate the card to "strong" while the prose says "no direct
        # evidence for <drug>" (the Parkinson bug). Every consumer (the supervisor ranking path
        # reads es.strength directly) depends on this.
        if summary.evidence_basis != "drug_specific":
            summary.strength = "none"
            summary.direction = "none"

        cache_set(
            "synthesize",
            cache_params,
            summary.model_dump(mode="json"),
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return summary

    async def safety_search(
        self,
        chembl_id: str,
        date_before: date | None = None,
        disease: str | None = None,
    ) -> SafetySearchResult:
        """Fetch adverse-event abstracts while preserving query provenance.

        Delegates to `pubmed_ae.search_adverse_events`, ranked by Europe PMC citation count with
        `date_before` honored by the underlying PubMed search. Fetches TWO pools and dedupes:
          - DRUG-LEVEL ([Majr]) — the drug's drug-wide safety signal.
          - DISEASE-SCOPED (when `disease` is set) — indication-specific safety papers the drug-level
            pool misses (e.g. rofecoxib×colorectal → APPROVe).

        Empty collections mean no adverse-event literature was retrieved.
        """
        pref_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]

        drug_level = await search_adverse_events(
            pref_name, self.cache_dir, date_before=date_before
        )
        disease_scoped: list[PubmedAbstract] = []
        if disease:
            normalized_disease = disease.lower().strip()
            disease_aliases = cache_get(
                "disease_aliases",
                {"chembl_id": chembl_id, "disease": normalized_disease},
                self.cache_dir,
            )
            disease_scoped = await search_adverse_events(
                pref_name,
                self.cache_dir,
                date_before=date_before,
                disease=disease,
                disease_aliases=disease_aliases,
            )

        def _convert(items: list[PubmedAbstract]) -> list[AbstractResult]:
            by_pmid: dict[str, AbstractResult] = {}
            for item in items:
                if not item.pmid or not item.abstract:
                    continue
                by_pmid.setdefault(
                    item.pmid,
                    AbstractResult(
                        pmid=item.pmid,
                        title=item.title,
                        abstract=item.abstract,
                        similarity=0.0,
                    ),
                )
            return list(by_pmid.values())

        return SafetySearchResult(
            drug_level=_convert(drug_level),
            disease_scoped=_convert(disease_scoped),
        )

    async def summarize_safety(
        self,
        chembl_id: str,
        disease: str,
        drug_profile: DrugProfile,
        safety_abstracts: list[AbstractResult],
        date_before: date | None = None,
    ) -> DrugSafetyAssessment:
        """Build a source-separated drug-wide safety assessment.

        Current production reports use exact openFDA boxed-warning text and deterministic
        Open Targets/FAERS metadata. Holdout reports use only date-filtered literature.
        """
        holdout = date_before is not None
        if holdout:
            warnings = []
            top_aes = []
            label_records = []
            label_data_available = None
        else:
            warnings = drug_profile.drug_warnings
            top_aes = sorted(
                [
                    event
                    for event in drug_profile.adverse_events
                    if event.log_likelihood_ratio is not None
                ],
                key=lambda event: event.log_likelihood_ratio,
                reverse=True,
            )[:SAFETY_TOP_ADVERSE_EVENTS]
            try:
                label_records = await self._get_label_safety_records(chembl_id)
                label_data_available = True
            except DataSourceError as exc:
                logger.warning(
                    "summarize_safety: openFDA label safety unavailable for %s: %s",
                    chembl_id,
                    exc,
                )
                label_records = []
                label_data_available = False

        pharmacovigilance_summary = self._format_pharmacovigilance(top_aes)

        cache_params = {
            "chembl_id": chembl_id,
            "disease": disease,
            "logic_version": "source_separated_safety_v2",
            "warnings": sorted(
                f"{w.warning_type}|{w.description or ''}|{w.toxicity_class or ''}"
                for w in warnings
            ),
            "adverse_events": sorted(
                f"{a.name}|{a.count}|{a.log_likelihood_ratio}" for a in top_aes
            ),
            "label_records": sorted(
                f"{record.set_id}|{record.effective_time}|{'|'.join(record.boxed_warnings)}"
                for record in label_records
            ),
            "label_data_available": label_data_available,
            "pmids": sorted(r.pmid for r in safety_abstracts),
            "date_before": date_before.isoformat() if date_before else None,
            "llm_model": _settings.llm_model,
        }
        cached = cache_get("summarize_safety", cache_params, self.cache_dir)
        if cached is not None:
            return DrugSafetyAssessment(**cached)

        regulatory_summary, regulatory_full_labels = (
            await self._format_regulatory_safety(chembl_id, label_records, warnings)
        )

        literature_summary = ""
        safety_pmids: list[str] = []
        safety_severity: str | None
        cacheable = True
        if holdout:
            if safety_abstracts:
                pref_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]
                abstracts_block = "\n\n".join(
                    f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract}"
                    for r in safety_abstracts
                )
                template = (_PROMPTS_DIR / "summarize_safety.txt").read_text()
                prompt = template.format(
                    drug_name=pref_name,
                    disease_name=disease,
                    abstracts=abstracts_block,
                )
                response = await query_llm(prompt)
                data = parse_last_json_object(response)
                if not isinstance(data, dict) or not isinstance(
                    data.get("verdicts"), list
                ):
                    logger.error(
                        "summarize_safety: unparseable holdout response for %s / %s: %s",
                        chembl_id,
                        disease,
                        response,
                    )
                    cacheable = False
                    safety_severity = None
                else:
                    try:
                        verdicts = [
                            SafetyPaperVerdict(**item) for item in data["verdicts"]
                        ]
                    except (TypeError, ValidationError):
                        verdicts = []
                        cacheable = False
                    abstracts_by_pmid = {
                        abstract.pmid: abstract for abstract in safety_abstracts
                    }
                    if {verdict.pmid for verdict in verdicts} != set(
                        abstracts_by_pmid
                    ) or len(verdicts) != len(abstracts_by_pmid):
                        cacheable = False
                    else:
                        confirmed_quotes = []
                        for verdict in verdicts:
                            if verdict.status != "confirmed_harm":
                                continue
                            source = abstracts_by_pmid[verdict.pmid]
                            quote = (verdict.evidence_quote or "").strip()
                            outcome = (verdict.adverse_outcome or "").strip()
                            source_text = (
                                f"{source.title}\n{source.abstract}".casefold()
                            )
                            if (
                                outcome
                                and quote
                                and len(quote.split()) <= 40
                                and quote.casefold() in source_text
                            ):
                                confirmed_quotes.append(quote)
                                safety_pmids.append(verdict.pmid)
                            else:
                                cacheable = False
                        if confirmed_quotes:
                            literature_summary = (
                                "Date-eligible literature reported: "
                                f'"{"; ".join(dict.fromkeys(confirmed_quotes))}" '
                                f"(PMID{'s' if len(safety_pmids) != 1 else ''}: "
                                f"{', '.join(safety_pmids)})."
                            )
                    safety_severity = None
            else:
                safety_severity = None
        else:
            safety_severity = self._ot_warning_severity(
                warnings,
                top_aes,
                has_boxed_warning=any(
                    record.set_id and record.boxed_warnings for record in label_records
                ),
            )

        safety_summary = "\n\n".join(
            section
            for section in (
                regulatory_summary,
                pharmacovigilance_summary,
                literature_summary,
            )
            if section
        )
        if not safety_summary and safety_severity == "none":
            safety_severity = None

        assessment = DrugSafetyAssessment(
            regulatory_summary=regulatory_summary,
            regulatory_full_labels=regulatory_full_labels,
            pharmacovigilance_summary=pharmacovigilance_summary,
            literature_summary=literature_summary,
            safety_summary=safety_summary,
            safety_pmids=safety_pmids,
            safety_severity=safety_severity,
            label_data_available=label_data_available,
        )

        if cacheable:
            cache_set(
                "summarize_safety",
                cache_params,
                assessment.model_dump(mode="json"),
                self.cache_dir,
                ttl=CACHE_TTL,
            )
        return assessment

    async def _judge_one_indication_harm(
        self,
        chembl_id: str,
        pref_name: str,
        disease: str,
        abstract: AbstractResult,
        semaphore: asyncio.Semaphore,
    ) -> SafetyPaperVerdict:
        """Adjudicate ONE disease-scoped abstract in isolation.

        The prompt is unchanged; only the batch size is. Judged alongside nineteen other abstracts,
        a paper reporting a real attributable harm was graded "safety_assessed_only" because the
        surrounding abstracts read as reassuring (sildenafil x heart failure, PMID 25782985: reduced
        left-ventricular contractility versus placebo, confirmed_harm 3/3 alone and
        safety_assessed_only 3/3 in its batch). Every failure mode returns "unclear", which the
        aggregate turns into an unknown harm verdict rather than a "no harm" one.
        """
        cache_params = {
            "chembl_id": chembl_id,
            "disease": disease,
            "pmid": abstract.pmid,
            "logic_version": "per_paper_harm_v1",
            "llm_model": _settings.llm_model,
        }
        cached = cache_get("indication_harm_verdict", cache_params, self.cache_dir)
        if cached is not None:
            return SafetyPaperVerdict(**cached)

        unclear = SafetyPaperVerdict(pmid=abstract.pmid, status="unclear")
        template = (_PROMPTS_DIR / "classify_indication_harm.txt").read_text()
        prompt = template.format(
            drug=pref_name,
            disease=disease,
            abstracts=(
                f"PMID: {abstract.pmid}\nTitle: {abstract.title}\n"
                f"Abstract: {abstract.abstract}"
            ),
        )
        async with semaphore:
            try:
                response = await query_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "classify_indication_harm: LLM call failed for %s / %s / PMID %s: %s",
                    chembl_id,
                    disease,
                    abstract.pmid,
                    exc,
                )
                return unclear

        data = parse_last_json_object(response)
        if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
            logger.warning(
                "classify_indication_harm: unparseable response for %s / %s / PMID %s: %s",
                chembl_id,
                disease,
                abstract.pmid,
                response,
            )
            return unclear

        try:
            verdicts = [SafetyPaperVerdict(**item) for item in data["verdicts"]]
        except (TypeError, ValidationError) as exc:
            logger.warning(
                "classify_indication_harm: invalid verdict for %s / %s / PMID %s: %s",
                chembl_id,
                disease,
                abstract.pmid,
                exc,
            )
            return unclear

        if len(verdicts) != 1 or verdicts[0].pmid != abstract.pmid:
            logger.warning(
                "classify_indication_harm: PMID mismatch for %s / %s / PMID %s; got %s",
                chembl_id,
                disease,
                abstract.pmid,
                [verdict.pmid for verdict in verdicts],
            )
            return unclear

        cache_set(
            "indication_harm_verdict",
            cache_params,
            verdicts[0].model_dump(mode="json"),
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return verdicts[0]

    async def classify_indication_harm(
        self,
        chembl_id: str,
        disease: str,
        safety_abstracts: list[AbstractResult],
    ) -> tuple[bool | None, str, list[str]]:
        """Adjudicate each disease-scoped abstract and aggregate confirmed harms."""
        if not safety_abstracts:
            return None, "", []

        pref_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]
        abstracts_by_pmid = {abstract.pmid: abstract for abstract in safety_abstracts}
        semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)
        verdicts = await asyncio.gather(
            *(
                self._judge_one_indication_harm(
                    chembl_id, pref_name, disease, abstract, semaphore
                )
                for abstract in abstracts_by_pmid.values()
            )
        )

        confirmed: list[SafetyPaperVerdict] = []
        has_unclear = False
        for verdict in verdicts:
            if verdict.status == "unclear":
                has_unclear = True
                continue
            if verdict.status != "confirmed_harm":
                continue
            if verdict.study_subjects != "patients":
                # Non-human work is not patient risk; the model must name what it read before a
                # harm counts, and only "patients" survives.
                logger.info(
                    "classify_indication_harm: dropping non-patient harm PMID %s (%s) for %s / %s",
                    verdict.pmid,
                    verdict.study_subjects,
                    chembl_id,
                    disease,
                )
                continue
            source = abstracts_by_pmid[verdict.pmid]
            quote = (verdict.evidence_quote or "").strip()
            outcome = (verdict.adverse_outcome or "").strip()
            searchable_text = f"{source.title}\n{source.abstract}".casefold()
            if (
                not outcome
                or not quote
                or len(quote.split()) > SAFETY_QUOTE_MAX_WORDS
                or quote.casefold() not in searchable_text
            ):
                has_unclear = True
                continue
            confirmed.append(verdict)

        if confirmed:
            pmids = [verdict.pmid for verdict in confirmed]
            quotes = [
                verdict.evidence_quote.strip()
                for verdict in confirmed
                if verdict.evidence_quote
            ]
            summary = (
                f"Disease-scoped literature for {pref_name} in {disease} reported: "
                f'"{"; ".join(dict.fromkeys(quotes))}" '
                f"(PMID{'s' if len(pmids) != 1 else ''}: {', '.join(pmids)})."
            )
            harm: bool | None = True
        elif has_unclear:
            return None, "", []
        else:
            harm = False
            summary = ""
            pmids = []

        # No aggregate cache entry: the per-paper verdicts are already cached, and a whole-batch key
        # threw away every verdict for a disease as soon as one abstract joined or left the set.
        return harm, summary, pmids

    async def _get_label_safety_records(
        self, chembl_id: str
    ) -> list[FDALabelSafetyRecord]:
        """Fetch label safety once per retrieval service and ChEMBL identifier."""
        task = self._label_safety_tasks.get(chembl_id)
        if task is None:

            async def _fetch() -> list[FDALabelSafetyRecord]:
                drug_names = await get_all_drug_names(chembl_id, self.cache_dir)
                async with FDAClient(cache_dir=self.cache_dir) as client:
                    return await client.get_all_label_safety(drug_names)

            task = asyncio.create_task(_fetch())
            self._label_safety_tasks[chembl_id] = task
        try:
            return await task
        except Exception:
            self._label_safety_tasks.pop(chembl_id, None)
            raise

    @staticmethod
    def _collect_boxed_warnings(
        label_records: list[FDALabelSafetyRecord],
    ) -> dict[str, tuple[str, str, list[str]]]:
        """One entry per approved product's openFDA label record: (text, effective_time, names)."""
        boxed_by_set: dict[str, tuple[str, str, list[str]]] = {}
        for record in label_records:
            if not record.set_id:
                continue
            text = next((t.strip() for t in record.boxed_warnings if t.strip()), None)
            if text is None:
                continue
            boxed_by_set[record.set_id] = (
                text,
                record.effective_time or "",
                record.brand_names or record.generic_names or ["unnamed product"],
            )
        return boxed_by_set

    @staticmethod
    def _format_boxed_warnings_appendix(
        boxed_by_set: dict[str, tuple[str, str, list[str]]],
    ) -> str:
        """Full verbatim boxed-warning text per product, newest first — the source of record
        for the LLM-summarized digest shown in the Drug Safety section."""
        if not boxed_by_set:
            return ""
        ordered = sorted(
            boxed_by_set.values(), key=lambda entry: entry[1], reverse=True
        )
        blocks = []
        for warning_text, effective_time, names in ordered:
            header = f"{names[0]} (effective {effective_time or 'unknown'})"
            blocks.append(f"{header}:\n{warning_text}")
        return "\n\n".join(blocks)

    async def _summarize_boxed_warnings(
        self, chembl_id: str, boxed_by_set: dict[str, tuple[str, str, list[str]]]
    ) -> str:
        """LLM digest of every distinct boxed-warning text, calling out any label that differs
        in substance rather than blending it away. The verbatim texts remain available via
        `_format_boxed_warnings_appendix` — this is a readability aid, not the source of record.

        Only resolves a drug display name (a network/cache call) and calls the LLM when there
        is more than one distinct text to reconcile — the common single-label-text case stays
        fully deterministic and network-free.
        """
        if not boxed_by_set:
            return ""

        by_text: dict[str, list[str]] = {}
        for warning_text, _effective_time, names in boxed_by_set.values():
            by_text.setdefault(warning_text, []).append(names[0])
        distinct_texts = list(by_text.items())

        if len(distinct_texts) == 1:
            warning_text, _names = distinct_texts[0]
            n = len(boxed_by_set)
            agreement = (
                f"all {n} FDA-approved product labels"
                if n > 1
                else "the FDA-approved product label"
            )
            return (
                f"FDA label boxed-warning text ({agreement} agree verbatim):\n"
                f"{warning_text}"
            )

        cache_params = {
            "chembl_id": chembl_id,
            "logic_version": "summarize_boxed_warnings_v1",
            "distinct_texts": sorted(t for t, _names in distinct_texts),
            "llm_model": _settings.llm_model,
        }
        cached = cache_get("summarize_boxed_warnings", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        drug_name = (await get_all_drug_names(chembl_id, self.cache_dir))[0]
        label_blocks = "\n\n".join(
            f"Product(s): {', '.join(names)}\n{t}" for t, names in distinct_texts
        )
        template = (_PROMPTS_DIR / "summarize_boxed_warnings.txt").read_text()
        prompt = template.format(
            n=len(boxed_by_set),
            drug_name=drug_name,
            label_blocks=label_blocks,
        )
        response = await query_llm(prompt)
        summary = response.strip()

        result = f"FDA label boxed-warning text ({len(boxed_by_set)} product labels, LLM-summarized — see appendix for verbatim text):\n{summary}"
        cache_set(
            "summarize_boxed_warnings",
            cache_params,
            result,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return result

    async def _format_regulatory_safety(
        self,
        chembl_id: str,
        label_records: list[FDALabelSafetyRecord],
        warnings: list[DrugWarning],
    ) -> tuple[str, str]:
        """Format label text and Open Targets warning metadata without conflating them.

        Returns (regulatory_summary, regulatory_full_labels): the summary is an LLM digest of
        every distinct boxed-warning text (or the text itself, when every label agrees
        verbatim); regulatory_full_labels is the complete verbatim list for the report's
        appendix, so nothing from the source labels is lost to the summarization step.
        """
        boxed_by_set = self._collect_boxed_warnings(label_records)
        full_labels = self._format_boxed_warnings_appendix(boxed_by_set)

        sections: list[str] = []
        boxed_summary = await self._summarize_boxed_warnings(chembl_id, boxed_by_set)
        if boxed_summary:
            sections.append(boxed_summary)

        warning_types = sorted(
            {
                warning.warning_type.strip()
                for warning in warnings
                if warning.warning_type
            }
        )
        toxicity_classes = sorted(
            {
                warning.toxicity_class.strip()
                for warning in warnings
                if warning.toxicity_class
            }
        )
        if warning_types or toxicity_classes:
            metadata_parts = []
            if warning_types:
                metadata_parts.append(f"warning type: {', '.join(warning_types)}")
            if toxicity_classes:
                metadata_parts.append(
                    f"toxicity categories: {', '.join(toxicity_classes)}"
                )
            sections.append(
                "Open Targets warning metadata: " + "; ".join(metadata_parts) + "."
            )
        return "\n\n".join(sections), full_labels

    @staticmethod
    def _format_pharmacovigilance(top_aes: list[AdverseEvent]) -> str:
        """Format scored Open Targets/FAERS signals without causal language or zero fills."""
        if not top_aes:
            return ""
        terms = []
        for event in top_aes:
            details = [f"logLR {event.log_likelihood_ratio:.1f}"]
            if event.count is not None:
                details.append(f"{event.count} reports")
            terms.append(f"{event.name} ({'; '.join(details)})")
        return (
            "Open Targets/FAERS pharmacovigilance signals: "
            + ", ".join(terms)
            + ". These are reporting associations, not proof of causation."
        )

    @staticmethod
    def _ot_warning_severity(
        warnings: list[DrugWarning],
        top_aes: list[AdverseEvent],
        *,
        has_boxed_warning: bool,
    ) -> str:
        """Deterministic production safety severity from OT warning_type. Withdrawn outranks Black
        Box Warning; an OT adverse-event signal with no formal warning → 'serious'; else 'none'.
        """
        types = {(w.warning_type or "").strip().lower() for w in warnings}
        if "withdrawn" in types:
            return "withdrawn"
        if has_boxed_warning or "black box warning" in types:
            return "black_box"
        if warnings or top_aes:
            return "serious"
        return "none"

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
            "logic_version": "deterministic_direct_v1",
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

        # Build the direct query deterministically. Supplemental queries remain LLM-generated.
        quoted_disease = f'"{disease_term}"'
        direct_query = f"{pref_name} AND {quoted_disease}"

        # Substitute the <DISEASE> placeholder with the quoted MeSH preferred term.
        # The prompt instructs the LLM to emit `<DISEASE>` instead of writing the
        # disease name directly, so the JSON array never contains embedded quotes.
        substituted = [q.replace("<DISEASE>", quoted_disease) for q in raw]

        # Case-normalised dedup: direct query stays first and model duplicates are removed.
        seen: dict[str, str] = {}
        for term in [direct_query, *substituted]:
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
