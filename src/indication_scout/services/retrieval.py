"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import asyncio
import calendar
import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from indication_scout.config import get_settings
from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    CACHE_TTL,
    LITERATURE_DIRECTION_FLOOR,
    LITERATURE_EXTRA_PASS_TAGS,
    LITERATURE_RECENT_BUDGET,
    LITERATURE_RECENT_YEARS,
    LITERATURE_RERANK_HEADROOM,
    LITERATURE_SCREEN_BATCH,
    LITERATURE_SCREEN_MAX_DEPTH,
    LITERATURE_SCREEN_TARGET_PASSERS,
    LITERATURE_SHALLOW_DEPTH,
    LITERATURE_SYNTHESIS_MAX,
    LITERATURE_TRIAL_REFERENCE_RESERVE,
    OPEN_TARGETS_COMPETITOR_LOGIC_VERSION,
    PUBMED_NCT_QUERY_BATCH_SIZE,
    SYNTHESIS_MAX_TOKENS,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient, get_all_drug_names
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
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
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.services.citation_guard import (
    strip_findings_with_unknown_pmids,
    strip_sentences_with_unknown_pmids,
    unknown_pmids,
)
from indication_scout.services.disease_helper import (
    merge_duplicate_diseases,
    resolve_mesh_id,
)
from indication_scout.services.embeddings import embed_async
from indication_scout.services.llm import (
    parse_last_json_object,
    parse_llm_response,
    query_llm,
    query_small_llm,
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


# Pubtype multiplicative boosts on semantic similarity, surfacing primary clinical evidence (RCTs, phase trials) over
# reviews/commentary. Unlisted types get 1.0; a record's boost is the max across its pubtype list.
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


_PMID_GRADE_PROMPT = (_PROMPTS_DIR / "pmid_grade.txt").read_text()
_EVIDENCE_DIRECTION_PROMPT = (_PROMPTS_DIR / "judge_evidence_direction.txt").read_text()
_PMID_DRUG_IDENTITY_PROMPT = (_PROMPTS_DIR / "pmid_drug_identity.txt").read_text()
# "class_level" = the abstract's result belongs to the drug's mechanistic class, not the drug alone (a pooled
# sildenafil+tadalafil meta-analysis). It is judged per paper because evidence_basis is one verdict per candidate:
# otherwise one drug-specific animal study made a candidate "drug_specific" and class-level papers counted at full
# weight (sildenafil x ischemic stroke read "moderate, supports" on rodent data plus two class-level papers).
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


def _names_in_text(drug_names: list[str], title: str, abstract: str | None) -> bool:
    """Return whether the title or abstract contains a supplied name for the exact drug.

    Whole-token match after lowercasing and splitting on non-alphanumerics, so "bupropion" does not match
    "bupropion-SR" differently from "bupropion SR", and a different drug sharing a prefix does not match.
    This is the one name rule: the pre-ranking pool filter and the drug-identity gate both call it, so a paper
    the filter drops is exactly a paper the gate would have rejected without a model call.
    """

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

    text = f" {normalize(f'{title} {abstract or ''}')} "
    return any(
        normalized_name and f" {normalized_name} " in text
        for name in drug_names
        if (normalized_name := normalize(name))
    )


def pub_year(pub_date: str | None) -> int | None:
    """Year of a `pubmed_abstracts.pub_date` ("2021", "2021-Nov", "2021-Nov-03", "2021-11", "2021-11-03"); None when the
    value does not start with a four-digit year."""
    match = re.match(r"^(\d{4})(?:-|$)", pub_date or "")
    return int(match.group(1)) if match else None


def _recent_pass(
    ranked: list["AbstractResult"],
    pub_years: dict[str, int],
    reference_year: int,
    exclude: set[str],
) -> list["AbstractResult"]:
    """The first LITERATURE_RECENT_BUDGET papers of `ranked` (relevance order) published in or after
    `reference_year - LITERATURE_RECENT_YEARS` and not in `exclude`. An undated paper is skipped here only.
    """
    start_year = reference_year - LITERATURE_RECENT_YEARS
    return [
        result
        for result in ranked
        if result.pmid not in exclude
        and (year := pub_years.get(result.pmid)) is not None
        and year >= start_year
    ][:LITERATURE_RECENT_BUDGET]


def _union_tagged(
    window: list["AbstractResult"], extra: list["AbstractResult"], tag: str
) -> list["AbstractResult"]:
    """`window` followed by the papers of `extra` it lacks, one copy per PMID. A paper in both keeps its window position
    and gains `tag` in `selected_by`."""
    in_window = {result.pmid for result in window}
    extra_pmids = {result.pmid for result in extra}
    merged = [
        (
            result.model_copy(update={"selected_by": [*result.selected_by, tag]})
            if result.pmid in extra_pmids and tag not in result.selected_by
            else result
        )
        for result in window
    ]
    merged += [
        result.model_copy(update={"selected_by": [tag]})
        for result in extra
        if result.pmid not in in_window
    ]
    return merged


# "NCT 02062190" is matched too: 24 of about 3,500 NCT-citing abstracts in the local store write the space.
_NCT_PATTERN = re.compile(r"\bNCT\s?(\d{8})\b")


def nct_ids(result: "AbstractResult") -> frozenset[str]:
    """ClinicalTrials.gov numbers cited in the title or abstract, normalized to NCT plus eight digits."""
    return frozenset(
        f"NCT{digits}"
        for digits in _NCT_PATTERN.findall(f"{result.title} {result.abstract}")
    )


def _graded_rank_key(
    result: "AbstractResult", grades: dict[str, PmidJudgment]
) -> tuple[int, float, bool, int, str]:
    """Sort key: graded relevance, then embedding similarity, then the more recent paper (undated last), then PMID."""
    return (
        -grades[result.pmid].relevance,
        -result.similarity,
        result.pub_year is None,
        -(result.pub_year or 0),
        result.pmid,
    )


def merge_same_trial(
    passers: list["AbstractResult"], grades: dict[str, PmidJudgment]
) -> tuple[list["AbstractResult"], dict[str, list[str]]]:
    """Collapse papers that report one registered trial in the same graded direction into one study.

    Papers are grouped when they cite exactly the same set of NCT numbers and carry the same graded verdict. The paper
    with the highest graded relevance leads the group and is kept; the others are returned under its PMID as secondary
    reports. Grouping on any shared NCT number would chain a pooled analysis of trials A and B to the primary reports of
    both, merging two independent trials (431 stored abstracts cite more than one NCT number). Papers without an NCT
    number, and papers on one trial with different verdicts, are never merged. Every paper in `passers` must be graded.

    Returns:
        The kept papers in input order, and a map from each lead PMID to its secondary report PMIDs.
    """
    groups: dict[tuple[frozenset[str], str], list[AbstractResult]] = {}
    for result in passers:
        ids = nct_ids(result)
        if ids:
            groups.setdefault((ids, grades[result.pmid].verdict), []).append(result)
    secondary: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda result: _graded_rank_key(result, grades))
        secondary[ranked[0].pmid] = [result.pmid for result in ranked[1:]]
    merged_away = {pmid for pmids in secondary.values() for pmid in pmids}
    return [result for result in passers if result.pmid not in merged_away], secondary


def _direction_slots(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Split `cap` slots across directions in proportion to `counts` (every count > 0, total > cap), with a floor of
    LITERATURE_DIRECTION_FLOOR (or the direction's count, if smaller) per direction. Rounding remainders go to the
    largest directions."""
    total = sum(counts.values())
    floors = {d: min(LITERATURE_DIRECTION_FLOOR, n) for d, n in counts.items()}
    if sum(floors.values()) > cap:
        raise ValueError(f"synthesis cap {cap} is below the direction floors {floors}")
    slots = {d: min(n, max(floors[d], cap * n // total)) for d, n in counts.items()}
    # Floors can push the total over the cap: take the excess from the directions with the most slots above their floor.
    while sum(slots.values()) > cap:
        largest = max(
            (d for d in slots if slots[d] > floors[d]), key=lambda d: (slots[d], d)
        )
        slots[largest] -= 1
    # Rounding down leaves slots unused: give them to the largest directions that still have unread papers.
    while sum(slots.values()) < cap:
        largest = max(
            (d for d in slots if slots[d] < counts[d]), key=lambda d: (counts[d], d)
        )
        slots[largest] += 1
    return slots


def select_synthesis_set(
    passers: list["AbstractResult"], grades: dict[str, PmidJudgment], cap: int
) -> list["AbstractResult"]:
    """Pick the papers whose abstracts synthesis reads, at most `cap`, from the graded, same-trial-merged passers.

    When everything fits, everything is sent. Otherwise slots are split across graded verdicts in proportion to their
    counts (see _direction_slots), so the sample keeps the shape of the whole set. Within a verdict, controlled human
    studies come first, then the best paper from each kept extra pass (LITERATURE_EXTRA_PASS_TAGS), then the rest; each
    tier is ordered by graded relevance, with similarity and recency breaking ties. Returns papers in input order.
    """
    if len(passers) <= cap:
        return list(passers)
    by_verdict: dict[str, list[AbstractResult]] = {}
    for result in passers:
        by_verdict.setdefault(grades[result.pmid].verdict, []).append(result)
    slots = _direction_slots(
        {verdict: len(members) for verdict, members in by_verdict.items()}, cap
    )
    chosen: set[str] = set()
    for verdict, members in by_verdict.items():
        ranked = sorted(members, key=lambda result: _graded_rank_key(result, grades))
        ordered = [
            result.pmid
            for result in ranked
            if grades[result.pmid].is_human and grades[result.pmid].is_controlled
        ]
        for tag in LITERATURE_EXTRA_PASS_TAGS:
            best = next(
                (
                    result.pmid
                    for result in ranked
                    if tag in result.selected_by and result.pmid not in ordered
                ),
                None,
            )
            if best is not None:
                ordered.append(best)
        ordered += [result.pmid for result in ranked if result.pmid not in ordered]
        chosen.update(ordered[: slots[verdict]])
    return [result for result in passers if result.pmid in chosen]


def _design_label(grade: PmidJudgment) -> str:
    if not grade.is_human:
        return "non-human (animal or in vitro)"
    return "controlled human" if grade.is_controlled else "uncontrolled human"


def _screened_counts_text(
    merged: list["AbstractResult"], grades: dict[str, PmidJudgment], selected: int
) -> str:
    """The direction and design tally of every graded, merged paper, for the synthesis prompt."""
    directions: dict[str, int] = {}
    designs: dict[str, int] = {}
    for result in merged:
        grade = grades[result.pmid]
        directions[grade.verdict] = directions.get(grade.verdict, 0) + 1
        designs[_design_label(grade)] = designs.get(_design_label(grade), 0) + 1
    if selected == len(merged):
        scope = f"The {selected} abstracts above are every paper that passed screening."
    else:
        scope = (
            f"The {selected} abstracts above are a sample of the {len(merged)} papers that passed screening. The counts "
            f"below describe all {len(merged)}; judge strength and direction from them as well as from the abstracts."
        )
    direction_text = ", ".join(
        f"{verdict} {n}" for verdict, n in sorted(directions.items())
    )
    design_text = ", ".join(f"{design} {n}" for design, n in sorted(designs.items()))
    return f"{scope}\nGraded direction counts: {direction_text}\nStudy design counts: {design_text}"


def _mentions_exact_drug(drug_names: list[str], result: "AbstractResult") -> bool:
    """Return whether the title or abstract contains a supplied name for the exact drug."""
    return _names_in_text(drug_names, result.title, result.abstract)


async def _judge_pmid_drug_identity(
    chembl_id: str,
    drug_names: list[str],
    abstracts: list["AbstractResult"],
    cache_dir: Path,
) -> dict[str, str]:
    """Return each PMID's drug-role verdict: "studied", "class_level" or "not_studied".

    Abstracts mentioning none of the accepted drug names are rejected deterministically. The rest are judged in isolated
    calls cached by ChEMBL ID and PMID, so the surrounding batch cannot affect a decision; the question is
    disease-independent, so one entry serves every candidate. Missing or invalid responses fail closed as "not_studied"
    and are not cached.
    """
    if not abstracts:
        return {}

    names = list(dict.fromkeys(name.strip() for name in drug_names if name.strip()))
    formatted_names = "\n".join(f"- {name}" for name in names)
    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(result: "AbstractResult") -> tuple[str, str]:
        if not _mentions_exact_drug(names, result):
            # Logged because this branch excludes a paper without an LLM call; a name the corpus uses but ChEMBL lacks
            # would otherwise drop real evidence with no trace.
            logger.debug(
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

    Isolated one-question-per-abstract calls, cached by ChEMBL ID, PMID and disease. This is the therapeutic-target gate:
    left in the combined synthesize prompt, a related condition could grade as the candidate (sildenafil for neonatal
    hypoxic-ischemic brain injury read as evidence for ischemic stroke). Missing or invalid responses fail closed as
    ``False`` and are not cached.
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
            "logic_version": "treats_disease_v6_genetic_proxy",
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


async def _judge_pmid_grades(
    chembl_id: str,
    drug: str,
    disease: str,
    abstracts: list["AbstractResult"],
    cache_dir: Path,
) -> dict[str, PmidJudgment]:
    """Grade each abstract in an isolated call: efficacy direction (supporting | contradicting | mixed | neutral), the
    design of the study it reports (human, controlled), and a 1-10 relevance score. This is the AUTHORITATIVE direction
    and design for each paper. One call per paper, cached by ChEMBL ID, PMID and disease, so a paper's grade cannot move
    with the papers around it: graded in one batch, about one label in ten changed when the batch was split in two, and
    the batch overused "mixed" for failed trials and inconclusive reviews. A missing or unusable response leaves the PMID
    out of the result and is not cached; the caller drops that paper.
    """
    if not abstracts:
        return {}

    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def grade_one(result: "AbstractResult") -> tuple[str, PmidJudgment | None]:
        cache_params = {
            "chembl_id": chembl_id,
            "pmid": result.pmid,
            "disease": disease,
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "pmid_grade_v2_interventional_pilot",
        }
        cached = cache_get("pmid_grade", cache_params, cache_dir)
        if isinstance(cached, dict):
            try:
                return result.pmid, PmidJudgment(**cached)
            except ValidationError:
                logger.warning(
                    "pmid_grade: discarding invalid cached grade for PMID %s",
                    result.pmid,
                )

        prompt = _PMID_GRADE_PROMPT.format(
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
                    "pmid_grade: LLM call failed for %s / %s / PMID %s; dropping paper: %s",
                    chembl_id,
                    disease,
                    result.pmid,
                    exc,
                )
                return result.pmid, None
        data = parse_last_json_object(response)
        try:
            judgment = PmidJudgment(**data) if isinstance(data, dict) else None
        except ValidationError:
            judgment = None
        if judgment is None:
            logger.warning(
                "pmid_grade: unusable grade for %s / %s / PMID %s; dropping paper. Response was: %s",
                chembl_id,
                disease,
                result.pmid,
                response,
            )
            return result.pmid, None

        cache_set(
            "pmid_grade",
            cache_params,
            judgment.model_dump(mode="json"),
            cache_dir,
            ttl=CACHE_TTL,
        )
        return result.pmid, judgment

    graded = await asyncio.gather(*(grade_one(result) for result in abstracts))
    return {pmid: judgment for pmid, judgment in graded if judgment is not None}


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

    # Both attempts produced a validated judgment (the None paths return above), so it is set here.
    judgment = cast(EvidenceDirectionJudgment, judgment)
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
    # What put the paper in the screening window: "relevance" (the adaptive relevance cut), "trial_reference" (a
    # gate-passing publication linked from a ClinicalTrials.gov record), "recent" (the recent pass). Empty outside
    # semantic_search.
    selected_by: list[str] = []
    # Publication year from the stored pub_date; None when undated or outside semantic_search.
    pub_year: int | None = None


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

        Fetches raw competitor data, then uses an LLM to merge duplicate disease names and remove overly broad terms.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            date_before: Optional temporal holdout cutoff, forwarded to the OT client to suppress its current-state
                approved-indications strip. The cache key includes the cutoff.

        Returns:
            Dict mapping disease name to set of competitor drug names.
        """
        cache_params = {
            "chembl_id": chembl_id,
            "date_before": date_before.isoformat() if date_before else None,
            "top_k": _settings.literature_top_k,
            # The merge runs over the prefetched disease list, so a change to the prefetch size changes this result.
            "prefetch_max": _settings.open_targets_competitor_prefetch_max,
            "logic_version": "cache_disease_aliases_v2_opus_merge",
            # Built from the raw competitor list, so it goes stale whenever that list's selection logic changes.
            "competitor_logic_version": OPEN_TARGETS_COMPETITOR_LOGIC_VERSION,
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
            # Read back by DrugSafetyService.safety_search (services/drug_safety.py); keep the namespace and key shape identical.
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
        """Fetch drug + target data from Open Targets, enrich with ATC descriptions from ChEMBL, and return a DrugProfile.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").

        Returns:
            DrugProfile with all fields populated; atc_descriptions is [] if the drug has no ATC classifications.
        """
        async with OpenTargetsClient(cache_dir=self.cache_dir) as open_targets_client:
            rich = await open_targets_client.get_rich_drug_data(chembl_id)

        atc_descriptions = []
        if rich.drug is not None and rich.drug.atc_classifications:
            async with ChEMBLClient() as chembl_client:
                for code in rich.drug.atc_classifications:
                    atc_descriptions.append(
                        await chembl_client.get_atc_description(code)
                    )

        return DrugProfile.from_rich_drug_data(rich, atc_descriptions)

    def get_stored_pmids(self, pmids: list[str], db: Session) -> set[str]:
        """Return the subset of the given PMIDs that already exist in pubmed_abstracts.

        Used by fetch_and_cache to avoid re-fetching and re-embedding stored abstracts, via one bulk SELECT.

        Args:
            pmids: Candidate PMIDs to check.
            db: Active SQLAlchemy session.

        Returns:
            Set of PMIDs present in the pubmed_abstracts table.
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

        Calls client.fetch_abstracts on the set difference of all_pmids and stored_pmids, skipping the network call when
        there are no new PMIDs.

        Args:
            all_pmids: Full list of PMIDs from a PubMed search result.
            stored_pmids: PMIDs already present in pubmed_abstracts (from get_stored_pmids).
            client: Open PubMedClient session to reuse (owned by the caller).

        Returns:
            PubmedAbstract objects for the newly fetched PMIDs; empty if all were already stored.
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
            new_abstracts = await self.fetch_new_abstracts(linked_pmids, stored, client)
        abstracts_with_text = [
            abstract for abstract in new_abstracts if abstract.abstract
        ]
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

        Embeds "<title>. <abstract>" for each abstract in a single embed_async() batch; empty input skips embed().

        Args:
            abstracts: Abstracts to embed.

        Returns:
            List of (PubmedAbstract, embedding vector) pairs in input order.
        """
        if not abstracts:
            return []

        emit_progress(PHASE_LITERATURE, f"Embedding {len(abstracts)} new abstracts")
        texts = [f"{a.title}. {a.abstract or ''}" for a in abstracts]
        vectors = await embed_async(texts)
        return list(zip(abstracts, vectors, strict=True))

    def insert_abstracts(
        self,
        pairs: list[tuple[PubmedAbstract, list[float]]],
        db: Session,
    ) -> None:
        """Bulk-insert (abstract, embedding) pairs into pubmed_abstracts.

        Uses INSERT ... ON CONFLICT DO NOTHING, so re-running with stored PMIDs is idempotent. Does nothing when pairs
        is empty.

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

        Formats from _parse_pubmed_xml: "2023", "2023-Mar", "2023-03", "2023-03-15". Partial dates use the LAST day of
        the range so the cutoff comparison errs toward "later" ("2023-Mar" becomes 2023-03-31: excluded by a 2023-04-01
        cutoff but not a 2023-04-30 one). Returns None for missing or unparseable input; the caller decides what to do.
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

        Only PMIDs found in the DB are returned; values may be None (nullable column).
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

        Reads pub_date from pgvector for stored PMIDs (no HTTP) and falls back to esummary for unknowns via
        client._filter_pmids_by_date. Missing or unparseable dates are KEPT, as in the client.
        """
        if not pmids:
            return []

        known = self._read_pub_dates_from_db(pmids, db)
        # The SELECT starts a transaction; end it before the PubMed request below so the connection returns to the pool.
        db.rollback()
        from_db_kept: list[str] = []
        unknown: list[str] = []
        for pmid in pmids:
            if pmid not in known:
                unknown.append(pmid)
                continue
            parsed = self._parse_pub_date_conservative(known[pmid])
            if parsed is None:
                # Stored row but no usable date: keep the PMID, matching the production policy.
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

        Args:
            queries: PubMed keyword queries (e.g. from expand_search_terms).
            db: Active SQLAlchemy session.
            date_before: Optional temporal holdout cutoff; only articles published before it are returned.
            direct_query: Deterministic drug-disease query to retrieve completely; other queries keep the configured
                relevance-result cap.

        Returns:
            Deduplicated list of PMIDs across all queries. Not every PMID has a row in pubmed_abstracts: articles
            without an abstract (letters, editorials) are excluded, and semantic_search skips them via
            WHERE pmid = ANY(:pmids).
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

            # Per-query attribution: how many PMIDs each query returned (PMIDs omitted to keep logs readable).
            for _q, _pmids in zip(queries, search_results, strict=True):
                # logger.warning("[QUERYMAP] query=%r returned %d pmids", _q, len(_pmids))
                pass

            # 1.5 Cutoff post-guard: PubMed's maxdate filter is not strictly respected, so re-verify each PMID's date from
            # pgvector for stored PMIDs and esummary only for unknowns (cuts NCBI traffic on re-runs).
            if date_before is not None:
                all_pmids = await self._filter_pmids_by_date(
                    all_pmids, date_before, db, client
                )

            # 2. Single bulk check against pgvector
            stored = self.get_stored_pmids(all_pmids, db)
            # Release the read transaction before fetching/embedding; the later insert starts and commits its own.
            db.rollback()

            # 3. Single fetch for all new abstracts
            _t_fetch = time.perf_counter()
            new_abstracts = await self.fetch_new_abstracts(all_pmids, stored, client)
            _dt_fetch = time.perf_counter() - _t_fetch

        # Articles with no abstract (letters, editorials) are excluded: nothing to embed.
        abstracts_with_text = [a for a in new_abstracts if a.abstract]

        # 4. Single embed call for the batch (embed_async chunks internally and releases the lock between chunks so
        # concurrent candidates' query-embeds stay responsive).
        _t_embed = time.perf_counter()
        pairs = await self.embed_abstracts(abstracts_with_text)
        _dt_embed = time.perf_counter() - _t_embed

        # 5. Single bulk insert
        _t_insert = time.perf_counter()
        self.insert_abstracts(pairs, db)
        _dt_insert = time.perf_counter() - _t_insert

        logger.info(
            "[LIT] fetch_and_cache: %d pmids, %d already stored, %d new abstracts embedded",
            len(all_pmids),
            len(stored),
            len(abstracts_with_text),
        )

        return all_pmids

    async def semantic_search(
        self,
        disease: str,
        chembl_id: str,
        pmids: list[str],
        db: Session,
        date_before: date | None = None,
        depth: int | None = None,
        target_passers: int = LITERATURE_SCREEN_TARGET_PASSERS,
        adaptive: bool | None = None,
        recent_pass: bool | None = None,
    ) -> list[AbstractResult]:
        """For a given drug, disease, and list of PMIDs, return the screening window of drug-named abstracts.

        Builds a natural-language query from drug and disease (e.g. "Evidence for metformin as a treatment for colorectal
        cancer..."), embeds it with BioLORD-2023, and runs a cosine similarity search restricted to the given PMIDs. Papers
        whose title and abstract name none of the drug's accepted names are dropped BEFORE ranking: the drug-identity gate
        rejects them without a model call anyway, so left in they only displace readable papers from the shortlist (for
        bupropion they are over 80% of most pools).

        The window is read in relevance order, LITERATURE_SCREEN_BATCH papers at a time, through the exact-drug and
        disease-treatment gates, and stops at the first batch boundary where `target_passers` papers have passed both, or
        at `depth`. Both gates are cached per paper, so `synthesize` re-running them over the window costs nothing.

        Args:
            disease: e.g. "colorectal cancer"
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            pmids: e.g. ["29734553", "31245678", "30198432"]
            date_before: Optional leak-free cutoff. pgvector is a shared cache that may hold abstracts fetched without a
                cutoff, so the date post-guard is re-applied before re-ranking.
            depth: The most papers the window can hold. None takes LITERATURE_SCREEN_MAX_DEPTH when the window is
                adaptive and LITERATURE_SHALLOW_DEPTH when it is not.
            target_passers: With an adaptive window, stop reading once this many papers have passed both gates.
            adaptive: Screen the window through the gates and stop at `target_passers`. False reads a fixed `depth`
                papers without running the gates (the depth sweep in scripts/validation/compare_literature_shortlist.py
                uses that). None takes the literature_deep_screening setting.
            recent_pass: Append the recent pass. None takes the literature_recent_pass setting.

        After the window, the recent pass appends up to LITERATURE_RECENT_BUDGET rerank candidates published in
        the last LITERATURE_RECENT_YEARS that the window missed.

        Returns:
            The window ranked by descending similarity x publication-type boost, then the recent-pass papers in the same
            order. Each paper's `selected_by` says which step selected it.
        """
        # Holdout post-guard: drop PMIDs published on/after the cutoff that leaked in via the shared pgvector cache, using
        # the same filter as fetch_and_cache.
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

        # Over-fetch cap: keep this many drug-named papers by similarity, then rerank by pubtype in Python. The cap gives
        # the boost headroom to reorder (a 2x RCT boost at rank 50 can beat a review at rank 5) without fetching pubtypes
        # for the whole pool. It is applied AFTER the drug-name filter, so it never truncates below `depth` on its own.
        if adaptive is None:
            adaptive = _settings.literature_deep_screening
        if recent_pass is None:
            recent_pass = _settings.literature_recent_pass
        if depth is None:
            depth = (
                LITERATURE_SCREEN_MAX_DEPTH if adaptive else LITERATURE_SHALLOW_DEPTH
            )
        top_k = depth
        rerank_cap = max(top_k * LITERATURE_RERANK_HEADROOM, 100)

        # Relevance audit: how many fetched PMIDs have an embedding row (competed in the rerank), independent of the
        # rerank_cap LIMIT below; compared with len(pmids) it exposes abstracts embedded but never surfaced. Read-only COUNT.
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
        # No LIMIT here: the drug-name filter runs on the returned rows, so the cap has to apply after it or a pool that
        # is mostly other-drug papers would be cut before the drug's own papers are reached.
        pool_rows = db.execute(
            text("""
                SELECT pmid, title, abstract,
                       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity,
                       pub_date
                FROM pubmed_abstracts
                WHERE pmid = ANY(:pmids)
                ORDER BY similarity DESC
            """),
            {
                "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
                "pmids": pmids,
            },
        ).fetchall()
        named_rows = [
            row for row in pool_rows if _names_in_text(drug_names, row[1], row[2])
        ]
        rows = named_rows[:rerank_cap]
        logger.info(
            "[LIT] drug-name filter %s / %s: %d stored -> %d name the drug -> %d kept for rerank",
            chembl_id,
            disease,
            len(pool_rows),
            len(named_rows),
            len(rows),
        )
        baseline_pmids = {str(row[0]) for row in rows}
        rows_by_pmid = {str(row[0]): row for row in rows}
        # No explicit Row[...] annotation on the linked rows: SQLAlchemy 2.1 made Row variadic, so Row[Any] means a
        # one-column row and mypy rejects row[3]. Inferring from fetchall() works on 2.0 and 2.1.
        if linked_pmids:
            linked_rows = db.execute(
                text("""
                    SELECT pmid, title, abstract,
                           1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity,
                           pub_date
                    FROM pubmed_abstracts
                    WHERE pmid = ANY(:pmids)
                    ORDER BY similarity DESC
                """),
                {
                    "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
                    "pmids": linked_pmids,
                },
            ).fetchall()
            for row in linked_rows:
                rows_by_pmid.setdefault(str(row[0]), row)
        all_rows = list(rows_by_pmid.values())
        # Rows are fully materialized, so the read transaction can end before the awaited PubMed and agent work.
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

        pub_years = {
            str(row[0]): year
            for row in all_rows
            if (year := pub_year(row[4])) is not None
        }
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
                pub_year=pub_years.get(str(pmid)),
            )
            scored.append((result, boost, final_score))

        scored.sort(key=lambda x: x[2], reverse=True)
        baseline_scored = [item for item in scored if item[0].pmid in baseline_pmids]

        if adaptive:
            top_k = await self._screen_depth(
                chembl_id,
                drug_names,
                disease,
                [item[0] for item in baseline_scored[:top_k]],
                target_passers,
            )

        # The recent pass ranks the full rerank set; the trial reserve below may truncate baseline_scored to the window.
        rerank_candidates = [item[0] for item in baseline_scored]
        trial_reserved: set[str] = set()
        if linked_pmids:
            trial_reference_set = set(linked_pmids)
            linked = [
                result for result, _, _ in scored if result.pmid in trial_reference_set
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
                result.pmid for result in exact_drug if on_topic.get(result.pmid, False)
            }
            reserved = [item for item in scored if item[0].pmid in eligible_pmids][
                :LITERATURE_TRIAL_REFERENCE_RESERVE
            ]
            reserved_pmids = {item[0].pmid for item in reserved}
            if reserved_pmids and not reserved_pmids.intersection(
                item[0].pmid for item in baseline_scored[:top_k]
            ):
                # The reserved papers take the window's last slots. A window the gates emptied (top_k 0) still carries
                # them, so a trial-linked passer is never lost to an unproductive pool.
                baseline_scored = (
                    baseline_scored[: max(top_k - len(reserved), 0)] + reserved
                )
                baseline_scored.sort(key=lambda item: item[2], reverse=True)
                top_k = max(top_k, len(reserved))
                trial_reserved = reserved_pmids

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

        window = [
            item[0].model_copy(
                update={
                    "selected_by": [
                        (
                            "trial_reference"
                            if item[0].pmid in trial_reserved
                            else "relevance"
                        )
                    ]
                }
            )
            for item in baseline_scored[:top_k]
        ]
        if not recent_pass:
            return window
        # Recent pass: the relevance cut under-reads recent papers on heavily studied pairs (metformin: 124 vs 71 recent
        # passers against the same number of extra papers read down the ranking). Measured from date_before in
        # holdout runs, whose pool is already cut off there.
        recent = _recent_pass(
            rerank_candidates,
            pub_years,
            (date_before or date.today()).year,
            {result.pmid for result in window},
        )
        logger.info(
            "[LIT] recent pass %s / %s: %d papers added to a window of %d",
            chembl_id,
            disease,
            len(recent),
            len(window),
        )
        return _union_tagged(window, recent, "recent")

    async def _screen_depth(
        self,
        chembl_id: str,
        drug_names: list[str],
        disease: str,
        ranked: list[AbstractResult],
        target_passers: int,
    ) -> int:
        """Return how many of `ranked` to keep: read LITERATURE_SCREEN_BATCH papers at a time through the exact-drug and
        disease-treatment gates and stop at the first batch boundary where `target_passers` have passed both, or at the
        end of the list. Lightly studied pairs stop early and heavily studied pairs stop once synthesis has enough.
        """
        passers = 0
        reached = 0
        for start in range(0, len(ranked), LITERATURE_SCREEN_BATCH):
            batch = ranked[start : start + LITERATURE_SCREEN_BATCH]
            drug_identity = await _judge_pmid_drug_identity(
                chembl_id, drug_names, batch, self.cache_dir
            )
            studied = [r for r in batch if drug_identity.get(r.pmid) == "studied"]
            on_topic = await _judge_pmid_treats_disease(
                chembl_id, drug_names[0], disease, studied, self.cache_dir
            )
            passers += sum(1 for r in studied if on_topic.get(r.pmid, False))
            reached = start + len(batch)
            if passers >= target_passers:
                break
        logger.info(
            "[LIT] screen %s / %s: read %d of %d ranked papers, %d passed both gates (target %d)",
            chembl_id,
            disease,
            reached,
            len(ranked),
            passers,
            target_passers,
        )
        return reached

    async def synthesize(
        self,
        chembl_id: str,
        disease: str,
        top_abstracts: list[AbstractResult],
        approved_indications: list[str] | None = None,
    ) -> EvidenceSummary:
        """Summarize PubMed evidence for a drug-disease pair using an LLM.

        Formats the top abstracts from semantic_search into a prompt, calls the LLM, and parses the JSON response into
        an EvidenceSummary.

        Args:
            chembl_id: ChEMBL ID of the drug (e.g. "CHEMBL1431").
            disease: Candidate disease (e.g. "colorectal cancer").
            top_abstracts: Output of semantic_search — list of dicts with keys "pmid", "title", "abstract", "similarity".

        Returns:
            EvidenceSummary with all fields populated from the LLM response.
        """
        # Cache key uses sorted PMIDs so orderings of the same evidence collapse to one entry. The sorted approved set
        # is also in the key, since the same PMIDs grade differently under different approved lists.
        approved = sorted(
            {i.strip() for i in (approved_indications or []) if i.strip()}
        )
        cache_params = {
            "chembl_id": chembl_id,
            "disease": disease,
            "pmids": sorted(r.pmid for r in top_abstracts),
            "approved_indications": approved,
            # The cap decides which of the same PMIDs synthesis reads.
            "synthesis_max": LITERATURE_SYNTHESIS_MAX,
            "llm_model": _settings.llm_model,
            # Bump when the relevance prompt or derived fields (direction rollup, strength cap) change, so stale judgments
            # cannot preserve behavior the new rules reject.
            "logic_version": "per_pmid_class_and_target_gates_v9_contradicting_trials",
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
        # Two isolated per-paper gates run BEFORE the combined prompt, so it only sees abstracts already established as this
        # drug alone, given to treat this disease. Both are cached per paper, so a paper's fate doesn't move with the batch.
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
        gated_abstracts = [
            result
            for result in top_abstracts
            if drug_identity.get(result.pmid) == "studied"
            and on_topic.get(result.pmid, False)
        ]
        # Every paper that passed both gates is graded on its own before synthesis; the grade is authoritative for its
        # direction and design. A paper the grader could not grade is dropped (fail closed), and the summary is then not
        # cached, so a transient grader failure is retried on the next run instead of persisting as missing evidence.
        pmid_grades = await _judge_pmid_grades(
            chembl_id, pref_name, disease, gated_abstracts, self.cache_dir
        )
        graded_abstracts = [
            result for result in gated_abstracts if result.pmid in pmid_grades
        ]
        grading_complete = len(graded_abstracts) == len(gated_abstracts)
        # Same-trial reports in one direction count once; past the cap, code samples the rest by direction. Secondary
        # reports and unselected passers are evidence that exists but was not read, so neither is listed as contaminated.
        merged_abstracts, secondary_reports = merge_same_trial(
            graded_abstracts, pmid_grades
        )
        synthesis_abstracts = select_synthesis_set(
            merged_abstracts, pmid_grades, LITERATURE_SYNTHESIS_MAX
        )
        selected_pmids = {result.pmid for result in synthesis_abstracts}
        screened_not_selected = [
            result.pmid
            for result in merged_abstracts
            if result.pmid not in selected_pmids
        ]
        unread_pmids = set(screened_not_selected).union(
            pmid for pmids in secondary_reports.values() for pmid in pmids
        )
        if secondary_reports or screened_not_selected:
            logger.info(
                "[LIT] %s / %s: %d graded, %d after same-trial merge, %d selected for synthesis",
                pref_name,
                disease,
                len(graded_abstracts),
                len(merged_abstracts),
                len(synthesis_abstracts),
            )
        # Class-level papers never enter the supporting/contradicting lists; they only distinguish "no evidence at all"
        # from "evidence exists, but only for the class".
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
            if grading_complete:
                cache_set(
                    "synthesize",
                    cache_params,
                    summary.model_dump(mode="json"),
                    self.cache_dir,
                    ttl=CACHE_TTL,
                )
            return summary

        logger.info(
            "[LIT] synthesizing %s: %d abstracts", disease, len(synthesis_abstracts)
        )
        formatted = "\n\n".join(
            f"PMID: {r.pmid}\n"
            f"Year: {r.pub_year if r.pub_year is not None else 'unknown'}\n"
            f"Graded direction: {pmid_grades[r.pmid].verdict}\n"
            f"Design: {_design_label(pmid_grades[r.pmid])}\n"
            f"Title: {r.title}\nAbstract: {r.abstract}"
            for r in synthesis_abstracts
        )

        template = (_PROMPTS_DIR / "synthesize.txt").read_text()
        prompt = template.format(
            drug_name=pref_name,
            disease_name=disease,
            abstracts=formatted,
            screened_counts=_screened_counts_text(
                merged_abstracts, pmid_grades, len(synthesis_abstracts)
            ),
            approved_indications=", ".join(approved) if approved else "(none)",
        )

        # Identifier lists below are rebuilt in code, so they cannot name an unsupplied abstract. The prose is the
        # exception (the model types those digits), so a cited PMID never supplied costs a retry, then the sentence.
        allowed_pmids = {r.pmid for r in synthesis_abstracts}
        prose_context = f"{pref_name} / {disease} synthesis"
        data = None
        for attempt in (1, 2):
            response = await query_llm(
                prompt if attempt == 1 else prompt + _PMID_RETRY_NOTE,
                max_tokens=SYNTHESIS_MAX_TOKENS,
            )
            # Tolerant parse: the merged prompt is long, so the model sometimes emits prose before the JSON or an
            # empty/overflowed response. parse_last_json_object scans for the last balanced {...} block. On a genuine
            # failure, DEGRADE to a safe floor (basis=none, strength none, all abstracts contaminated) rather than raise;
            # one bad LLM response must not crash the pipeline.
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
            # The loop breaks on a None parse, so reaching here means both attempts parsed a dict.
            data = cast(dict[str, Any], data)
            data["summary"] = strip_sentences_with_unknown_pmids(
                str(data.get("summary") or ""), allowed_pmids, context=prose_context
            )
            data["key_findings"] = strip_findings_with_unknown_pmids(
                [str(f) for f in (data.get("key_findings") or [])],
                allowed_pmids,
                context=prose_context,
            )

        # The floor is never cached: an unparseable response (e.g. one cut off at max_tokens) would otherwise persist as
        # "no evidence" for the cache TTL (semaglutide x heart failure, strong/supports read as none/none).
        synthesis_parsed = data is not None
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

        # Per-PMID directional verdict over the INPUT PMID set: the prompt labels each abstract
        # "contaminated"/"supporting"/"contradicting"/"mixed"; an omitted or unrecognizably labelled PMID counts as
        # contaminated (an unclassified abstract never counts as evidence). The supporting/contradicting/relevant lists
        # are built here from this map, not emitted by the LLM, which removed a bucketing error that mis-placed a positive
        # trial as contradicting (BRAVE-I).
        verdicts = data.get("verdicts")
        input_pmids = [r.pmid for r in top_abstracts if r.pmid not in unread_pmids]
        synthesis_pmids = {r.pmid for r in synthesis_abstracts}
        # Relevant = anything not contaminated. "neutral" (PK/safety/mechanism, set by the per-paper grade) is relevant
        # and counts toward study_count but is excluded from supporting AND contradicting below. synthesize never emits it.
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
            verdict_of = dict.fromkeys(input_pmids, "contaminated")

        # AUTHORITATIVE per-paper DIRECTION from the grades made before synthesis. synthesize's verdict map decides
        # relevant-vs-contaminated; the grade decides supporting/contradicting/mixed/neutral for every relevant paper,
        # replacing phrase-matching guards that couldn't attribute a benefit to the right drug arm (metformin x hepatic
        # steatosis). Every synthesis paper carries a grade, since ungraded papers were dropped before the call.
        relevant_for_direction = [
            r for r in synthesis_abstracts if verdict_of[r.pmid] in _RELEVANT_VERDICTS
        ]
        for result in relevant_for_direction:
            graded_verdict = pmid_grades[result.pmid].verdict
            if verdict_of[result.pmid] != graded_verdict:
                logger.debug(
                    "synthesize: pmid_grade set PMID %s %s->%s for %s / %s",
                    result.pmid,
                    verdict_of[result.pmid],
                    graded_verdict,
                    chembl_id,
                    disease,
                )
                verdict_of[result.pmid] = graded_verdict

        relevant_pmids = [p for p in input_pmids if verdict_of[p] in _RELEVANT_VERDICTS]
        contaminated_pmids = [
            p for p in input_pmids if verdict_of[p] not in _RELEVANT_VERDICTS
        ]
        # supporting = supporting + mixed; contradicting = contradicting + mixed (a mixed abstract appears in both).
        supporting_pmids = [
            p for p in input_pmids if verdict_of[p] in ("supporting", "mixed")
        ]
        contradicting_pmids = [
            p for p in input_pmids if verdict_of[p] in ("contradicting", "mixed")
        ]
        # neutral = relevant but non-efficacy (PK/safety/mechanism): in neither directional list, surfaced separately so a
        # cited-as-context PMID doesn't look dropped.
        neutral_pmids = [p for p in input_pmids if verdict_of[p] == "neutral"]
        # Contradicting papers whose study gave the drug to people and is not a pilot: the only evidence that may close a
        # signal on literature. Mixed papers are excluded (they report a benefit as well).
        contradicting_trial_pmids = [
            p
            for p in input_pmids
            if verdict_of[p] == "contradicting"
            and pmid_grades[p].is_human
            and pmid_grades[p].is_interventional
            and not pmid_grades[p].is_pilot
        ]

        data["relevant_pmids"] = relevant_pmids
        data["contaminated_pmids"] = contaminated_pmids
        data["supporting_pmids"] = supporting_pmids
        data["contradicting_pmids"] = contradicting_pmids
        data["neutral_pmids"] = neutral_pmids
        data["contradicting_trial_pmids"] = contradicting_trial_pmids
        data["secondary_report_pmids"] = secondary_reports
        data["screened_not_selected_pmids"] = screened_not_selected
        data["study_count"] = len(relevant_pmids)
        summary = EvidenceSummary(**data)

        # The synthesis model sometimes emits is_observational=False while describing every relevant human study as
        # uncontrolled, and the card renders False as "RCT-backed / controlled". So require a per-abstract controlled-design
        # signal first. Only a paper with an EFFICACY verdict may certify the pair (neutral papers are excluded from the
        # grade, so they must not set the design word). The judgment is per abstract because a document-wide phrase match
        # can't tell a study's own design from one it cites (sildenafil × ischemic stroke: an uncontrolled 12-patient safety
        # study describing placebo-controlled rat experiments supplied the "RCT-backed" claim). The guard is one-way: it
        # withdraws an unsupported controlled claim and never upgrades. With no efficacy-verdict paper, the synthesis value
        # stands.
        directional_judgments = [
            pmid_grades[result.pmid]
            for result in relevant_for_direction
            if verdict_of[result.pmid] in ("supporting", "contradicting", "mixed")
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

        # Animal-only is also derived from the per-abstract judgments: the synthesis model counts ANY human paper, even a
        # neutral safety study, as human evidence (sildenafil x ischemic stroke read "moderate, supports" on two rodent
        # studies plus a 12-patient safety study). Only efficacy-verdict papers count. If none is human the pair is
        # animal-only and strength is capped at "weak"; if at least one is human it is not. Both are one-way (a grade is
        # never raised), and with no efficacy-verdict paper the synthesis values stand.
        if summary.evidence_basis == "drug_specific" and directional_judgments:
            any_human = any(
                pmid_judgment.is_human for pmid_judgment in directional_judgments
            )
            summary.is_animal_only = not any_human
            if not any_human:
                summary.is_observational = None
                if summary.strength in ("strong", "moderate"):
                    logger.warning(
                        "synthesize: capping strength %s -> weak for %s / %s (no human efficacy paper)",
                        summary.strength,
                        pref_name,
                        disease,
                    )
                    summary.strength = "weak"

        # Preserve the synthesis model's evidence-weighted overall direction when papers disagree (a presence-only rollup let
        # any positive case report cancel controlled negative evidence). One-sided and no-efficacy cases stay deterministic.
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
            # Every relevant abstract is "neutral" (PK / safety-only / mechanism), so there is no efficacy result to grade.
            # Without this branch the LLM's direction survived unchecked (sildenafil x astrocytoma read "weak, supports" on
            # four neutral abstracts) and the non-zero study_count carried it past the supervisor's zero-evidence gate.
            summary.direction = "none"

        # DETERMINISTIC strength cap, the one clinical-accuracy invariant not trusted to the prompt: strength/direction grade
        # DRUG-SPECIFIC evidence only, so when evidence_basis is not "drug_specific" (class_level / approved / none) they
        # MUST be "none", else a class-level RCT body could inflate the card to "strong" while the prose says "no direct
        # evidence for <drug>" (the Parkinson bug). The supervisor ranking path reads es.strength directly.
        if summary.evidence_basis != "drug_specific":
            summary.strength = "none"
            summary.direction = "none"

        if grading_complete and synthesis_parsed:
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

        Combines the drug's synonyms, gene targets, mechanisms of action, and ATC classifications with the organ term
        extracted from the disease name. Results are cached by drug/disease pair and deduplicated (case-insensitive).

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

        # Resolve the disease to its canonical MeSH preferred term for the prompt: PubMed's auto-term-mapping breaks on bare
        # multi-word phrases around AND, so the prompt has the LLM wrap the term in double quotes. Falls back to the raw
        # disease name if the MeSH lookup misses.
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

        # Substitute the <DISEASE> placeholder with the quoted MeSH preferred term; the LLM emits the placeholder so the
        # JSON array never contains embedded quotes.
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
