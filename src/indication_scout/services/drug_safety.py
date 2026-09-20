"""Drug safety service: adverse-event retrieval, openFDA label safety, and harm adjudication."""

import asyncio
import logging
from datetime import date
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError

from indication_scout.agents.literature.pubmed_ae import search_adverse_events
from indication_scout.config import get_settings
from indication_scout.constants import (
    CACHE_TTL,
    SAFETY_QUOTE_MAX_WORDS,
    SAFETY_TOP_ADVERSE_EVENTS,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import get_all_drug_names
from indication_scout.data_sources.fda import FDAClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_fda import FDALabelSafetyRecord
from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.models.model_safety import (
    DrugSafetyAssessment,
    SafetyPaperVerdict,
)
from indication_scout.services.llm import parse_last_json_object, query_llm
from indication_scout.services.retrieval import AbstractResult
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


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


class DrugSafetyService:
    """Drug safety assessment bound to a specific cache directory.

    Instantiate with the appropriate cache_dir for the calling context
    (e.g. DEFAULT_CACHE_DIR for production, TEST_CACHE_DIR for tests).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._label_safety_tasks: dict[
            str, asyncio.Task[list[FDALabelSafetyRecord]]
        ] = {}

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
            # Written by RetrievalService.get_drug_competitors in services/retrieval.py; keep the namespace and key
            # shape byte-identical with that writer.
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
                # The comprehension above already excludes None ratios.
                key=lambda event: cast(float, event.log_likelihood_ratio),
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
            "logic_version": "per_paper_harm_v2",
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
            # Lead each quote with the named outcome. A quote alone can read as "no harm" when the
            # harm sits in its last sentence (bupropion x cocaine use disorder, PMID 25494008:
            # "...without seizures. CONCLUSION: ...possibility of oral bupropion addiction").
            findings = [
                f'{verdict.adverse_outcome.strip()} — "{verdict.evidence_quote.strip()}"'
                for verdict in confirmed
                if verdict.evidence_quote and verdict.adverse_outcome
            ]
            summary = (
                f"Disease-scoped literature for {pref_name} in {disease} reported: "
                f"{'; '.join(dict.fromkeys(findings))} "
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
