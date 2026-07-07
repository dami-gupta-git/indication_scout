"""
Open Targets Platform GraphQL client.

Two primary methods:
  1. get_drug        — Fetch drug data (indications, targets, warnings, adverse events)
  2. get_target_data — Fetch target data (associations, pathways, interactions)

Plus convenience accessors for specific target data slices.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypedDict

from indication_scout.config import get_settings
from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    CACHE_TTL,
    CLINICAL_STAGE_RANK,
    DEFAULT_CACHE_DIR,
    INTERACTION_TYPE_MAP,
    OPEN_TARGETS_BASE_URL,
)
from indication_scout.markers import no_review
from indication_scout.utils.cache import cache_get, cache_set
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient, get_all_drug_names
from indication_scout.helpers.drug_helpers import normalize_drug_name

from indication_scout.models.model_open_targets import (
    Association,
    Pathway,
    Interaction,
    ClinicalDisease,
    DrugSummary,
    MousePhenotype,
    TargetData,
    GeneticConstraint,
    AdverseEvent,
    BiologicalModel,
    DrugData,
    DrugTarget,
    EvidenceRecord,
    MechanismOfAction,
    DrugWarning,
    Indication,
    SafetyLiability,
    SafetyEffect,
    DiseaseSynonyms,
    RichDrugData,
    VariantFunctionalConsequence,
)

logger = logging.getLogger(__name__)

_settings = get_settings()

# Per-target evidences cache. One file per target_id holds
# {efo_id: {records, cached_at, ttl}}. Replaces the prior `target_evidences`
# namespace which fanned out one file per (target_id, efo_id) pair.
# TTL is per-pair so adding a new efo_id later does not reset existing
# entries' freshness. Reads are concurrent-safe; writes are serialized
# in get_target_evidences (one merge+write after all fetches gather).
_TARGET_EVIDENCES_NS = "target_evidences"


def _target_evidences_path(target_id: str, cache_dir: Path) -> Path:
    """Return the per-target cache file path for the evidences namespace."""
    return cache_dir / _TARGET_EVIDENCES_NS / f"{target_id}.json"


def _load_target_evidences(
    target_id: str, cache_dir: Path
) -> dict[str, dict[str, Any]]:
    """Load the per-target evidences file as {efo_id: {records, cached_at, ttl}}.

    Returns an empty dict if the file is missing or unparseable. Expired
    entries are dropped lazily; the file is rewritten only when a caller
    invokes _save_target_evidences.
    """
    path = _target_evidences_path(target_id, cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}

    fresh: dict[str, dict[str, Any]] = {}
    now = datetime.now()
    for efo_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            ttl = int(entry.get("ttl", CACHE_TTL))
            records = entry["records"]
        except (KeyError, TypeError, ValueError):
            continue
        if (now - cached_at).total_seconds() > ttl:
            continue
        if not isinstance(records, list):
            continue
        fresh[efo_id] = {
            "records": records,
            "cached_at": entry["cached_at"],
            "ttl": ttl,
        }
    return fresh


def _save_target_evidences(
    target_id: str,
    new_records: dict[str, list[dict[str, Any]]],
    cache_dir: Path,
    ttl: int = CACHE_TTL,
) -> None:
    """Merge new_records into the per-target file and write it back.

    Existing unexpired entries are preserved; new EFO entries overwrite any
    prior entry for the same efo_id (refreshing its cached_at).
    """
    if not new_records:
        return
    path = _target_evidences_path(target_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_target_evidences(target_id, cache_dir)
    now_iso = datetime.now().isoformat()
    for efo_id, records in new_records.items():
        existing[efo_id] = {
            "records": records,
            "cached_at": now_iso,
            "ttl": ttl,
        }
    payload = {
        "ns": _TARGET_EVIDENCES_NS,
        "target_id": target_id,
        "entries": existing,
    }
    path.write_text(json.dumps(payload, default=str, indent=2))


class CompetitorRawData(TypedDict):
    diseases: dict[str, set[str]]
    disease_efo_ids: dict[str, str]
    drug_indications: list[str]


class OpenTargetsClient(BaseClient):
    BASE_URL = OPEN_TARGETS_BASE_URL
    PAGE_SIZE = _settings.open_targets_page_size

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        super().__init__()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _source_name(self) -> str:
        return "open_targets"

    # ------------------------------------------------------------------
    # Public: get_drug and get_target and get_rich_drug_data
    # ------------------------------------------------------------------

    async def get_rich_drug_data(self, chembl_id: str) -> RichDrugData:
        """Fetch drug data and all associated target data in parallel."""
        drug = await self.get_drug(chembl_id)
        targets = await asyncio.gather(
            *[self.get_target_data(t.target_id) for t in drug.targets]
        )
        return RichDrugData(drug=drug, targets=list(targets))

    async def get_drug(self, chembl_id: str) -> DrugData:
        """Fetch drug data by ChEMBL ID, enriched with ATC codes from ChEMBL.

        Also warms the chembl_id_to_names cache (which doubles as the reverse
        index for resolve_drug_name) so downstream lookups via get_all_drug_names
        don't re-hit the API.
        """
        cached = cache_get("drug", {"chembl_id": chembl_id}, self.cache_dir)
        if cached:
            return DrugData.model_validate(cached)

        async with ChEMBLClient() as chembl_client:
            drug_data, molecule, names_result = await asyncio.gather(
                self._fetch_drug(chembl_id),
                chembl_client.get_molecule(chembl_id),
                get_all_drug_names(chembl_id, self.cache_dir),
                return_exceptions=True,
            )

        if isinstance(drug_data, BaseException):
            raise DataSourceError(
                self._source_name,
                f"Failed to fetch drug data for '{chembl_id}': {drug_data}",
            )

        if isinstance(molecule, BaseException):
            logger.warning("ChEMBL ATC lookup failed for %s: %s", chembl_id, molecule)
        else:
            drug_data.atc_classifications = molecule.atc_classifications

        if isinstance(names_result, BaseException):
            logger.warning(
                "ChEMBL drug-names warmup failed for %s: %s", chembl_id, names_result
            )

        cache_set(
            "drug",
            {"chembl_id": chembl_id},
            drug_data.model_dump(),
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return drug_data

    async def get_drug_competitors(
        self,
        chembl_id: str,
        min_stage: str = "PHASE_3",
        date_before: date | None = None,
    ) -> CompetitorRawData:
        """Fetch competitor drugs for a given drug, grouped by disease.

        When `date_before` is set (temporal holdout mode), the OT-derived
        approved-indications strip is suppressed: OT's `drug.indications`
        reflects today's approval state and would leak post-cutoff
        approvals into a holdout. The caller is expected to apply its own
        cutoff-aware approval filter (the hardcoded approvals table) on the
        returned competitor list.
        """
        min_rank = CLINICAL_STAGE_RANK.get(min_stage, 0)

        cache_params = {
            "chembl_id": chembl_id,
            "min_stage": min_stage,
            "date_before": date_before.isoformat() if date_before else None,
            "prefetch_max": _settings.open_targets_competitor_prefetch_max,
        }
        cached = cache_get("competitors_raw", cache_params, self.cache_dir)
        if cached is not None:
            return CompetitorRawData(
                diseases={
                    disease: set(drugs) for disease, drugs in cached["diseases"].items()
                },
                disease_efo_ids=dict(cached.get("disease_efo_ids") or {}),
                drug_indications=cached["drug_indications"],
            )

        drug = await self.get_drug(chembl_id)
        targets = drug.targets

        # Build set of approved indication names to exclude from repurposing
        # candidates. Skipped under date_before — see method docstring.
        if date_before is None:
            approved_indications: set[str] = {
                i.disease_name.lower()
                for i in drug.indications
                if i.max_clinical_stage == "APPROVAL" and i.disease_name is not None
            }
        else:
            approved_indications = set()

        siblings_with_stage: dict[str, dict[str, int]] = {}
        id_to_canonical: dict[str, str] = {}

        all_summaries = await asyncio.gather(
            *[self.get_target_data_drug_summaries(t.target_id) for t in targets]
        )
        for t, summaries in zip(targets, all_summaries):
            logger.debug(t.mechanism_of_action)
            for summary in summaries:
                stage_rank = CLINICAL_STAGE_RANK.get(
                    summary.max_clinical_stage or "", 0
                )
                if stage_rank >= min_rank:
                    drug_name = normalize_drug_name(summary.drug_name)
                    for cd in summary.diseases:
                        if cd.disease_name is None:
                            continue
                        if cd.disease_id and cd.disease_id in id_to_canonical:
                            disease = id_to_canonical[cd.disease_id]
                        else:
                            disease = cd.disease_name.lower()
                            if cd.disease_id:
                                id_to_canonical[cd.disease_id] = disease
                        if disease not in siblings_with_stage:
                            siblings_with_stage[disease] = {}
                        existing = siblings_with_stage[disease].get(drug_name, 0)
                        siblings_with_stage[disease][drug_name] = max(
                            existing, stage_rank
                        )

        # Remove diseases that are already approved indications for this drug.
        for key in list(siblings_with_stage):
            if key in approved_indications:
                del siblings_with_stage[key]

        siblings: dict[str, set[str]] = {
            disease: set(drugs.keys()) for disease, drugs in siblings_with_stage.items()
        }

        # Remove overly broad disease terms (e.g. "cancer", "carcinoma") that
        # produce noisy, unfocused PubMed queries.
        for key in list(siblings):
            words = {w.lower() for w in key.split()}
            if words <= BROADENING_BLOCKLIST:
                del siblings[key]

        sorted_siblings = dict(
            sorted(siblings.items(), key=lambda item: len(item[1]), reverse=True)
        )

        drug_indications = list(approved_indications)
        top_40 = dict(
            list(sorted_siblings.items())[
                : _settings.open_targets_competitor_prefetch_max
            ]
        )

        # Inverse of id_to_canonical, scoped to the diseases that survived ranking and trimming.
        disease_efo_ids = {
            canonical: efo_id
            for efo_id, canonical in id_to_canonical.items()
            if canonical in top_40
        }

        result = CompetitorRawData(
            diseases=top_40,
            disease_efo_ids=disease_efo_ids,
            drug_indications=drug_indications,
        )

        cache_set(
            "competitors_raw",
            cache_params,
            {
                "diseases": {d: list(drugs) for d, drugs in top_40.items()},
                "disease_efo_ids": disease_efo_ids,
                "drug_indications": drug_indications,
            },
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return result

    async def get_drug_indications(self, chembl_id: str) -> list[Indication]:
        drug = await self.get_drug(chembl_id)
        return drug.indications

    async def get_drug_target_competitors(
        self, chembl_id: str
    ) -> dict[str, list[DrugSummary]]:
        """For each target of a drug, fetch all drugs acting on that target.

        Returns a dict mapping target symbol (e.g. "GLP1R") to the list of
        DrugSummary objects from Open Targets' drugAndClinicalCandidates for that target.
        """
        drug = await self.get_drug(chembl_id)
        result: dict[str, list[DrugSummary]] = {}
        for target in drug.targets:
            drug_summaries = await self.get_target_data_drug_summaries(target.target_id)
            result[target.target_symbol] = drug_summaries
        return result

    async def get_target_data(self, target_id: str) -> TargetData:
        """Fetch target data by ID."""
        cached = cache_get("target", {"target_id": target_id}, self.cache_dir)
        if cached:
            return TargetData.model_validate(cached)

        target_data = await self._fetch_target(target_id)

        cache_set(
            "target",
            {"target_id": target_id},
            target_data.model_dump(),
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return target_data

    # ------------------------------------------------------------------
    # Public accessors — convenience methods using get_drug/get_target
    # ------------------------------------------------------------------

    async def get_target_data_associations(self, target_id: str) -> list[Association]:
        target = await self.get_target_data(target_id)
        return [
            a
            for a in target.associations
            if a.overall_score >= _settings.open_targets_association_min_score
        ]

    async def get_target_data_pathways(self, target_id: str) -> list[Pathway]:
        target = await self.get_target_data(target_id)
        return target.pathways

    async def get_target_data_interactions(self, target_id: str) -> list[Interaction]:
        target = await self.get_target_data(target_id)
        return target.interactions

    async def get_target_data_drug_summaries(self, target_id: str) -> list[DrugSummary]:
        target = await self.get_target_data(target_id)
        return target.drug_summaries

    async def get_target_data_mouse_phenotypes(
        self, target_id: str
    ) -> list[MousePhenotype]:
        target = await self.get_target_data(target_id)
        return target.mouse_phenotypes

    async def get_target_data_safety_liabilities(
        self, target_id: str
    ) -> list[SafetyLiability]:
        target = await self.get_target_data(target_id)
        return target.safety_liabilities

    async def get_target_data_genetic_constraints(
        self, target_id: str
    ) -> list[GeneticConstraint]:
        target = await self.get_target_data(target_id)
        return target.genetic_constraint

    async def get_target_evidences(
        self, target_id: str, efo_ids: list[str]
    ) -> dict[str, list[EvidenceRecord]]:
        """Fetch per-(target, disease) evidence records.

        Returns a dict keyed by disease_id mapping to a list of evidence
        records. Each record carries directionality fields used to classify
        whether the drug's action aligns with or opposes the disease
        mechanism. Empty list for any efo_id with no evidence.

        Fans out per-efo so each (target, disease) pair gets its own
        200-record budget from OT. Batched queries share that budget
        across all efoIds, which starves rare diseases when they're
        batched with common ones. Per-efo fetches run in parallel via
        asyncio.gather.

        Cache is one file per target_id (`target_evidences/<target_id>.json`)
        holding {efo_id: records}. Reads happen once at the top; writes are
        batched into a single merge+rewrite after all fresh fetches complete,
        so concurrent EFO fetches for the same target cannot stomp the file.
        """
        if not efo_ids:
            return {}

        cached = _load_target_evidences(target_id, self.cache_dir)

        results: dict[str, list[EvidenceRecord]] = {}
        missing: list[str] = []
        for efo_id in efo_ids:
            entry = cached.get(efo_id)
            if entry is None:
                missing.append(efo_id)
            else:
                results[efo_id] = [
                    EvidenceRecord.model_validate(r) for r in entry["records"]
                ]

        if missing:
            fresh = await asyncio.gather(
                *[self._fetch_evidences_single(target_id, efo) for efo in missing]
            )
            new_serialized: dict[str, list[dict[str, Any]]] = {}
            for efo_id, records in zip(missing, fresh):
                results[efo_id] = records
                new_serialized[efo_id] = [r.model_dump() for r in records]
            _save_target_evidences(
                target_id, new_serialized, self.cache_dir, ttl=CACHE_TTL
            )

        return {efo_id: results[efo_id] for efo_id in efo_ids}

    async def _fetch_evidences_single(
        self, target_id: str, efo_id: str
    ) -> list[EvidenceRecord]:
        """Fetch evidence records for a single (target, disease) pair (no cache)."""
        data = await self._graphql(
            self.BASE_URL,
            EVIDENCES_QUERY,
            variables={"id": target_id, "efoIds": [efo_id]},
        )
        raw_target = (data.get("data") or {}).get("target") or {}
        rows = (raw_target.get("evidences") or {}).get("rows") or []

        records = [self._parse_evidence(r) for r in rows]
        # Only keep records whose disease_id matches what we asked for —
        # defends against the API returning cross-linked disease rows.
        records = [r for r in records if r.disease_id == efo_id]
        return records

    async def get_disease_drugs(self, disease_id: str) -> list[DrugSummary]:
        """All drugs for a disease, any target, any mechanism."""
        cached = cache_get("disease_drugs", {"disease_id": disease_id}, self.cache_dir)
        if cached:
            return [DrugSummary.model_validate(d) for d in cached]

        data = await self._graphql(
            self.BASE_URL, DISEASE_DRUGS_QUERY, {"id": disease_id}
        )
        result = self._parse_disease_drugs(data["data"])

        cache_set(
            "disease_drugs",
            {"disease_id": disease_id},
            [d.model_dump() for d in result],
            self.cache_dir,
        )

        return result

    async def get_disease_synonyms(self, disease_name: str) -> DiseaseSynonyms:
        """Fetch exact and related synonyms for a disease by name."""
        disease_id = await self._resolve_disease_name(disease_name)

        cached = cache_get(
            "disease_synonyms", {"disease_id": disease_id}, self.cache_dir
        )
        if cached:
            return DiseaseSynonyms.model_validate(cached)

        data = await self._graphql(
            self.BASE_URL, DISEASE_SYNONYMS_QUERY, {"id": disease_id}
        )
        raw_disease = data["data"]["disease"]
        if raw_disease is None:
            raise DataSourceError(
                self._source_name,
                f"No disease found for '{disease_name}'",
            )
        relation_map: dict[str, str] = {
            "hasExactSynonym": "exact",
            "hasRelatedSynonym": "related",
            "hasNarrowSynonym": "narrow",
            "hasBroadSynonym": "broad",
        }
        grouped: dict[str, list[str]] = {v: [] for v in relation_map.values()}
        for entry in raw_disease.get("synonyms", []) or []:
            field = relation_map.get(entry.get("relation", ""))
            if field:
                grouped[field].extend(entry.get("terms", []) or [])

        parent_names = [p["name"] for p in raw_disease.get("parents", []) or []]

        result = DiseaseSynonyms(
            disease_id=raw_disease["id"],
            disease_name=raw_disease["name"].lower(),
            parent_names=parent_names,
            **grouped,
        )

        cache_set(
            "disease_synonyms",
            {"disease_id": disease_id},
            result.model_dump(),
            self.cache_dir,
        )

        return result

    # ------------------------------------------------------------------
    # Private: network calls
    # ------------------------------------------------------------------

    async def _resolve_disease_name(self, name: str) -> str:
        """Search by name → return EFO/MONDO disease ID."""
        data = await self._graphql(self.BASE_URL, DISEASE_SEARCH_QUERY, {"q": name})
        hits = data["data"]["search"]["hits"]
        disease_hits = [h for h in hits if h["entity"] == "disease"]
        if not disease_hits:
            raise DataSourceError(
                self._source_name,
                f"No disease found for '{name}'",
            )
        return disease_hits[0]["id"]

    async def resolve_disease_id(self, name: str) -> str | None:
        """Public, cached, non-raising wrapper around _resolve_disease_name.

        Returns OT's canonical disease ID for the given name, or None when no hit
        is found. Intended for dedup paths where a missing resolution is non-fatal.
        """
        cache_params = {"name": name.strip().lower()}
        cached = cache_get("disease_id_resolver", cache_params, self.cache_dir)
        if cached is not None:
            return cached or None
        try:
            disease_id = await self._resolve_disease_name(name)
        except DataSourceError:
            cache_set("disease_id_resolver", cache_params, "", self.cache_dir)
            return None
        cache_set("disease_id_resolver", cache_params, disease_id, self.cache_dir)
        return disease_id

    async def _fetch_drug(self, chembl_id: str) -> DrugData:
        """Fetch full drug node from Open Targets."""
        data = await self._graphql(
            self.BASE_URL,
            DRUG_QUERY,
            variables={"id": chembl_id},
        )
        raw_drug = data["data"]["drug"]
        if raw_drug is None:
            raise DataSourceError(
                self._source_name,
                f"No drug found for ChEMBL ID '{chembl_id}'",
            )
        return self._parse_drug_data(raw_drug)

    async def _fetch_target(self, target_id: str) -> TargetData:
        """Fetch full target node. Paginates associations if needed."""
        data = await self._graphql(
            self.BASE_URL,
            TARGET_QUERY,
            variables={"id": target_id},
        )
        raw_target = data["data"]["target"]
        if raw_target is None:
            raise DataSourceError(
                self._source_name,
                f"No target found for '{target_id}'",
            )
        target_data = self._parse_target_data(raw_target)

        # Paginate if we hit the association page limit
        if len(target_data.associations) >= self.PAGE_SIZE:
            target_data.associations = await self._paginate_associations(target_id)

        return target_data

    async def _paginate_associations(self, target_id: str) -> list[Association]:
        """Fetch all associations when count exceeds single page."""
        all_associations = []
        page_index = 0

        while True:
            data = await self._graphql(
                self.BASE_URL,
                ASSOCIATIONS_PAGE_QUERY,
                variables={
                    "id": target_id,
                    "index": page_index,
                    "size": self.PAGE_SIZE,
                },
            )

            rows = data["data"]["target"]["associatedDiseases"]["rows"]
            all_associations.extend(self._parse_association(r) for r in rows)

            if len(rows) < self.PAGE_SIZE:
                break
            page_index += 1

        return all_associations

    # ------------------------------------------------------------------
    # Parsers: raw GraphQL response → Pydantic models
    # ------------------------------------------------------------------

    def _parse_drug_data(self, raw: dict) -> DrugData:
        targets: list[DrugTarget] = []
        mechanisms_of_action: list[MechanismOfAction] = []
        # Open Targets aggregates MoA rows across source databases (ChEMBL, DrugBank, etc.),
        # which yields rows with identical (mechanism, action_type, targets). Dedupe on that key
        # so the same MoA isn't reported multiple times downstream.
        seen_moa: set[tuple] = set()
        seen_target: set[tuple] = set()
        for row in (raw.get("mechanismsOfAction") or {}).get("rows", []):
            moa = row["mechanismOfAction"]
            action_type = row.get("actionType")
            row_targets = row.get("targets", [])
            target_ids = [t["id"] for t in row_targets]
            target_symbols = [t["approvedSymbol"] for t in row_targets]
            moa_key = (moa, action_type, tuple(target_ids))
            if moa_key not in seen_moa:
                seen_moa.add(moa_key)
                mechanisms_of_action.append(
                    MechanismOfAction(
                        mechanism_of_action=moa,
                        action_type=action_type,
                        target_ids=target_ids,
                        target_symbols=target_symbols,
                    )
                )
            for t in row_targets:
                target_key = (t["id"], moa, action_type)
                if target_key in seen_target:
                    continue
                seen_target.add(target_key)
                targets.append(
                    DrugTarget(
                        target_id=t["id"],
                        target_symbol=t["approvedSymbol"],
                        mechanism_of_action=moa,
                        action_type=action_type,
                    )
                )

        warnings = [
            DrugWarning(
                warning_type=w.get("warningType", ""),
                description=w.get("description"),
                toxicity_class=w.get("toxicityClass"),
                country=w.get("country"),
                year=w.get("year"),
                efo_id=w.get("efoId"),
            )
            for w in raw.get("drugWarnings", [])
        ]

        indications = [
            Indication(
                id=row.get("id", ""),
                disease_id=row["disease"]["id"],
                disease_name=(row["disease"]["name"] or "").lower(),
                max_clinical_stage=row.get("maxClinicalStage"),
            )
            for row in (raw.get("indications") or {}).get("rows", [])
        ]

        adverse_events = [
            self._parse_adverse_event(ae)
            for ae in (raw.get("adverseEvents") or {}).get("rows", [])
        ]

        return DrugData(
            chembl_id=raw["id"],
            drug_type=raw.get("drugType"),
            maximum_clinical_stage=raw.get("maximumClinicalStage"),
            mechanisms_of_action=mechanisms_of_action,
            warnings=warnings,
            indications=indications,
            targets=targets,
            adverse_events=adverse_events,
            adverse_events_critical_value=(raw.get("adverseEvents") or {}).get(
                "criticalValue"
            ),
        )

    def _parse_target_data(self, raw: dict) -> TargetData:
        return TargetData(
            target_id=raw["id"],
            symbol=raw["approvedSymbol"],
            name=raw.get("approvedName", ""),
            function_descriptions=raw.get("functionDescriptions") or [],
            associations=[
                self._parse_association(r)
                for r in (raw.get("associatedDiseases") or {}).get("rows", [])
            ],
            pathways=[self._parse_pathway(p) for p in raw.get("pathways", [])],
            interactions=[
                self._parse_interaction(i)
                for i in (raw.get("interactions") or {}).get("rows", [])
            ],
            drug_summaries=[
                self._parse_drug_summary(d)
                for d in (raw.get("drugAndClinicalCandidates") or {}).get("rows", [])
            ],
            mouse_phenotypes=[
                self._parse_phenotype(p) for p in raw.get("mousePhenotypes", [])
            ],
            safety_liabilities=[
                self._parse_safety_liability(sl)
                for sl in raw.get("safetyLiabilities", [])
            ],
            genetic_constraint=[
                self._parse_constraint(c) for c in raw.get("geneticConstraint", [])
            ],
        )

    def _parse_association(self, raw: dict) -> Association:
        datatype_scores = {s["id"]: s["score"] for s in raw.get("datatypeScores", [])}
        datasource_scores = {
            s["id"]: s["score"] for s in raw.get("datasourceScores", [])
        }
        disease = raw["disease"]
        therapeutic_areas = [ta["name"] for ta in disease.get("therapeuticAreas", [])]
        return Association(
            disease_id=disease["id"],
            disease_name=(disease["name"] or "").lower(),
            disease_description=disease.get("description") or "",
            overall_score=raw["score"],
            datatype_scores=datatype_scores,
            datasource_scores=datasource_scores,
            therapeutic_areas=therapeutic_areas,
        )

    def _parse_evidence(self, raw: dict) -> EvidenceRecord:
        disease = raw.get("disease") or {}
        vfc_raw = raw.get("variantFunctionalConsequence") or {}
        vfc = (
            VariantFunctionalConsequence(
                id=vfc_raw.get("id") or "",
                label=vfc_raw.get("label") or "",
            )
            if vfc_raw
            else None
        )
        return EvidenceRecord(
            disease_id=disease.get("id") or "",
            datatype_id=raw.get("datatypeId") or "",
            score=raw.get("score"),
            direction_on_target=raw.get("directionOnTarget"),
            direction_on_trait=raw.get("directionOnTrait"),
            variant_functional_consequence=vfc,
        )

    def _parse_pathway(self, raw: dict) -> Pathway:
        return Pathway(
            pathway_id=raw.get("pathwayId", ""),
            pathway_name=raw.get("pathway", ""),
            top_level_pathway=raw.get("topLevelTerm", ""),
        )

    def _parse_interaction(self, raw: dict) -> Interaction:
        source = raw.get("sourceDatabase", "")
        target_b = raw.get("targetB", {}) or {}
        return Interaction(
            interacting_target_id=target_b.get("id", raw.get("intB", "")),
            interacting_target_symbol=target_b.get("approvedSymbol", ""),
            interaction_score=raw.get("score"),
            source_database=source,
            biological_role=raw.get("intBBiologicalRole", ""),
            evidence_count=raw.get("count", 0),
            interaction_type=INTERACTION_TYPE_MAP.get(source.lower()),
        )

    def _parse_drug_summary(self, raw: dict) -> DrugSummary:
        drug = raw.get("drug") or {}
        diseases = []
        for d in raw.get("diseases", []):
            d_node = d.get("disease") or {}
            d_name = d_node.get("name")
            diseases.append(
                ClinicalDisease(
                    disease_from_source=d.get("diseaseFromSource", ""),
                    disease_id=d_node.get("id"),
                    disease_name=d_name.lower() if d_name else None,
                )
            )
        return DrugSummary(
            id=raw.get("id", ""),
            drug_id=drug.get("id", ""),
            drug_name=drug.get("name", "").lower(),
            drug_type=drug.get("drugType"),
            max_clinical_stage=raw.get("maxClinicalStage"),
            diseases=diseases,
        )

    def _parse_phenotype(self, raw: dict) -> MousePhenotype:
        categories = [c["label"] for c in raw.get("modelPhenotypeClasses", [])]
        models = [
            BiologicalModel(
                allelic_composition=m.get("allelicComposition") or "",
                genetic_background=m.get("geneticBackground") or "",
                literature=m.get("literature") or [],
                model_id=m.get("id") or "",
            )
            for m in raw.get("biologicalModels") or []
        ]
        return MousePhenotype(
            phenotype_id=raw.get("modelPhenotypeId") or "",
            phenotype_label=raw.get("modelPhenotypeLabel") or "",
            phenotype_categories=categories,
            biological_models=models,
        )

    def _parse_adverse_event(self, raw: dict) -> AdverseEvent:
        return AdverseEvent(
            name=raw["name"],
            count=raw["count"],
            log_likelihood_ratio=raw["logLR"],
        )

    def _parse_safety_liability(self, raw: dict) -> SafetyLiability:
        effects = [
            SafetyEffect(
                direction=e.get("direction", ""),
                dosing=e.get("dosing"),
            )
            for e in raw.get("effects", [])
        ]
        return SafetyLiability(
            event=raw.get("event"),
            event_id=raw.get("eventId"),
            effects=effects,
            datasource=raw.get("datasource"),
            literature=raw.get("literature"),
            url=raw.get("url"),
        )

    def _parse_constraint(self, raw: dict) -> GeneticConstraint:
        return GeneticConstraint(
            constraint_type=raw["constraintType"],
            exp=raw.get("exp"),
            obs=raw.get("obs"),
            oe=raw.get("oe"),
            oe_lower=raw.get("oeLower"),
            oe_upper=raw.get("oeUpper"),
            score=raw.get("score"),
            upper_bin=raw.get("upperBin"),
            upper_bin6=raw.get("upperBin6"),
        )

    def _parse_disease_drugs(self, data: dict) -> list[DrugSummary]:
        """Parse disease drugs — one entry per drug."""
        disease = data.get("disease") or {}
        rows = (disease.get("drugAndClinicalCandidates") or {}).get("rows", [])
        return [self._parse_drug_summary(row) for row in rows]


# ------------------------------------------------------------------
# GraphQL queries
# ------------------------------------------------------------------

DISEASE_SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 1}) {
        hits { id entity }
    }
}
"""

DRUG_QUERY = """
query($id: String!) {
    drug(chemblId: $id) {
        id drugType
        maximumClinicalStage

        mechanismsOfAction {
            rows {
                mechanismOfAction actionType
                targets { id approvedSymbol }
            }
        }

        indications {
            rows {
                id maxClinicalStage
                disease { id name }
                clinicalReports {
                    id source clinicalStage hasExpertReview
                    title type trialOverallStatus trialLiterature
                    drugs { drugFromSource drug { id name } }
                    diseases { diseaseFromSource disease { id name } }
                }
            }
        }

        drugWarnings {
            warningType description toxicityClass
            country year efoId efoTerm
        }

        adverseEvents(page: {index: 0, size: 100}) {
            rows { name count logLR }
            criticalValue
        }
    }
}
"""

TARGET_QUERY = """
query($id: String!) {
    target(ensemblId: $id) {
        id approvedSymbol approvedName functionDescriptions

        associatedDiseases(page: {index: 0, size: 500}) {
            rows {
                disease {
                    id name description
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
                datasourceScores { id score }
            }
        }

        pathways { pathwayId pathway topLevelTerm }

        interactions(page: {index: 0, size: 200}) {
            rows {
                intB intBBiologicalRole score
                sourceDatabase count
                targetB { id approvedSymbol }
            }
        }

        drugAndClinicalCandidates {
            rows {
                id maxClinicalStage
                drug {
                    id name drugType
                    mechanismsOfAction {
                        rows {
                            mechanismOfAction actionType
                            targets { id approvedSymbol }
                        }
                    }
                }
                diseases { diseaseFromSource disease { id name } }
                clinicalReports {
                    id source clinicalStage hasExpertReview
                    title type trialOverallStatus trialLiterature
                    drugs { drugFromSource drug { id name } }
                    diseases { diseaseFromSource disease { id name } }
                }
            }
        }

        mousePhenotypes {
            modelPhenotypeId modelPhenotypeLabel
            modelPhenotypeClasses { id label }
            biologicalModels {
                allelicComposition geneticBackground
                id literature
            }
        }

        safetyLiabilities {
            event
            eventId
            effects { direction dosing }
            datasource
            literature
            url
        }

        geneticConstraint {
            constraintType score exp obs
            oe oeLower oeUpper upperBin upperBin6
        }
    }
}
"""

ASSOCIATIONS_PAGE_QUERY = """
query($id: String!, $index: Int!, $size: Int!) {
    target(ensemblId: $id) {
        associatedDiseases(page: {index: $index, size: $size}) {
            rows {
                disease {
                    id name description
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
                datasourceScores { id score }
            }
        }
    }
}
"""

EVIDENCES_QUERY = """
query($id: String!, $efoIds: [String!]!) {
    target(ensemblId: $id) {
        evidences(efoIds: $efoIds, size: 200) {
            rows {
                datatypeId
                score
                directionOnTarget
                directionOnTrait
                disease { id name }
                variantFunctionalConsequence { id label }
            }
        }
    }
}
"""

DISEASE_DRUGS_QUERY = """
query($id: String!) {
    disease(efoId: $id) {
        drugAndClinicalCandidates {
            rows {
                id maxClinicalStage
                drug {
                    id name drugType
                    mechanismsOfAction {
                        rows {
                            mechanismOfAction actionType
                            targets { id approvedSymbol }
                        }
                    }
                }
                clinicalReports {
                    id source clinicalStage hasExpertReview
                    title type trialOverallStatus trialLiterature
                    drugs { drugFromSource drug { id name } }
                    diseases { diseaseFromSource disease { id name } }
                }
            }
        }
    }
}
"""

DISEASE_SYNONYMS_QUERY = """
query($id: String!) {
    disease(efoId: $id) {
        id
        name
        parents { name }
        synonyms {
            relation
            terms
        }
    }
}
"""
