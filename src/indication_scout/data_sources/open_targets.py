"""
Open Targets Platform GraphQL client.

Two primary methods:
  1. get_drug        — Fetch drug data (indications, targets, warnings, adverse events)
  2. get_target_data — Fetch target data (associations, pathways, interactions, expression)

Plus convenience accessors for specific target data slices.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from indication_scout.constants import (
    CACHE_TTL,
    DEFAULT_CACHE_DIR,
    INTERACTION_TYPE_MAP,
    OPEN_TARGETS_BASE_URL,
    OPEN_TARGETS_PAGE_SIZE,
)
from indication_scout.markers import no_review
from indication_scout.utils.cache import cache_get, cache_set
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.helpers.drug_helpers import normalize_drug_name

from indication_scout.models.model_open_targets import (
    Association,
    Pathway,
    Interaction,
    DrugSummary,
    TissueExpression,
    MousePhenotype,
    TargetData,
    GeneticConstraint,
    AdverseEvent,
    CellTypeExpression,
    RNAExpression,
    BiologicalModel,
    DrugData,
    DrugTarget,
    ProteinExpression,
    DrugWarning,
    Indication,
    SafetyLiability,
    SafetyEffect,
    DiseaseSynonyms,
    RichDrugData,
)

logger = logging.getLogger(__name__)


class OpenTargetsClient(BaseClient):
    BASE_URL = OPEN_TARGETS_BASE_URL
    PAGE_SIZE = OPEN_TARGETS_PAGE_SIZE

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

    async def get_rich_drug_data(self, drug_name: str) -> RichDrugData:
        """Fetch drug data and all associated target data in parallel."""
        drug = await self.get_drug(drug_name)
        targets = await asyncio.gather(
            *[self.get_target_data(t.target_id) for t in drug.targets]
        )
        return RichDrugData(drug=drug, targets=list(targets))

    async def get_drug(self, drug_name: str) -> DrugData:
        """Fetch drug data by name, enriched with ATC codes from ChEMBL."""
        chembl_id = await self._resolve_drug_name(drug_name)

        cached = cache_get("drug", {"chembl_id": chembl_id}, self.cache_dir)
        if cached:
            return DrugData.model_validate(cached)

        async with ChEMBLClient() as chembl_client:
            drug_data, molecule = await asyncio.gather(
                self._fetch_drug(chembl_id),
                chembl_client.get_molecule(chembl_id),
                return_exceptions=True,
            )

        if isinstance(drug_data, BaseException):
            raise drug_data

        if isinstance(molecule, BaseException):
            logger.warning("ChEMBL ATC lookup failed for %s: %s", chembl_id, molecule)
        else:
            drug_data.atc_classifications = molecule.atc_classifications

        cache_set(
            "drug",
            {"chembl_id": chembl_id},
            drug_data.model_dump(),
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        return drug_data

    @no_review
    # TODO needs rework
    async def get_drug_competitors(self, name: str) -> dict[str, set[str]]:
        """Fetch phase-4 competitor drugs for bupropion, grouped by disease."""
        name = name.lower()
        drug = await self.get_drug(name)
        targets = drug.targets

        siblings: dict[str, set[str]] = {}

        for t in targets:
            logger.info(t.mechanism_of_action)
            summaries = await self.get_target_data_drug_summaries(t.target_id)
            # drugs = set([normalize_drug_name(s.drug_name.lower()) for s in summaries])
            # diseases = set([s.disease_name.lower() for s in summaries])
            for summary in summaries:
                if summary.phase >= 3:
                    disease = summary.disease_name
                    drug_name = normalize_drug_name(summary.drug_name)
                    if disease in siblings:
                        siblings[disease].add(drug_name)
                    else:
                        siblings[disease] = {drug_name}

        for key in list(siblings):
            val = siblings[key]
            if name in val:
                del siblings[key]

        sorted_data = dict(
            sorted(siblings.items(), key=lambda item: len(item[1]), reverse=True)
        )
        top_10 = dict(list(sorted_data.items())[:10])
        return top_10

    async def get_drug_indications(self, drug_name: str) -> list[Indication]:
        drug = await self.get_drug(drug_name)
        return drug.indications

    async def get_drug_target_competitors(
        self, drug_name: str
    ) -> dict[str, list[DrugSummary]]:
        """For each target of a drug, fetch all drugs acting on that target.

        Returns a dict mapping target symbol (e.g. "GLP1R") to the list of
        DrugSummary objects from Open Targets' knownDrugs for that target.
        """
        drug = await self.get_drug(drug_name)
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

    async def get_target_data_associations(
        self, target_id: str, min_score: float = 0.1
    ) -> list[Association]:
        target = await self.get_target_data(target_id)
        return [a for a in target.associations if a.overall_score >= min_score]

    async def get_target_data_pathways(self, target_id: str) -> list[Pathway]:
        target = await self.get_target_data(target_id)
        return target.pathways

    async def get_target_data_interactions(self, target_id: str) -> list[Interaction]:
        target = await self.get_target_data(target_id)
        return target.interactions

    async def get_target_data_drug_summaries(self, target_id: str) -> list[DrugSummary]:
        target = await self.get_target_data(target_id)
        return target.drug_summaries

    async def get_target_data_tissue_expression(
        self, target_id: str
    ) -> list[TissueExpression]:
        target = await self.get_target_data(target_id)
        return target.expressions

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

    async def get_disease_drugs(self, disease_id: str) -> list[DrugSummary]:
        """All drugs for a disease, any target, any mechanism."""
        cached = cache_get("disease_drugs", {"disease_id": disease_id}, self.cache_dir)
        if cached:
            return [DrugSummary.model_validate(d) for d in cached]

        data = await self._graphql(
            self.BASE_URL, DISEASE_DRUGS_QUERY, {"id": disease_id, "size": 200}
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
            disease_name=raw_disease["name"],
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

    async def _resolve_drug_name(self, name: str) -> str:
        """Search by name → return ChEMBL ID."""
        data = await self._graphql(self.BASE_URL, SEARCH_QUERY, {"q": name})
        hits = data["data"]["search"]["hits"]
        drug_hits = [h for h in hits if h["entity"] == "drug"]
        if not drug_hits:
            raise DataSourceError(
                self._source_name,
                f"No drug found for '{name}'",
            )
        return drug_hits[0]["id"]

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
        targets = []
        for row in (raw.get("mechanismsOfAction") or {}).get("rows", []):
            for t in row.get("targets", []):
                targets.append(
                    DrugTarget(
                        target_id=t["id"],
                        target_symbol=t["approvedSymbol"],
                        mechanism_of_action=row["mechanismOfAction"],
                        action_type=row.get("actionType"),
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
                disease_id=row["disease"]["id"],
                disease_name=row["disease"]["name"],
                max_phase=row["maxPhaseForIndication"],
                references=row.get("references", []),
            )
            for row in (raw.get("indications") or {}).get("rows", [])
        ]

        adverse_events = [
            self._parse_adverse_event(ae)
            for ae in (raw.get("adverseEvents") or {}).get("rows", [])
        ]

        return DrugData(
            chembl_id=raw["id"],
            name=raw["name"],
            synonyms=raw.get("synonyms", []),
            trade_names=raw.get("tradeNames", []),
            drug_type=raw.get("drugType"),
            is_approved=raw.get("isApproved"),
            max_clinical_phase=raw.get("maximumClinicalTrialPhase"),
            year_first_approved=raw.get("yearOfFirstApproval"),
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
                for d in (raw.get("knownDrugs") or {}).get("rows", [])
            ],
            expressions=[self._parse_expression(e) for e in raw.get("expressions", [])],
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
        disease = raw["disease"]
        therapeutic_areas = [ta["name"] for ta in disease.get("therapeuticAreas", [])]
        return Association(
            disease_id=disease["id"],
            disease_name=disease["name"],
            overall_score=raw["score"],
            datatype_scores=datatype_scores,
            therapeutic_areas=therapeutic_areas,
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
        return DrugSummary(
            drug_id=raw.get("drugId", ""),
            drug_name=raw.get("prefName", ""),
            disease_id=raw.get("diseaseId", ""),
            disease_name=raw.get("label", ""),
            phase=raw.get("phase", 0),
            status=raw.get("status"),
            mechanism_of_action=raw.get("mechanismOfAction", ""),
            clinical_trial_ids=raw.get("ctIds", []),
        )

    def _parse_expression(self, raw: dict) -> TissueExpression:
        tissue = raw.get("tissue", {})
        rna = raw.get("rna", {})
        protein = raw.get("protein", {})
        anatomical_systems = tissue.get("anatomicalSystems", [])
        cell_types = [
            CellTypeExpression(
                name=ct["name"],
                level=ct["level"],
                reliability=ct.get("reliability", False),
            )
            for ct in protein.get("cellType", []) or []
        ]
        return TissueExpression(
            tissue_id=tissue.get("id", ""),
            tissue_name=tissue.get("label", ""),
            tissue_anatomical_system=(
                anatomical_systems[0] if anatomical_systems else ""
            ),
            rna=RNAExpression(
                value=rna.get("value"),
                quantile=rna.get("level"),
                unit=rna.get("unit"),
            ),
            protein=ProteinExpression(
                level=protein.get("level"),
                reliability=protein.get("reliability"),
                cell_types=cell_types,
            ),
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
            meddra_code=raw.get("meddraCode"),
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
            oe=raw.get("oe"),
            oe_lower=raw.get("oeLower"),
            oe_upper=raw.get("oeUpper"),
            score=raw.get("score"),
            upper_bin=raw.get("upperBin"),
        )

    def _parse_disease_drugs(self, data: dict) -> list[DrugSummary]:
        """Parse and deduplicate disease drugs — one entry per drug, highest phase wins."""
        disease = data.get("disease") or {}
        rows = disease.get("knownDrugs", {}).get("rows", [])

        by_drug: dict[str, DrugSummary] = {}
        for row in rows:
            drug = self._parse_drug_summary(row)
            if drug.drug_id not in by_drug or drug.phase > by_drug[drug.drug_id].phase:
                by_drug[drug.drug_id] = drug
        return list(by_drug.values())


# ------------------------------------------------------------------
# GraphQL queries
# ------------------------------------------------------------------

SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
        hits { id entity }
    }
}
"""

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
        id name synonyms tradeNames drugType
        isApproved maximumClinicalTrialPhase yearOfFirstApproval

        mechanismsOfAction {
            rows {
                mechanismOfAction actionType
                targets { id approvedSymbol }
            }
        }

        indications {
            approvedIndications
            rows {
                maxPhaseForIndication
                disease { id name }
                references { source ids }
            }
        }

        drugWarnings {
            warningType description toxicityClass
            country year efoId efoTerm
        }

        adverseEvents(page: {index: 0, size: 100}) {
            rows { name meddraCode count logLR }
            criticalValue
        }
    }
}
"""

TARGET_QUERY = """
query($id: String!) {
    target(ensemblId: $id) {
        id approvedSymbol approvedName

        associatedDiseases(page: {index: 0, size: 500}) {
            rows {
                disease {
                    id name
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
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

        knownDrugs(size: 200) {
            rows {
                drugId prefName diseaseId label
                phase status mechanismOfAction ctIds
            }
        }

        expressions {
            tissue {
                id label
                anatomicalSystems
            }
            rna { value unit level }
            protein {
                level reliability
                cellType { name level reliability }
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
                    id name
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
            }
        }
    }
}
"""

DISEASE_DRUGS_QUERY = """
query($id: String!, $size: Int!) {
    disease(efoId: $id) {
        knownDrugs(size: $size) {
            rows {
                drugId prefName targetId approvedSymbol
                diseaseId label phase status
                mechanismOfAction ctIds
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
