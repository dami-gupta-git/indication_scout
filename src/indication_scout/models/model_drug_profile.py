"""Pydantic model for a drug's profile used by the RAG pipeline."""

from pydantic import BaseModel, model_validator

from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_open_targets import RichDrugData


class DrugProfile(BaseModel):
    """Structured drug profile consumed by expand_search_terms.

    Built from RichDrugData + pre-fetched ATCDescription objects via
    the from_rich_drug_data factory classmethod.
    """

    name: str = ""
    synonyms: list[str] = []
    target_gene_symbols: list[str] = []
    mechanisms_of_action: list[str] = []
    atc_codes: list[str] = []
    atc_descriptions: list[str] = []
    drug_type: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

    @classmethod
    def from_rich_drug_data(
        cls,
        rich: RichDrugData,
        atc_descriptions: list[ATCDescription] | None = None,
    ) -> "DrugProfile":
        """Build a DrugProfile from a RichDrugData and optional pre-fetched ATC descriptions.

        Args:
            rich: Combined drug + target data from Open Targets.
            atc_descriptions: ATCDescription objects for each ATC code on the drug,
                fetched by the caller via ChEMBLClient.get_atc_description.
                If None or empty, atc_descriptions on the profile will be [].
        """
        drug = rich.drug
        return cls(
            name=drug.name if drug else "",
            synonyms=list(
                dict.fromkeys(
                    s for s in (drug.synonyms + drug.trade_names if drug else []) if s
                )
            ),
            target_gene_symbols=list(
                dict.fromkeys(t.symbol for t in rich.targets if t.symbol)
            ),
            mechanisms_of_action=list(
                dict.fromkeys(
                    dt.mechanism_of_action
                    for dt in (drug.targets if drug else [])
                    if dt.mechanism_of_action
                )
            ),
            atc_codes=drug.atc_classifications if drug else [],
            # Only level3 and level4 descriptions are included. level1/level2 are too
            # broad to generate useful PubMed queries (e.g. "ALIMENTARY TRACT AND
            # METABOLISM AND colon") and are intentionally excluded.
            atc_descriptions=list(
                dict.fromkeys(
                    desc
                    for atc in (atc_descriptions or [])
                    for desc in (atc.level3_description, atc.level4_description)
                    if desc
                )
            ),
            drug_type=drug.drug_type if drug else "",
        )
