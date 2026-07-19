"""Unit tests for DrugProfile Pydantic model."""

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_open_targets import (
    AdverseEvent,
    DrugData,
    DrugWarning,
    RichDrugData,
)


def test_drug_profile_coerce_nones_converts_null_lists_to_empty():
    """DrugProfile with list fields set to None must coerce them to []."""
    profile = DrugProfile(
        chembl_id="CHEMBL1431",
        target_gene_symbols=None,
        mechanisms_of_action=None,
        atc_codes=None,
        atc_descriptions=None,
        drug_warnings=None,
        adverse_events=None,
    )

    assert profile.target_gene_symbols == []
    assert profile.mechanisms_of_action == []
    assert profile.atc_codes == []
    assert profile.atc_descriptions == []
    assert profile.drug_warnings == []
    assert profile.adverse_events == []


def test_from_rich_drug_data_carries_opentargets_safety_signal():
    """from_rich_drug_data carries OT drugWarnings + adverseEvents onto the profile (previously
    fetched then discarded)."""
    rich = RichDrugData(
        drug=DrugData(
            chembl_id="CHEMBL1477",
            drug_type="Small molecule",
            warnings=[
                DrugWarning(
                    warning_type="Withdrawn",
                    description="increased risk of rhabdomyolysis",
                    toxicity_class="musculoskeletal toxicity",
                    year=2001,
                ),
            ],
            adverse_events=[
                AdverseEvent(name="myalgia", count=5, log_likelihood_ratio=13.2),
            ],
        ),
        targets=[],
    )

    profile = DrugProfile.from_rich_drug_data(rich)

    assert len(profile.drug_warnings) == 1
    assert profile.drug_warnings[0].warning_type == "Withdrawn"
    assert profile.drug_warnings[0].description == "increased risk of rhabdomyolysis"
    assert profile.drug_warnings[0].toxicity_class == "musculoskeletal toxicity"
    assert profile.drug_warnings[0].year == 2001
    assert len(profile.adverse_events) == 1
    assert profile.adverse_events[0].name == "myalgia"
    assert profile.adverse_events[0].log_likelihood_ratio == 13.2


def test_from_rich_drug_data_empty_safety_when_ot_has_none():
    """No OT warnings/adverse events -> empty lists (no fabrication)."""
    rich = RichDrugData(
        drug=DrugData(chembl_id="CHEMBL25", drug_type="Small molecule"),
        targets=[],
    )

    profile = DrugProfile.from_rich_drug_data(rich)

    assert profile.drug_warnings == []
    assert profile.adverse_events == []
