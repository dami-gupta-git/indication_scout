"""Unit tests for DrugProfile Pydantic model."""

from indication_scout.models.model_drug_profile import DrugProfile


def test_drug_profile_coerce_nones_converts_null_lists_to_empty():
    """DrugProfile with list fields set to None must coerce them to []."""
    profile = DrugProfile(
        name="METFORMIN",
        synonyms=None,
        target_gene_symbols=None,
        mechanisms_of_action=None,
        atc_codes=None,
        atc_descriptions=None,
    )

    assert profile.synonyms == []
    assert profile.target_gene_symbols == []
    assert profile.mechanisms_of_action == []
    assert profile.atc_codes == []
    assert profile.atc_descriptions == []
