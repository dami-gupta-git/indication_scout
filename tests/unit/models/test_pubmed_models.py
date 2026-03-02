"""Unit tests for PubMed Pydantic models."""

from indication_scout.models.model_pubmed_abstract import PubmedAbstract


def test_pubmed_abstract_coerce_nones_converts_null_lists_to_empty():
    """PubmedAbstract with list fields set to None must coerce them to []."""
    abstract = PubmedAbstract(
        pmid="12345678",
        authors=None,
        mesh_terms=None,
        keywords=None,
    )

    assert abstract.authors == []
    assert abstract.mesh_terms == []
    assert abstract.keywords == []


def test_pubmed_abstract_coerce_nones_preserves_genuine_nones():
    """PubmedAbstract fields with default=None must stay None when passed None."""
    abstract = PubmedAbstract(
        pmid="12345678",
        title=None,
        abstract=None,
        journal=None,
        pub_date=None,
    )

    assert abstract.title is None
    assert abstract.abstract is None
    assert abstract.journal is None
    assert abstract.pub_date is None
