"""Unit tests for Europe PMC condition extraction."""

import pytest

from indication_scout.models.model_europe_pmc import EuropePMCArticle
from indication_scout.services.condition_extraction import (
    build_prompt,
    parse_response,
)

ARTICLE = EuropePMCArticle(
    source="MED",
    record_id="12345678",
    pmid="12345678",
    doi="10.1000/example",
    title="Duloxetine for fibromyalgia: a randomized trial.",
    abstract="Patients with fibromyalgia received duloxetine for 12 weeks.",
    journal="Journal of Pain",
    first_publication_date="2005-06-01",
    pub_year=2005,
    pub_types=["Journal Article"],
    cited_by_count=42,
    is_open_access=False,
)


def test_build_prompt_includes_drug_title_and_abstract():
    prompt = build_prompt("duloxetine", ARTICLE)

    assert "duloxetine" in prompt
    assert "Duloxetine for fibromyalgia: a randomized trial." in prompt
    assert "Patients with fibromyalgia received duloxetine for 12 weeks." in prompt
    assert "Do NOT use any knowledge about duloxetine beyond this text" in prompt


@pytest.mark.parametrize(
    "response, expected",
    [
        ("fibromyalgia", ["fibromyalgia"]),
        ("NONE", []),
        ("  none  ", []),
        (
            "generalized anxiety disorder\nmajor depressive disorder",
            ["generalized anxiety disorder", "major depressive disorder"],
        ),
        ("- fibromyalgia\n• chronic pain", ["fibromyalgia", "chronic pain"]),
    ],
)
def test_parse_response(response, expected):
    assert parse_response(response) == expected


def test_parse_response_dedups_preserving_order():
    """The same condition often appears in both title and abstract."""
    parsed = parse_response("fibromyalgia\nchronic pain\nFibromyalgia")
    assert parsed == ["fibromyalgia", "chronic pain"]


def test_parse_response_discards_explanatory_prose():
    """The model sometimes explains its reasoning instead of answering; a sentence is not a
    condition name. Observed on real abstracts, hence the word-count guard."""
    response = (
        "the paper describes methotrexate-induced epidermal necrosis as an adverse effect "
        "in a patient with erythrodermic psoriasis, so the drug was not being tested as a "
        "treatment for that condition"
    )
    assert parse_response(response) == []


def test_parse_response_keeps_long_but_plausible_condition_names():
    """Real condition names can be several words and must survive the prose guard."""
    parsed = parse_response(
        "chronic obstructive pulmonary disease exacerbation\nstress urinary incontinence in women"
    )
    assert parsed == [
        "chronic obstructive pulmonary disease exacerbation",
        "stress urinary incontinence in women",
    ]


def test_parse_response_mixed_none_and_conditions():
    """A stray NONE line alongside real conditions drops only that line."""
    assert parse_response("fibromyalgia\nNONE\nchronic pain") == [
        "fibromyalgia",
        "chronic pain",
    ]
