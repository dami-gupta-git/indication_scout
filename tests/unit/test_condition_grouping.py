"""Unit tests for Europe PMC condition grouping."""

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.services import condition_grouping
from indication_scout.services.condition_grouping import group_conditions


def _merge_returning(merge, remove):
    """Stub merge_duplicate_diseases with a fixed result, ignoring its arguments."""

    async def _stub(diseases, drug_indications, max_tokens=None):
        return {"merge": merge, "remove": remove}

    return _stub


async def test_groups_aliases_under_canonical(monkeypatch):
    """Aliases collapse into one candidate carrying every source article."""
    monkeypatch.setattr(
        condition_grouping,
        "merge_duplicate_diseases",
        _merge_returning({"alopecia areata": ["patchy hair loss"]}, []),
    )

    grouped = await group_conditions(
        {
            "alopecia areata": ["MED:1", "MED:2"],
            "patchy hair loss": ["PPR:9"],
            "psoriasis": ["MED:3"],
        },
        approved_indications=[],
    )

    assert len(grouped) == 2

    alopecia = grouped[0]
    assert alopecia.name == "alopecia areata"
    assert alopecia.aliases == ["patchy hair loss"]
    assert alopecia.article_keys == ["MED:1", "MED:2", "PPR:9"]
    assert alopecia.paper_count == 3

    psoriasis = grouped[1]
    assert psoriasis.name == "psoriasis"
    assert psoriasis.aliases == []
    assert psoriasis.article_keys == ["MED:3"]
    assert psoriasis.paper_count == 1


async def test_removes_approved_indications(monkeypatch):
    """A condition the drug is already approved for is dropped, along with its aliases."""
    monkeypatch.setattr(
        condition_grouping,
        "merge_duplicate_diseases",
        _merge_returning(
            {"rheumatoid arthritis": ["RA"]}, ["rheumatoid arthritis"]
        ),
    )

    grouped = await group_conditions(
        {
            "rheumatoid arthritis": ["MED:1"],
            "RA": ["MED:2"],
            "uveitis": ["MED:3"],
        },
        approved_indications=["rheumatoid arthritis"],
    )

    assert len(grouped) == 1
    assert grouped[0].name == "uveitis"
    assert grouped[0].aliases == []
    assert grouped[0].article_keys == ["MED:3"]


async def test_post_cutoff_approval_is_not_removed(monkeypatch):
    """Holdout case: colchicine cut at 2008 must keep atherosclerotic cardiovascular disease,
    approved only in 2022. The cutoff-resolved list omits it, so nothing removes it."""
    monkeypatch.setattr(
        condition_grouping,
        "merge_duplicate_diseases",
        _merge_returning({}, ["gout"]),
    )

    grouped = await group_conditions(
        {
            "gout": ["MED:1"],
            "atherosclerotic cardiovascular disease": ["MED:2", "MED:3"],
        },
        approved_indications=["gout"],
    )

    assert len(grouped) == 1
    assert grouped[0].name == "atherosclerotic cardiovascular disease"
    assert grouped[0].aliases == []
    assert grouped[0].article_keys == ["MED:2", "MED:3"]


async def test_empty_extraction_input_makes_no_llm_call(monkeypatch):
    """No conditions means no merge call at all."""

    async def _fail(*args, **kwargs):
        raise AssertionError("merge_duplicate_diseases must not be called")

    monkeypatch.setattr(condition_grouping, "merge_duplicate_diseases", _fail)

    assert await group_conditions({}, approved_indications=["gout"]) == []


async def test_unparseable_merge_propagates(monkeypatch):
    """A failed merge must not degrade to 'nothing removed' — approved indications would then
    surface as novel candidates."""

    async def _raise(diseases, drug_indications, max_tokens=None):
        raise DataSourceError("llm", "unparseable response")

    monkeypatch.setattr(condition_grouping, "merge_duplicate_diseases", _raise)

    with pytest.raises(DataSourceError):
        await group_conditions({"gout": ["MED:1"]}, approved_indications=["gout"])


async def test_ignores_aliases_absent_from_input(monkeypatch):
    """An alias the LLM invents is not ours to map; the real name still groups."""
    monkeypatch.setattr(
        condition_grouping,
        "merge_duplicate_diseases",
        _merge_returning(
            {"chronic pain": ["neuropathic pain", "a name never extracted"]}, []
        ),
    )

    grouped = await group_conditions(
        {"chronic pain": ["MED:1"], "neuropathic pain": ["MED:2"]},
        approved_indications=[],
    )

    assert len(grouped) == 1
    assert grouped[0].name == "chronic pain"
    assert grouped[0].aliases == ["neuropathic pain"]
    assert grouped[0].article_keys == ["MED:1", "MED:2"]
