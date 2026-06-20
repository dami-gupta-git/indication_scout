"""Unit tests for services/literature_strength — drug-specific strength parsing, enum
validation, caching, and empty-set / parse-failure handling. The live LLM call is mocked; the
live judgment (class-level ≠ strong) is exercised in the integration test.
"""

from unittest.mock import AsyncMock, patch

from indication_scout.services.literature_strength import (
    LiteratureStrength,
    _parse_strength,
    judge_literature_strength,
)

_ABS = [{"pmid": "111", "title": "T", "abstract": "A"}]
_PMIDS = ["111"]

_OK = (
    '{"verdicts": {"111": "relevant"}, '
    '"evidence_basis": "drug_specific", "strength": "strong", "direction": "supports", '
    '"is_observational": false, "reason": "two RCTs of this drug"}'
)


def test_parse_strength_plain_json():
    s = _parse_strength(_OK, input_pmids=_PMIDS)
    assert s == LiteratureStrength(
        strength="strong",
        direction="supports",
        evidence_basis="drug_specific",
        is_observational=False,
        relevant_pmids=("111",),
        contaminated_pmids=(),
    )


def test_parse_strength_fenced_json():
    s = _parse_strength(
        '```json\n{"verdicts": {"111": "contaminated"}, '
        '"evidence_basis": "class_level", "strength": "none", '
        '"direction": "none", "is_observational": null}\n```',
        input_pmids=_PMIDS,
    )
    assert s.evidence_basis == "class_level"
    assert s.strength == "none"
    assert s.direction == "none"
    assert s.is_observational is None
    assert s.relevant_pmids == ()
    assert s.contaminated_pmids == ("111",)


def test_parse_strength_per_pmid_split():
    """Mixed verdict map splits the input PMIDs into relevant / contaminated over the input set."""
    s = _parse_strength(
        '{"verdicts": {"111": "relevant", "222": "contaminated"}, '
        '"evidence_basis": "drug_specific", "strength": "moderate", '
        '"direction": "supports", "is_observational": true}',
        input_pmids=["111", "222"],
    )
    assert s.relevant_pmids == ("111",)
    assert s.contaminated_pmids == ("222",)
    assert s.evidence_basis == "drug_specific"
    assert s.strength == "moderate"


def test_parse_strength_omitted_pmid_is_contaminated():
    """An input PMID missing from the verdicts map is conservatively treated as contaminated."""
    s = _parse_strength(
        '{"verdicts": {"111": "relevant"}, '
        '"evidence_basis": "drug_specific", "strength": "weak", '
        '"direction": "supports", "is_observational": true}',
        input_pmids=["111", "999"],
    )
    assert s.relevant_pmids == ("111",)
    assert s.contaminated_pmids == ("999",)


def test_parse_strength_empty_relevant_with_drug_specific_is_none():
    """A drug_specific grade with no surviving relevant abstract is inconsistent -> None."""
    assert (
        _parse_strength(
            '{"verdicts": {"111": "contaminated"}, '
            '"evidence_basis": "drug_specific", "strength": "strong", '
            '"direction": "supports", "is_observational": false}',
            input_pmids=_PMIDS,
        )
        is None
    )


def test_parse_strength_missing_verdicts_is_none():
    assert (
        _parse_strength(
            '{"evidence_basis": "drug_specific", "strength": "strong", '
            '"direction": "supports", "is_observational": false}',
            input_pmids=_PMIDS,
        )
        is None
    )


def test_parse_strength_unknown_basis_is_none():
    assert (
        _parse_strength(
            '{"verdicts": {"111": "relevant"}, "evidence_basis": "indirect", '
            '"strength": "strong", "direction": "supports", "is_observational": false}',
            input_pmids=_PMIDS,
        )
        is None
    )


def test_parse_strength_unknown_strength_is_none():
    assert (
        _parse_strength(
            '{"verdicts": {"111": "relevant"}, "evidence_basis": "drug_specific", '
            '"strength": "very strong", "direction": "supports", "is_observational": false}',
            input_pmids=_PMIDS,
        )
        is None
    )


def test_parse_strength_non_bool_is_observational_is_none():
    assert (
        _parse_strength(
            '{"verdicts": {"111": "relevant"}, "evidence_basis": "drug_specific", '
            '"strength": "moderate", "direction": "supports", "is_observational": "yes"}',
            input_pmids=_PMIDS,
        )
        is None
    )


def test_parse_strength_garbage_is_none():
    assert _parse_strength("not json at all", input_pmids=_PMIDS) is None


def test_parse_strength_class_level_forces_none_strength_and_direction():
    """ENFORCED invariant: a class_level (or none) basis carries no drug strength, even if the
    model returns one — every consumer (incl. the supervisor ranking path) depends on this.
    """
    s = _parse_strength(
        '{"verdicts": {"111": "contaminated"}, "evidence_basis": "class_level", '
        '"strength": "moderate", "direction": "supports", "is_observational": false}',
        input_pmids=_PMIDS,
    )
    assert s.evidence_basis == "class_level"
    assert s.strength == "none"
    assert s.direction == "none"


def test_parse_strength_basis_none_forces_none_strength_and_direction():
    s = _parse_strength(
        '{"verdicts": {"111": "contaminated"}, "evidence_basis": "none", '
        '"strength": "weak", "direction": "mixed", "is_observational": null}',
        input_pmids=_PMIDS,
    )
    assert s.evidence_basis == "none"
    assert s.strength == "none"
    assert s.direction == "none"


async def test_judge_empty_abstracts_returns_none_without_llm(tmp_path):
    """No abstracts → None, and the LLM is never called."""
    with patch(
        "indication_scout.services.literature_strength.query_llm", new=AsyncMock()
    ) as mock_llm:
        s = await judge_literature_strength(
            [], drug="d", indication="i", cache_dir=tmp_path
        )
    assert s is None
    mock_llm.assert_not_awaited()


async def test_judge_returns_parsed_judgment(tmp_path):
    with patch(
        "indication_scout.services.literature_strength.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        s = await judge_literature_strength(
            _ABS, drug="d", indication="i", cache_dir=tmp_path
        )
    assert s.evidence_basis == "drug_specific"
    assert s.strength == "strong"
    assert s.direction == "supports"
    assert s.is_observational is False


async def test_judge_parse_failure_returns_none(tmp_path):
    """An unparseable response returns None (caller keeps the synthesize values)."""
    with patch(
        "indication_scout.services.literature_strength.query_llm",
        new=AsyncMock(return_value="the model rambled without JSON"),
    ):
        s = await judge_literature_strength(
            _ABS, drug="d", indication="i", cache_dir=tmp_path
        )
    assert s is None


async def test_judge_caches_and_does_not_recall_llm(tmp_path):
    """Second call with the same PMID set hits the cache — the LLM is called once and all
    fields round-trip."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.literature_strength.query_llm", new=mock):
        first = await judge_literature_strength(
            _ABS, drug="d", indication="i", cache_dir=tmp_path
        )
        second = await judge_literature_strength(
            _ABS, drug="d", indication="i", cache_dir=tmp_path
        )
    assert first == second
    assert second.evidence_basis == "drug_specific"
    assert second.strength == "strong"
    assert mock.await_count == 1


async def test_judge_cache_key_is_pmid_order_independent(tmp_path):
    """Two orderings of the same abstract set share one cache entry."""
    a = [
        {"pmid": "111", "title": "T1", "abstract": "A1"},
        {"pmid": "222", "title": "T2", "abstract": "A2"},
    ]
    b = list(reversed(a))
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.literature_strength.query_llm", new=mock):
        await judge_literature_strength(a, drug="d", indication="i", cache_dir=tmp_path)
        await judge_literature_strength(b, drug="d", indication="i", cache_dir=tmp_path)
    assert mock.await_count == 1


def test_parse_strength_approved_forces_none_strength_and_direction():
    """The new 'approved' basis (paper studies an approved sub-indication) carries no repurposing
    strength, even if the model returns one — same enforced invariant as class_level/none."""
    s = _parse_strength(
        '{"evidence_basis": "approved", "strength": "strong", '
        '"direction": "supports", "is_observational": true}'
    )
    assert s.evidence_basis == "approved"
    assert s.strength == "none"
    assert s.direction == "none"
    assert s.is_observational is True


async def test_judge_cache_key_includes_approved_indications(tmp_path):
    """The same PMID set under DIFFERENT approved lists must not collide — the approved set is
    part of the cache key, so each list triggers its own LLM call."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.literature_strength.query_llm", new=mock):
        await judge_literature_strength(
            _ABS, drug="d", indication="i", cache_dir=tmp_path, approved_indications=[]
        )
        await judge_literature_strength(
            _ABS,
            drug="d",
            indication="i",
            cache_dir=tmp_path,
            approved_indications=["major depressive disorder"],
        )
    assert mock.await_count == 2
