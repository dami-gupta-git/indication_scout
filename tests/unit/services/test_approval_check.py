"""Unit tests for services/approval_check — no network, no LLM calls."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.services.approval_check import (
    _coerce_label,
    _load_drug_approvals_table,
    _parse_approval_decisions,
    _table_as_label_text,
    extract_approved_from_labels,
    get_approved_indications,
    get_combination_only_indications,
    get_fda_approved_disease_mapping,
    list_approved_indications_at,
    list_approved_indications_from_labels,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("approved", "approved"),
        ("combination_only", "combination_only"),
        ("contaminated", "contaminated"),
        ("none", "none"),
        # A stale bool-shaped cache value must be REJECTED, never coerced to a
        # truthy "approved" — this is the regression the label cache guards.
        (True, None),
        ("bogus", None),
    ],
)
def test_coerce_label(value, expected):
    assert _coerce_label(value) == expected


def test_parse_approval_decisions_accepts_structured_labels_and_exact_anchor():
    parsed = {
        "inflammatory bowel disease": {
            "label": "contaminated",
            "matched_approved_indication": "Crohn disease",
            "reason": "Inflammatory bowel disease contains Crohn disease.",
        },
        "ulcerative colitis": {
            "label": "none",
            "reason": "Ulcerative colitis is a sibling of Crohn disease.",
        },
    }

    result = _parse_approval_decisions(
        parsed,
        ["inflammatory bowel disease", "ulcerative colitis"],
        ["Crohn disease"],
    )

    assert result == {
        "inflammatory bowel disease": "contaminated",
        "ulcerative colitis": "none",
    }


def test_parse_approval_decisions_rejects_unverified_anchor_and_invalid_shape():
    parsed = {
        "inflammatory bowel disease": {
            "label": "contaminated",
            "matched_approved_indication": "ulcerative colitis",
            "reason": "The candidate contains ulcerative colitis.",
        },
        "celiac disease": "none",
    }

    result = _parse_approval_decisions(
        parsed,
        ["inflammatory bowel disease", "celiac disease"],
        ["Crohn disease"],
    )

    assert result == {}


async def test_sildenafil_hypertension_uses_curated_contamination(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        result = await get_fda_approved_disease_mapping(
            drug_name="sildenafil",
            candidate_diseases=["hypertension"],
            approved_indications=["pulmonary arterial hypertension"],
            cache_dir=tmp_path,
        )

    assert result == {"hypertension": "contaminated"}
    mock_llm.assert_not_awaited()


async def test_missing_label_falls_back_to_the_table_as_label_text(tmp_path):
    """Baricitinib, October 2026: openFDA's name index lost the Olumiant label. The table entries stand in for the label
    text in the same prompt, so the model judges every candidate; those verdicts are not cached."""
    answer = json.dumps(
        {
            "alopecia areata": {"reason": "listed", "label": "approved"},
            "psoriasis": {"reason": "not listed", "label": "none"},
        }
    )
    llm_mock = AsyncMock(return_value=answer)
    with (
        patch(
            "indication_scout.services.approval_check.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2105759"),
        ),
        patch(
            "indication_scout.services.approval_check.get_all_drug_names",
            new=AsyncMock(return_value=["baricitinib"]),
        ),
        patch(
            "indication_scout.services.approval_check.FDAClient.get_all_label_indications",
            new=AsyncMock(return_value=[]),
        ),
        patch("indication_scout.services.approval_check.query_llm", new=llm_mock),
    ):
        result = await get_fda_approved_disease_mapping(
            drug_name="baricitinib",
            candidate_diseases=["alopecia areata", "psoriasis"],
            approved_indications=["rheumatoid arthritis", "alopecia areata"],
            cache_dir=tmp_path,
        )

    assert result == {"alopecia areata": "approved", "psoriasis": "none"}
    prompt = llm_mock.await_args.args[0]
    assert (
        "INDICATIONS AND USAGE (baricitinib, from a curated FDA approvals record):\n"
        "- rheumatoid arthritis\n"
        "- severe covid-19 hospitalization\n"
        "- alopecia areata"
    ) in prompt
    assert _cached_labels(tmp_path) == []


def test_table_as_label_text_names_the_combination_partner():
    assert _table_as_label_text("bupropion", date(2026, 11, 25)) == (
        "INDICATIONS AND USAGE (bupropion, from a curated FDA approvals record):\n"
        "- major depressive disorder\n"
        "- smoking cessation\n"
        "- seasonal affective disorder\n"
        "- obesity (approved only in a fixed-dose combination product with naltrexone)\n"
        "- alzheimer's disease agitation (approved only in a fixed-dose combination product with dextromethorphan)"
    )


def test_table_as_label_text_is_none_for_a_drug_not_in_the_table():
    assert _table_as_label_text("atabecestat", date(2026, 11, 25)) is None


def _mapping_patches(
    label_texts=None, label_error=None, llm_response=None, alias_error=None
):
    """Patch everything around the label judgment: ChEMBL aliases, FDA label fetch, model call."""
    fda_mock = (
        AsyncMock(side_effect=label_error)
        if label_error is not None
        else AsyncMock(return_value=label_texts)
    )
    alias_mock = (
        AsyncMock(side_effect=alias_error)
        if alias_error is not None
        else AsyncMock(return_value=["sildenafil"])
    )
    llm_mock = AsyncMock(return_value=llm_response)
    patches = (
        patch(
            "indication_scout.services.approval_check.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL192"),
        ),
        patch(
            "indication_scout.services.approval_check.get_all_drug_names",
            new=alias_mock,
        ),
        patch(
            "indication_scout.services.approval_check.FDAClient.get_all_label_indications",
            new=fda_mock,
        ),
        patch("indication_scout.services.approval_check.query_llm", new=llm_mock),
    )
    return patches, llm_mock, fda_mock


async def _run_mapping(tmp_path, patches, candidates, drug_name="sildenafil"):
    for p in patches:
        p.start()
    try:
        return await get_fda_approved_disease_mapping(
            drug_name=drug_name,
            candidate_diseases=candidates,
            approved_indications=["pulmonary arterial hypertension"],
            cache_dir=tmp_path,
        )
    finally:
        for p in patches:
            p.stop()


def _cached_labels(tmp_path):
    files = list((tmp_path / "fda_drug_disease_approval").glob("*.json"))
    return [] if not files else json.loads(files[0].read_text())["entries"]


async def test_mapping_unresolved_when_no_label_text_and_no_table_entry(tmp_path):
    patches, llm_mock, _ = _mapping_patches(label_texts=[])
    result = await _run_mapping(
        tmp_path, patches, ["eisenmenger syndrome", "raynaud disease"], "atabecestat"
    )
    assert result == {
        "eisenmenger syndrome": "unresolved",
        "raynaud disease": "unresolved",
    }
    llm_mock.assert_not_awaited()
    assert _cached_labels(tmp_path) == []


async def test_mapping_unresolved_when_label_fetch_fails(tmp_path):
    patches, llm_mock, _ = _mapping_patches(
        label_error=DataSourceError("fda", "openFDA down")
    )
    result = await _run_mapping(
        tmp_path, patches, ["eisenmenger syndrome", "raynaud disease"]
    )
    assert result == {
        "eisenmenger syndrome": "unresolved",
        "raynaud disease": "unresolved",
    }
    llm_mock.assert_not_awaited()
    assert _cached_labels(tmp_path) == []


async def test_mapping_unresolved_when_model_answer_is_unparseable(tmp_path):
    patches, _, _ = _mapping_patches(
        label_texts=["INDICATIONS AND USAGE REVATIO is indicated for PAH"],
        llm_response="this is not json at all",
    )
    result = await _run_mapping(
        tmp_path, patches, ["eisenmenger syndrome", "raynaud disease"]
    )
    assert result == {
        "eisenmenger syndrome": "unresolved",
        "raynaud disease": "unresolved",
    }
    assert _cached_labels(tmp_path) == []


async def test_mapping_unresolved_for_candidate_the_answer_omits(tmp_path):
    answer = json.dumps(
        {"raynaud disease": {"reason": "not on the label", "label": "none"}}
    )
    patches, _, _ = _mapping_patches(
        label_texts=["INDICATIONS AND USAGE REVATIO is indicated for PAH"],
        llm_response=answer,
    )
    result = await _run_mapping(
        tmp_path, patches, ["eisenmenger syndrome", "raynaud disease"]
    )
    assert result == {"eisenmenger syndrome": "unresolved", "raynaud disease": "none"}
    cached = _cached_labels(tmp_path)
    assert set(cached) == {"raynaud disease"}
    assert cached["raynaud disease"]["label"] == "none"


async def test_mapping_judged_none_is_cached_and_not_unresolved(tmp_path):
    answer = json.dumps(
        {"raynaud disease": {"reason": "not on the label", "label": "none"}}
    )
    patches, llm_mock, _ = _mapping_patches(
        label_texts=["INDICATIONS AND USAGE REVATIO is indicated for PAH"],
        llm_response=answer,
    )
    first = await _run_mapping(tmp_path, patches, ["raynaud disease"])
    assert first == {"raynaud disease": "none"}
    assert _cached_labels(tmp_path)["raynaud disease"]["label"] == "none"

    patches2, llm_mock2, _ = _mapping_patches(label_texts=[], llm_response="unused")
    second = await _run_mapping(tmp_path, patches2, ["raynaud disease"])
    assert second == {"raynaud disease": "none"}  # served from the per-drug cache
    llm_mock2.assert_not_awaited()


async def test_mapping_alias_lookup_failure_keeps_cached_verdicts_and_leaves_the_rest_unresolved(tmp_path):
    """A ChEMBL failure must not fall back to a bare-name label search: that can miss a formulation's label (sildenafil
    without Revatio) and cache a conclusive "none". Cached verdicts are kept; uncached candidates stay unresolved."""
    answer = json.dumps(
        {"raynaud disease": {"reason": "not on the label", "label": "none"}}
    )
    patches, _, _ = _mapping_patches(
        label_texts=["INDICATIONS AND USAGE REVATIO is indicated for PAH"],
        llm_response=answer,
    )
    await _run_mapping(tmp_path, patches, ["raynaud disease"])

    patches2, llm_mock2, fda_mock2 = _mapping_patches(
        label_texts=["INDICATIONS AND USAGE VIAGRA is indicated for erectile dysfunction"],
        llm_response="unused",
        alias_error=RuntimeError("chembl down"),
    )
    result = await _run_mapping(
        tmp_path, patches2, ["raynaud disease", "eisenmenger syndrome"]
    )
    assert result == {"raynaud disease": "none", "eisenmenger syndrome": "unresolved"}
    fda_mock2.assert_not_awaited()
    llm_mock2.assert_not_awaited()
    assert set(_cached_labels(tmp_path)) == {"raynaud disease"}


async def test_mapping_ignores_verdicts_cached_before_salt_form_checks(
    tmp_path: Path,
) -> None:
    patches, _, _ = _mapping_patches(
        label_texts=["REVATIO is indicated for pulmonary arterial hypertension"],
        llm_response=json.dumps(
            {"raynaud disease": {"reason": "not on the label", "label": "none"}}
        ),
    )
    await _run_mapping(tmp_path, patches, ["raynaud disease"])
    (cache_path,) = (tmp_path / "fda_drug_disease_approval").glob("*.json")
    payload = json.loads(cache_path.read_text())
    del payload["salt_forms_checked"]
    cache_path.write_text(json.dumps(payload))

    failure_patches, llm_mock, fda_mock = _mapping_patches(
        alias_error=DataSourceError("chembl", "HTTP 500", 500),
    )
    result = await _run_mapping(tmp_path, failure_patches, ["raynaud disease"])

    assert result == {"raynaud disease": "unresolved"}
    fda_mock.assert_not_awaited()
    llm_mock.assert_not_awaited()
    assert json.loads(cache_path.read_text()) == payload


def test_unresolved_is_not_a_judged_label():
    assert _coerce_label("unresolved") is None


# --- extract_approved_from_labels ---


async def test_extract_approved_returns_matched_candidates(tmp_path):
    llm_response = json.dumps(["obesity", "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity and type 2 diabetes."],
            candidate_diseases=["obesity", "type 2 diabetes mellitus", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity", "type 2 diabetes mellitus"}


async def test_extract_approved_returns_empty_on_no_matches(tmp_path):
    llm_response = json.dumps([])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for condition X."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=[],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_empty_candidates_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=["Some label text."],
            candidate_diseases=[],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["obesity"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_rejects_hallucinated_disease(tmp_path):
    llm_response = json.dumps(["obesity", "made up disease"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_invalid_json_returns_empty(tmp_path):
    llm_response = "This is not valid JSON at all"

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_case_insensitive_matching(tmp_path):
    llm_response = json.dumps(["Obesity"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        first = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )
        second = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert first == {"obesity"}
    assert second == {"obesity"}
    mock_llm.assert_awaited_once()


# --- list_approved_indications_from_labels ---


async def test_list_approved_indications_returns_parsed_list(tmp_path):
    llm_response = json.dumps(["type 2 diabetes mellitus", "chronic weight management"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[
                "OZEMPIC is indicated as an adjunct to diet and exercise to "
                "improve glycemic control in adults with type 2 diabetes mellitus."
            ],
            cache_dir=tmp_path,
        )

    assert result == ["type 2 diabetes mellitus", "chronic weight management"]


async def test_list_approved_indications_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[],
            cache_dir=tmp_path,
        )

    assert result == []
    mock_llm.assert_not_awaited()


async def test_list_approved_indications_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["MASH"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[
                "Indicated for MASH with moderate to advanced liver fibrosis."
            ],
            cache_dir=tmp_path,
        )

    assert result == ["MASH"]


async def test_list_approved_indications_invalid_json_returns_empty(tmp_path):
    llm_response = "not valid json"

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label text."],
            cache_dir=tmp_path,
        )

    assert result == []


async def test_list_approved_indications_extracts_array_from_wrapped_object(tmp_path):
    llm_response = json.dumps({"approved": ["obesity"]})

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert result == ["obesity"]


async def test_list_approved_indications_dedupes_case_insensitively(tmp_path):
    llm_response = json.dumps(
        ["Obesity", "obesity", "  type 2 diabetes mellitus  ", ""]
    )

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["Obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_skips_non_string_items(tmp_path):
    llm_response = json.dumps(["obesity", 42, None, "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        first = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )
        second = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert first == ["obesity"]
    assert second == ["obesity"]
    mock_llm.assert_awaited_once()


# --- get_approved_indications (hardcoded table lookup) ---------------------
#
# Reads the real drug_approvals.json file. The table is loaded via
# functools.lru_cache, so each test calls cache_clear() to ensure the file
# is re-read in the current test's process state.


@pytest.fixture(autouse=True)
def _clear_approvals_cache():
    """Reset the lru_cache on the table loader between tests."""
    _load_drug_approvals_table.cache_clear()
    yield
    _load_drug_approvals_table.cache_clear()


@pytest.mark.parametrize(
    "as_of, candidate, expected_in_set",
    [
        # 2025-01-01 cutoff: MASH approval is 2025-08-15, NOT yet approved
        # → MASH stays a candidate (this is the holdout-recoverable case).
        (date(2025, 1, 1), "MASH", False),
        (date(2025, 1, 1), "non-alcoholic steatohepatitis (MASH)", False),
        # 2025-08-15 is the approval date itself; the lookup uses strict
        # less-than, so on the day of approval it's still NOT yet approved.
        (date(2025, 8, 15), "MASH", False),
        # 2025-08-16: one day after approval → MASH is approved → dropped.
        (date(2025, 8, 16), "MASH", True),
        # 2026-01-01: well after the MASH approval → dropped.
        (date(2026, 1, 1), "MASH", True),
        # 2026-01-01: short-form ↔ long-form — table entry "MASH" matches a
        # candidate that contains it. Verifies the LLM matcher equates them.
        (date(2026, 1, 1), "non-alcoholic steatohepatitis (MASH)", True),
    ],
)
async def test_get_approved_indications_semaglutide_mash(
    as_of, candidate, expected_in_set
):
    """Verify the cutoff semantics for semaglutide × MASH against the real table.

    MASH was approved 2025-08-15. The lookup uses strict less-than on the
    cutoff, so a holdout dated on or before that day must NOT see MASH as
    approved (and therefore must NOT drop it from the candidate allowlist).

    The LLM matcher is mocked: when MASH is pre-cutoff approved, it returns the
    candidate as 'removed' (same condition); otherwise nothing reaches it. This
    keeps the test a deterministic unit test (no network).
    """
    expected_remove = [candidate] if expected_in_set else []
    with patch(
        "indication_scout.services.disease_helper.merge_duplicate_diseases",
        new=AsyncMock(return_value={"merge": {}, "remove": expected_remove}),
    ):
        result = await get_approved_indications(
            drug_name="semaglutide",
            candidate_diseases=[candidate],
            as_of=as_of,
        )
    if expected_in_set:
        assert result == {candidate}
    else:
        assert result == set()


async def test_get_approved_indications_semaglutide_pre_2017_returns_empty():
    """Cutoff before semaglutide's first approval (2017-12-05) → empty set.

    No pre-cutoff approvals, so the function short-circuits before the LLM
    matcher — stays a pure unit test (no network).
    """
    result = await get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=[
            "type 2 diabetes mellitus",
            "MASH",
            "chronic weight management",
        ],
        as_of=date(2017, 1, 1),
    )
    assert result == set()


async def test_get_approved_indications_semaglutide_2022_returns_three():
    """Cutoff 2022-01-01 → T2DM (2017), CV risk (2020), and obesity (2021)
    are all approved; CKD (2025) and MASH (2025) are not.

    LLM matcher mocked to remove exactly the three pre-cutoff-approved
    candidates (deterministic unit test, no network).
    """
    candidates = [
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
        "chronic kidney disease",
        "MASH",
    ]
    approved = [
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
    ]
    with patch(
        "indication_scout.services.disease_helper.merge_duplicate_diseases",
        new=AsyncMock(return_value={"merge": {}, "remove": approved}),
    ):
        result = await get_approved_indications(
            drug_name="semaglutide",
            candidate_diseases=candidates,
            as_of=date(2022, 1, 1),
        )
    assert result == {
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
    }


@pytest.mark.parametrize(
    "lookup, table_diseases, matched",
    [
        (
            get_approved_indications,
            ["major depressive disorder", "smoking cessation", "seasonal affective disorder"],
            {"major depressive disorder"},
        ),
        (
            get_combination_only_indications,
            ["obesity", "alzheimer's disease agitation"],
            {"obesity"},
        ),
    ],
)
async def test_bupropion_table_splits_single_drug_and_combination_entries(
    lookup, table_diseases, matched
):
    """Contrave (with naltrexone) and Auvelity (with dextromethorphan) are combination products: the single-drug lookup
    never sees them, and the combination lookup sees only them."""
    candidates = ["major depressive disorder", "obesity", "alzheimer disease"]
    matcher = AsyncMock(return_value={"merge": {}, "remove": sorted(matched)})
    with patch(
        "indication_scout.services.disease_helper.merge_duplicate_diseases", new=matcher
    ):
        result = await lookup(
            drug_name="bupropion",
            candidate_diseases=candidates,
            as_of=date(2026, 11, 25),
        )
    matcher.assert_awaited_once_with(candidates, table_diseases)
    assert result == matched


def test_list_approved_indications_at_bupropion_leaves_out_combination_products():
    assert list_approved_indications_at("bupropion", date(2026, 11, 25)) == [
        "major depressive disorder",
        "smoking cessation",
        "seasonal affective disorder",
    ]


async def test_get_approved_indications_returns_empty_when_as_of_is_none():
    """as_of=None → callers should use the live FDA path; lookup returns empty."""
    result = await get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=["MASH"],
        as_of=None,
    )
    assert result == set()


async def test_get_approved_indications_uncurated_drug_returns_empty(caplog):
    """Drug not in the table → empty set + warning logged."""
    with caplog.at_level("WARNING"):
        result = await get_approved_indications(
            drug_name="not-a-real-drug",
            candidate_diseases=["obesity"],
            as_of=date(2025, 1, 1),
        )
    assert result == set()
    assert any(
        "not in hardcoded approvals table" in rec.message for rec in caplog.records
    )


def test_load_drug_approvals_table_rejects_duplicate_key(tmp_path):
    """A repeated top-level drug key must raise, not silently keep the last value."""
    bad_table = tmp_path / "drug_approvals.json"
    bad_table.write_text("""
        {
          "methotrexate": [{"disease": "psoriasis", "approved": "1972-12-31"}],
          "methotrexate": [{"disease": "rheumatoid arthritis", "approved": "1988-12-31"}]
        }
        """)
    with patch(
        "indication_scout.services.approval_check.DRUG_APPROVALS_PATH", bad_table
    ):
        with pytest.raises(ValueError, match="methotrexate"):
            _load_drug_approvals_table()


def test_load_drug_approvals_table_has_verified_finasteride_records():
    table = _load_drug_approvals_table()

    assert table["finasteride"] == [
        {"disease": "benign prostatic hyperplasia", "approved": "1992-06-19"},
        {"disease": "male pattern hair loss", "approved": "1997-12-19"},
    ]


# --- list_approved_indications_at ------------------------------------------


def test_list_approved_indications_at_semaglutide_2022():
    """Cutoff 2022-01-01 → only the three pre-2022 approvals returned, in
    table order (the loader preserves JSON file order).
    """
    result = list_approved_indications_at("semaglutide", date(2022, 1, 1))
    assert result == [
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
    ]


def test_list_approved_indications_at_semaglutide_pre_2017_returns_empty():
    """Cutoff before any approval → empty list."""
    result = list_approved_indications_at("semaglutide", date(2017, 1, 1))
    assert result == []


def test_list_approved_indications_at_returns_empty_when_as_of_is_none():
    """as_of=None → empty list (callers should use the live FDA path)."""
    result = list_approved_indications_at("semaglutide", None)
    assert result == []
