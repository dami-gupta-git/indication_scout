"""Unit tests for the per-trial relevance checks (PLAN_per_trial_relevance): the drug-role and approved-subtype checks'
caching and failure handling, and how services/trial_relevance combines the four checks."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.trial_approved_subtype import (
    judge_trials_approved_subtype,
)
from indication_scout.services.trial_drug_role import judge_trial_drug_roles
from indication_scout.services.trial_relevance import TrialRelevance, classify_trials

TRIAL = Trial(nct_id="NCT00000001", title="Metformin in prediabetes")


# --- drug role ---------------------------------------------------------------------------------------------------


async def test_drug_role_is_cached_and_served_from_cache(tmp_path):
    first = AsyncMock(return_value=json.dumps({"reason": "r", "drug_role": "studied"}))
    with patch("indication_scout.services.trial_drug_role.query_small_llm", new=first):
        assert await judge_trial_drug_roles("metformin", [TRIAL], tmp_path) == {"NCT00000001": "studied"}
    second = AsyncMock(return_value=json.dumps({"reason": "r", "drug_role": "comparator"}))
    with patch("indication_scout.services.trial_drug_role.query_small_llm", new=second):
        assert await judge_trial_drug_roles("metformin", [TRIAL], tmp_path) == {"NCT00000001": "studied"}
    second.assert_not_awaited()


async def test_drug_role_unreadable_answer_is_none_and_not_cached(tmp_path):
    bad = AsyncMock(return_value="no json here")
    with patch("indication_scout.services.trial_drug_role.query_small_llm", new=bad):
        assert await judge_trial_drug_roles("metformin", [TRIAL], tmp_path) == {"NCT00000001": None}
    good = AsyncMock(return_value=json.dumps({"reason": "r", "drug_role": "absent"}))
    with patch("indication_scout.services.trial_drug_role.query_small_llm", new=good):
        assert await judge_trial_drug_roles("metformin", [TRIAL], tmp_path) == {"NCT00000001": "absent"}
    good.assert_awaited_once()


# --- approved subtype -----------------------------------------------------------------------------------------------


async def test_approved_subtype_cache_is_keyed_on_the_approved_list(tmp_path):
    yes = AsyncMock(return_value=json.dumps({"reason": "r", "verdict": "approved_subtype"}))
    with patch("indication_scout.services.trial_approved_subtype.query_small_llm", new=yes):
        assert await judge_trials_approved_subtype("semaglutide", [TRIAL], ["MASH", "obesity"], tmp_path) == {
            "NCT00000001": True
        }
    no = AsyncMock(return_value=json.dumps({"reason": "r", "verdict": "not_approved_subtype"}))
    with patch("indication_scout.services.trial_approved_subtype.query_small_llm", new=no):
        # Same list in another order: served from the cache.
        assert await judge_trials_approved_subtype("semaglutide", [TRIAL], ["obesity", "MASH"], tmp_path) == {
            "NCT00000001": True
        }
        no.assert_not_awaited()
        # A changed list asks again.
        assert await judge_trials_approved_subtype("semaglutide", [TRIAL], ["obesity"], tmp_path) == {
            "NCT00000001": False
        }
        no.assert_awaited_once()


async def test_approved_subtype_with_no_approved_list_makes_no_call(tmp_path):
    llm = AsyncMock()
    with patch("indication_scout.services.trial_approved_subtype.query_small_llm", new=llm):
        assert await judge_trials_approved_subtype("metformin", [TRIAL], [], tmp_path) == {"NCT00000001": False}
    llm.assert_not_awaited()


async def test_approved_subtype_unreadable_answer_is_none(tmp_path):
    with patch(
        "indication_scout.services.trial_approved_subtype.query_small_llm",
        new=AsyncMock(return_value="not json"),
    ):
        assert await judge_trials_approved_subtype("metformin", [TRIAL], ["type 2 diabetes"], tmp_path) == {
            "NCT00000001": None
        }


# --- combining the four checks ---------------------------------------------------------------------------------


def _patch_checks(role, treats, subtype, combination):
    """Stub the four checks with fixed answers for TRIAL; returns the mocks to inspect calls."""
    mocks = {
        "role": AsyncMock(return_value={"NCT00000001": role}),
        "treats": AsyncMock(return_value={"NCT00000001": treats}),
        "subtype": AsyncMock(return_value={"NCT00000001": subtype}),
        "isolation": AsyncMock(return_value={"NCT00000001"} if combination else set()),
    }
    base = "indication_scout.services.trial_relevance."
    patches = [
        patch(base + "judge_trial_drug_roles", new=mocks["role"]),
        patch(base + "judge_trials_treat_disease", new=mocks["treats"]),
        patch(base + "judge_trials_approved_subtype", new=mocks["subtype"]),
        patch(base + "find_combination_only_trials", new=mocks["isolation"]),
    ]
    return patches, mocks


@pytest.mark.parametrize(
    "role, treats, subtype, combination, verdict, excluded_by, checks_called",
    [
        ("studied", True, False, False, "relevant", None, ["role", "treats", "subtype", "isolation"]),
        ("comparator", True, False, False, "excluded", "drug role", ["role"]),
        ("studied", False, False, False, "excluded", "treats the disease", ["role", "treats"]),
        ("studied", True, True, False, "excluded", "approved subtype", ["role", "treats", "subtype"]),
        ("studied", True, False, True, "combination", None, ["role", "treats", "subtype", "isolation"]),
    ],
)
async def test_classify_trials_combines_the_checks_in_order(
    role, treats, subtype, combination, verdict, excluded_by, checks_called, tmp_path
):
    patches, mocks = _patch_checks(role, treats, subtype, combination)
    for p in patches:
        p.start()
    try:
        result = await classify_trials("metformin", "prediabetes", [TRIAL], ["type 2 diabetes"], tmp_path)
    finally:
        for p in patches:
            p.stop()

    assert result == TrialRelevance(
        verdicts={"NCT00000001": verdict},
        excluded_by={"NCT00000001": excluded_by} if excluded_by else {},
    )
    # Each check's trial list is its second positional argument, except treats-the-disease (third).
    trial_arg = {"role": 1, "treats": 2, "subtype": 1, "isolation": 1}
    reached = [
        name
        for name, m in mocks.items()
        if m.await_count and TRIAL in m.await_args.args[trial_arg[name]]
    ]
    assert reached == checks_called


async def test_classify_trials_excludes_on_an_unreadable_role(tmp_path):
    patches, _ = _patch_checks(None, True, False, False)
    for p in patches:
        p.start()
    try:
        result = await classify_trials("metformin", "prediabetes", [TRIAL], [], tmp_path)
    finally:
        for p in patches:
            p.stop()
    assert result == TrialRelevance(
        verdicts={"NCT00000001": "excluded"},
        excluded_by={"NCT00000001": "drug role (no usable answer)"},
    )
