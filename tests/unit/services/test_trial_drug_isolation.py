"""Unit tests for services/trial_drug_isolation: the arm comparison and the two-check agreement rule."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.models.model_clinical_trials import ArmGroup, Intervention, Trial
from indication_scout.services.trial_drug_isolation import (
    classify_arms,
    find_combination_only_trials,
)


def _trial(nct: str, arms: dict[str, list[str]], interventions: list[str]) -> Trial:
    return Trial(
        nct_id=nct,
        title=f"Trial {nct}",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name=name)
            for name in interventions
        ],
        arm_groups=[
            ArmGroup(label=label, arm_type="", intervention_names=[f"Drug: {n}" for n in names])
            for label, names in arms.items()
        ],
    )


# Naltrexone/bupropion fixed-dose product against placebo (NCT03539900 shape).
NB_VS_PLACEBO = _trial(
    "NCT00000001",
    {"NB": ["NB combination"], "Placebo": ["Placebo"]},
    ["NB combination", "Placebo"],
)
NB_AGENTS = {"nb combination": ["naltrexone", "bupropion"], "placebo": []}

# Metformin + letrozole vs myo-inositol + letrozole (NCT07585396 shape): head-to-head on a shared backbone.
HEAD_TO_HEAD = _trial(
    "NCT00000002",
    {"Met": ["Metformin", "Letrozole"], "MI": ["Myoinositol", "Letrozole"]},
    ["Metformin", "Letrozole", "Myoinositol"],
)
HEAD_TO_HEAD_AGENTS = {
    "metformin": ["metformin"],
    "letrozole": ["letrozole"],
    "myoinositol": ["myo-inositol"],
}

# Metformin + clomiphene vs placebo + clomiphene: add-on design.
ADD_ON = _trial(
    "NCT00000003",
    {"Met": ["Metformin", "Clomiphene"], "Pbo": ["Placebo", "Clomiphene"]},
    ["Metformin", "Clomiphene", "Placebo"],
)
ADD_ON_AGENTS = {"metformin": ["metformin"], "clomiphene": ["clomiphene"], "placebo": []}

# No arms reported; the drug is the only drug listed.
NO_ARMS_ONLY_DRUG = _trial("NCT00000004", {}, ["Sildenafil", "Placebo"])
NO_ARMS_ONLY_DRUG_AGENTS = {"sildenafil": ["sildenafil"], "placebo": []}

# No arms reported; another drug is listed too, so the design cannot be read.
NO_ARMS_TWO_DRUGS = _trial("NCT00000005", {}, ["Metformin", "Clomiphene"])
NO_ARMS_TWO_DRUGS_AGENTS = {"metformin": ["metformin"], "clomiphene": ["clomiphene"]}


@pytest.mark.parametrize(
    "drug, trial, agents, expected",
    [
        ("bupropion", NB_VS_PLACEBO, NB_AGENTS, "combination_only"),
        ("metformin", HEAD_TO_HEAD, HEAD_TO_HEAD_AGENTS, "isolated"),
        ("metformin hydrochloride", ADD_ON, ADD_ON_AGENTS, "isolated"),
        ("sildenafil", NO_ARMS_ONLY_DRUG, NO_ARMS_ONLY_DRUG_AGENTS, "isolated"),
        ("metformin", NO_ARMS_TWO_DRUGS, NO_ARMS_TWO_DRUGS_AGENTS, "unclear"),
    ],
)
def test_classify_arms(drug, trial, agents, expected):
    verdict, _ = classify_arms(drug, trial, agents)
    assert verdict == expected


def test_classify_arms_reports_the_arms_of_a_combination_trial():
    assert classify_arms("bupropion", NB_VS_PLACEBO, NB_AGENTS) == (
        "combination_only",
        "arms with the drug: [['naltrexone']]; arms without it: [[]]",
    )


def _llm(isolation_verdict: str, agents: dict[str, list[str]] | str):
    """query_small_llm stand-in answering the isolation prompt and the ingredient prompt."""
    listing = (
        agents
        if isinstance(agents, str)
        else json.dumps({str(i + 1): v for i, v in enumerate(agents.values())})
    )

    async def respond(prompt: str) -> str:
        if "List the active drug ingredients" in prompt:
            return listing
        return json.dumps({"reason": "r", "verdict": isolation_verdict})

    return AsyncMock(side_effect=respond)


@pytest.mark.parametrize(
    "isolation_verdict, agents, flagged",
    [
        ("combination_only", NB_AGENTS, {"NCT00000001"}),  # both checks agree
        ("isolated", NB_AGENTS, set()),  # the model check disagrees
        ("combination_only", {"nb combination": ["bupropion"], "placebo": []}, set()),  # arm comparison disagrees
        ("combination_only", "not json", set()),  # ingredient listing unusable
        ("not a verdict", NB_AGENTS, set()),  # isolation answer unusable
    ],
)
async def test_find_combination_only_trials_requires_both_checks(
    isolation_verdict, agents, flagged, tmp_path
):
    with patch(
        "indication_scout.services.trial_drug_isolation.query_small_llm",
        new=_llm(isolation_verdict, agents),
    ):
        result = await find_combination_only_trials("bupropion", [NB_VS_PLACEBO], tmp_path)
    assert result == flagged


async def test_find_combination_only_trials_caches_both_checks(tmp_path):
    first = _llm("combination_only", NB_AGENTS)
    with patch("indication_scout.services.trial_drug_isolation.query_small_llm", new=first):
        assert await find_combination_only_trials("bupropion", [NB_VS_PLACEBO], tmp_path) == {
            "NCT00000001"
        }
    assert first.await_count == 2

    second = _llm("isolated", {"nb combination": [], "placebo": []})
    with patch("indication_scout.services.trial_drug_isolation.query_small_llm", new=second):
        assert await find_combination_only_trials("bupropion", [NB_VS_PLACEBO], tmp_path) == {
            "NCT00000001"
        }
    second.assert_not_awaited()
