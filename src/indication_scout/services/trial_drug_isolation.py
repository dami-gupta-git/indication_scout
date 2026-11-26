"""Per-trial check: can this trial show the effect of the drug on its own, or only inside a combination?

A trial that gives the drug only together with another active agent and has no comparison isolating the drug (a
naltrexone/bupropion fixed-dose product against placebo, metformin + esomeprazole against placebo) is not evidence
for the drug alone, yet the clinical-trials agent counted such trials as the drug's development.

Two independent checks run per trial and a trial is flagged combination-only only when both agree:

1. A model reads the arms and returns isolated / combination_only / absent / unclear.
2. A model lists the active drug ingredients of each intervention; code then compares the arms' ingredient sets.

Each check alone was wrong on about 1 in 6 of its combination-only calls, in different trials: the first misreads
multi-arm designs, the second is misled by registry records that list every intervention under every arm. On 228
saved relevant trials across bupropion, metformin, semaglutide and sildenafil, the 26 trials both flagged were all
correct (2026-11-26). A disagreement, an unreadable answer or a failed call leaves the trial counted.

Both results are cached per trial and drug, so a verdict does not depend on which other trials shared the run.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Literal

from indication_scout.config import get_settings
from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import parse_last_json_object, query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_ISOLATION_PROMPT = (_PROMPTS_DIR / "trial_drug_isolation.txt").read_text()
_AGENTS_PROMPT = (_PROMPTS_DIR / "trial_intervention_agents.txt").read_text()
ISOLATION_NS = "trial_drug_isolation"
AGENTS_NS = "trial_intervention_agents"
# Longest slice of an intervention description given to either check.
_DESCRIPTION_CAP = 300

IsolationVerdict = Literal["isolated", "combination_only", "absent", "unclear"]
_VERDICTS = {"isolated", "combination_only", "absent", "unclear"}
_TARGET = "__target__"
# The registry prefixes each arm's intervention with its type ("Drug: Metformin").
_ARM_INTERVENTION_PREFIX = re.compile(r"^[A-Za-z /]+:\s*")


def _format_arms(trial: Trial) -> str:
    return (
        "\n".join(
            f"- {arm.label} [{arm.arm_type or 'type not reported'}]: "
            f"{', '.join(arm.intervention_names) or 'no intervention'}"
            for arm in trial.arm_groups
        )
        or "(none reported)"
    )


def _format_interventions(trial: Trial, numbered: bool) -> str:
    return "\n".join(
        f"{f'{i + 1}. ' if numbered else '- '}{iv.intervention_name} ({iv.intervention_type or 'type not reported'}): "
        f"{(iv.description or '')[:_DESCRIPTION_CAP]}"
        for i, iv in enumerate(trial.interventions)
    )


def classify_arms(
    drug: str, trial: Trial, agents_by_intervention: dict[str, list[str]]
) -> tuple[IsolationVerdict, str]:
    """Decide isolation from each intervention's active ingredients by comparing the arms' ingredient sets.

    `agents_by_intervention` maps a lowercased intervention name to its active ingredients. Isolated when an arm gives
    the drug as its only ingredient, when two arms differ only by the drug, or when two arms differ by swapping the
    drug for another ingredient on a shared remainder (head-to-head). Arms that cannot be matched to the intervention
    list fall back to the trial's full ingredient set.
    """
    target = drug.strip().lower()
    # The listing names generic ingredients without salts; the caller's drug name may carry one
    # ("metformin hydrochloride"), so an ingredient that is one of its words also matches.
    target_words = set(target.split())

    def mark(agent: str) -> str:
        agent = agent.strip().lower()
        return _TARGET if target in agent or agent in target_words else agent

    by_name = {
        name: frozenset(mark(a) for a in agents)
        for name, agents in agents_by_intervention.items()
    }

    arm_sets: list[frozenset[str]] = []
    for arm in trial.arm_groups:
        names = [
            _ARM_INTERVENTION_PREFIX.sub("", n).strip().lower()
            for n in arm.intervention_names
        ]
        if not all(n in by_name for n in names):
            arm_sets = []
            break
        arm_sets.append(frozenset().union(*(by_name[n] for n in names)))

    if not arm_sets:
        everything = frozenset().union(*by_name.values())
        if _TARGET not in everything:
            return "absent", "no intervention delivers the drug"
        if everything == {_TARGET}:
            return "isolated", "arms unusable; the drug is the only drug in the trial"
        return "unclear", f"arms unusable; drugs in the trial: {sorted(everything)}"

    with_drug = [s for s in arm_sets if _TARGET in s]
    without_drug = [s for s in arm_sets if _TARGET not in s]
    if not with_drug:
        return "absent", "no arm delivers the drug"
    for arm in with_drug:
        rest = arm - {_TARGET}
        if not rest:
            return "isolated", "an arm gives the drug as its only drug"
        for other in without_drug:
            if other == rest:
                return "isolated", f"two arms differ only by the drug (shared: {sorted(rest)})"
            if rest <= other and other - arm:
                return "isolated", f"head-to-head: the drug vs {sorted(other - arm)} on {sorted(rest)}"
    return (
        "combination_only",
        f"arms with the drug: {[sorted(s - {_TARGET}) for s in with_drug]}; "
        f"arms without it: {[sorted(s) for s in without_drug]}",
    )


async def _judge_isolation(
    drug: str, trial: Trial, cache_dir: Path
) -> IsolationVerdict | None:
    """Check 1: the model reads the arms and returns a verdict. None when no usable verdict was obtained."""
    cache_params = {
        "nct": trial.nct_id,
        "drug": drug.strip().lower(),
        "small_llm_model": _settings.small_llm_model,
        "logic_version": "trial_drug_isolation_v1",
    }
    cached = cache_get(ISOLATION_NS, cache_params, cache_dir)
    if isinstance(cached, str) and cached in _VERDICTS:
        return cached  # type: ignore[return-value]

    prompt = _ISOLATION_PROMPT.format(
        drug=drug,
        nct=trial.nct_id,
        title=trial.title,
        arms=_format_arms(trial),
        interventions=_format_interventions(trial, numbered=False),
    )
    try:
        response = await query_small_llm(prompt)
    except DataSourceError as exc:
        logger.warning(
            "trial_drug_isolation: isolation call failed for %s / %s: %s",
            drug,
            trial.nct_id,
            exc,
        )
        return None
    data = parse_last_json_object(response)
    verdict = (
        str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    )
    if verdict not in _VERDICTS:
        logger.warning(
            "trial_drug_isolation: unusable isolation verdict for %s / %s: %s",
            drug,
            trial.nct_id,
            response,
        )
        return None
    cache_set(ISOLATION_NS, cache_params, verdict, cache_dir, ttl=CACHE_TTL)
    return verdict  # type: ignore[return-value]


async def _list_intervention_agents(
    trial: Trial, cache_dir: Path
) -> dict[str, list[str]] | None:
    """Check 2, model step: each intervention's active ingredients, keyed by lowercased intervention name.

    Drug-independent, so cached per trial. None when the listing is unusable or incomplete.
    """
    cache_params = {
        "nct": trial.nct_id,
        "small_llm_model": _settings.small_llm_model,
        "logic_version": "trial_intervention_agents_v1",
    }
    cached = cache_get(AGENTS_NS, cache_params, cache_dir)
    if isinstance(cached, dict):
        return cached

    prompt = _AGENTS_PROMPT.format(
        nct=trial.nct_id,
        title=trial.title,
        interventions=_format_interventions(trial, numbered=True),
    )
    try:
        response = await query_small_llm(prompt)
    except DataSourceError as exc:
        logger.warning(
            "trial_drug_isolation: ingredient listing failed for %s: %s",
            trial.nct_id,
            exc,
        )
        return None
    data = parse_last_json_object(response)
    if not isinstance(data, dict):
        logger.warning(
            "trial_drug_isolation: unusable ingredient listing for %s: %s",
            trial.nct_id,
            response,
        )
        return None
    agents: dict[str, list[str]] = {}
    for i, iv in enumerate(trial.interventions):
        listed = data.get(str(i + 1))
        if not isinstance(listed, list) or not all(isinstance(a, str) for a in listed):
            logger.warning(
                "trial_drug_isolation: ingredient listing for %s omits intervention %d: %s",
                trial.nct_id,
                i + 1,
                response,
            )
            return None
        agents[iv.intervention_name.strip().lower()] = listed
    cache_set(AGENTS_NS, cache_params, agents, cache_dir, ttl=CACHE_TTL)
    return agents


async def find_combination_only_trials(
    drug: str,
    trials: list[Trial],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> set[str]:
    """Return the NCT ids of trials that give the drug only inside a combination, per both checks."""
    if not trials:
        return set()
    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def check(trial: Trial) -> tuple[str, bool]:
        async with semaphore:
            first, listing = await asyncio.gather(
                _judge_isolation(drug, trial, cache_dir),
                _list_intervention_agents(trial, cache_dir),
            )
        if first != "combination_only" or listing is None:
            return trial.nct_id, False
        second, why = classify_arms(drug, trial, listing)
        if second != "combination_only":
            logger.info(
                "trial_drug_isolation: %s / %s kept — checks disagree (model: %s; arm comparison: %s, %s)",
                drug,
                trial.nct_id,
                first,
                second,
                why,
            )
            return trial.nct_id, False
        logger.info(
            "trial_drug_isolation: %s / %s is combination-only (%s)",
            drug,
            trial.nct_id,
            why,
        )
        return trial.nct_id, True

    results = await asyncio.gather(*(check(t) for t in trials))
    return {nct for nct, flagged in results if flagged}
