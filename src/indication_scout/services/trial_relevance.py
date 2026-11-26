"""Per-trial relevance: combine four isolated, cached checks into relevant / excluded / combination-only.

Replaces the clinical-trials agent's single batch classification, whose per-trial verdicts depended on the rest of the
batch and moved between runs (PLAN_per_trial_relevance). A trial is relevant when, in this order:

1. drug role is "studied" (services.trial_drug_role);
2. the trial set out to treat the candidate disease (services.trial_target);
3. it does not study an approved indication of the drug or a narrower form (services.trial_approved_subtype);
4. the drug is not given only inside a combination with no arm isolating it (services.trial_drug_isolation).

Each check runs only on trials that passed the previous ones. A trial failing 1-3 is excluded; an unusable answer from
1-3 also excludes it for this run. A trial failing 4 goes to the combination-only list; 4 keeps the trial when its two
models disagree or a call fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.trial_approved_subtype import (
    judge_trials_approved_subtype,
)
from indication_scout.services.trial_drug_isolation import find_combination_only_trials
from indication_scout.services.trial_drug_role import judge_trial_drug_roles
from indication_scout.services.trial_target import judge_trials_treat_disease

logger = logging.getLogger(__name__)

TrialRelevanceVerdict = Literal["relevant", "excluded", "combination"]


class TrialRelevance(BaseModel):
    """Per-trial verdicts and, for each excluded trial, the check that excluded it."""

    verdicts: dict[str, TrialRelevanceVerdict] = {}
    excluded_by: dict[str, str] = {}

    def ncts(self, verdict: TrialRelevanceVerdict) -> list[str]:
        return sorted(n for n, v in self.verdicts.items() if v == verdict)


async def classify_trials(
    drug: str,
    disease: str,
    trials: list[Trial],
    approved_indications: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> TrialRelevance:
    """Classify each trial for this drug x disease pair. Every input trial gets exactly one verdict."""
    unique = list({t.nct_id: t for t in trials if t.nct_id}.values())
    result = TrialRelevance()

    def exclude(ncts: list[str], reason: str) -> None:
        for n in ncts:
            result.verdicts[n] = "excluded"
            result.excluded_by[n] = reason

    roles = await judge_trial_drug_roles(drug, unique, cache_dir)
    exclude(
        [t.nct_id for t in unique if roles.get(t.nct_id) != "studied"],
        "drug role",
    )
    for t in unique:
        if t.nct_id in result.excluded_by and roles.get(t.nct_id) is None:
            result.excluded_by[t.nct_id] = "drug role (no usable answer)"
    remaining = [t for t in unique if t.nct_id not in result.verdicts]

    treats = await judge_trials_treat_disease(drug, disease, remaining, cache_dir)
    exclude([t.nct_id for t in remaining if not treats.get(t.nct_id, False)], "treats the disease")
    remaining = [t for t in remaining if t.nct_id not in result.verdicts]

    subtype = await judge_trials_approved_subtype(drug, remaining, approved_indications, cache_dir)
    exclude(
        [t.nct_id for t in remaining if subtype.get(t.nct_id) is not False],
        "approved subtype",
    )
    remaining = [t for t in remaining if t.nct_id not in result.verdicts]

    combination = await find_combination_only_trials(drug, remaining, cache_dir)
    for t in remaining:
        result.verdicts[t.nct_id] = "combination" if t.nct_id in combination else "relevant"

    logger.info(
        "trial_relevance: %s x %s — %d relevant, %d excluded, %d combination-only",
        drug,
        disease,
        len(result.ncts("relevant")),
        len(result.ncts("excluded")),
        len(result.ncts("combination")),
    )
    return result
