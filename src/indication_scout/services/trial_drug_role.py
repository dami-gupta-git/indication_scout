"""Per-trial drug-role check — what part does the drug play in this trial?

Studied, comparator, background or absent. Asked once per trial in isolation and cached on the trial and the drug, so a
trial's role does not depend on which other trials were reviewed with it (PLAN_per_trial_relevance). Whether a trial
that gives the drug together with another agent counts as evidence for the drug alone is decided by
services.trial_drug_isolation, not here.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from indication_scout.config import get_settings
from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import parse_last_json_object, query_small_llm
from indication_scout.services.trial_target import _FIELD_CAP, _format_arms
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DRUG_ROLE_PROMPT = (_PROMPTS_DIR / "trial_drug_role.txt").read_text()
DRUG_ROLE_NS = "trial_drug_role"
DrugRole = Literal["studied", "comparator", "background", "absent"]
_ROLES = {"studied", "comparator", "background", "absent"}


def _format_interventions(trial: Trial) -> str:
    return (
        "\n".join(
            f"- {iv.intervention_name} ({iv.intervention_type or 'type not reported'}): "
            f"{(iv.description or '')[:300]}"
            for iv in trial.interventions
        )
        or "(none reported)"
    )


async def judge_trial_drug_roles(
    drug: str,
    trials: list[Trial],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, DrugRole | None]:
    """Return {nct_id: the drug's role in the trial}, None where no usable answer was obtained.

    An unusable answer is not cached, so a later run asks again.
    """
    if not trials:
        return {}
    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(trial: Trial) -> tuple[str, DrugRole | None]:
        cache_params = {
            "nct": trial.nct_id,
            "drug": drug.strip().lower(),
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "trial_drug_role_v1",
        }
        cached = cache_get(DRUG_ROLE_NS, cache_params, cache_dir)
        if isinstance(cached, str) and cached in _ROLES:
            return trial.nct_id, cached  # type: ignore[return-value]

        prompt = _DRUG_ROLE_PROMPT.format(
            drug=drug,
            nct=trial.nct_id,
            title=trial.title,
            summary=(trial.brief_summary or "")[:_FIELD_CAP],
            interventions=_format_interventions(trial),
            arms=_format_arms(trial),
        )
        async with semaphore:
            try:
                response = await query_small_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "trial_drug_role: LLM call failed for %s / %s: %s",
                    drug,
                    trial.nct_id,
                    exc,
                )
                return trial.nct_id, None
        data = parse_last_json_object(response)
        role = (
            str(data.get("drug_role", "")).strip().lower()
            if isinstance(data, dict)
            else ""
        )
        if role not in _ROLES:
            logger.warning(
                "trial_drug_role: unusable role for %s / %s: %s",
                drug,
                trial.nct_id,
                response,
            )
            return trial.nct_id, None
        cache_set(DRUG_ROLE_NS, cache_params, role, cache_dir, ttl=CACHE_TTL)
        return trial.nct_id, role  # type: ignore[return-value]

    return dict(await asyncio.gather(*(judge_one(t) for t in trials)))
