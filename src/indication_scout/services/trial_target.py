"""Per-trial therapeutic-target gate — did this trial set out to TREAT the candidate disease?

The clinical-trials agent already checks the drug's ROLE per trial (studied vs comparator vs
background) and enforces it deterministically at finalize. Nothing checked the trial's TARGET: a
trial can study the right drug in patients who have the right disease and still be aimed at a
complication (cardiomyopathy in muscular dystrophy), an already-approved comorbidity (pulmonary
hypertension in lung disease), or at pharmacokinetics rather than treatment. Left to the agent's
one whole-batch relevance call, those graded as development for the candidate indication and the
verdict moved between runs.

One isolated call per trial, cached on the trial and the disease, so a trial's verdict does not
depend on which other trials shared its batch. Mirrors the per-paper gate the literature path uses.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from indication_scout.config import get_settings
from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import parse_last_json_object, query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_TRIAL_TREATS_PROMPT = (_PROMPTS_DIR / "trial_treats_disease.txt").read_text()
_TRIAL_TREATS_VERDICTS = {"treats", "not_treats"}
TRIAL_TARGET_NS = "trial_treats_disease"
# Longest slice of a field given to the gate. Registry summaries run to several thousand
# characters; the objective is stated at the top, and the tail is eligibility boilerplate.
_FIELD_CAP = 1500


async def judge_trials_treat_disease(
    drug: str,
    disease: str,
    trials: list[Trial],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, bool]:
    """Return {nct_id: whether the trial set out to treat this disease with this drug}.

    Missing or invalid responses fail closed as ``False`` and are not cached, so a trial is dropped
    rather than counted on an answer that could not be read.
    """
    if not trials:
        return {}

    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(trial: Trial) -> tuple[str, bool]:
        cache_params = {
            "nct": trial.nct_id,
            "disease": disease,
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "trial_treats_disease_v1",
        }
        cached = cache_get(TRIAL_TARGET_NS, cache_params, cache_dir)
        if isinstance(cached, str) and cached in _TRIAL_TREATS_VERDICTS:
            return trial.nct_id, cached == "treats"

        prompt = _TRIAL_TREATS_PROMPT.format(
            drug=drug,
            disease=disease,
            nct=trial.nct_id,
            title=trial.title,
            summary=(trial.brief_summary or "")[:_FIELD_CAP],
            conditions="; ".join(trial.indications) or "(none listed)",
            outcomes="; ".join(o.measure for o in trial.primary_outcomes if o.measure)
            or "(none listed)",
        )
        async with semaphore:
            try:
                response = await query_small_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "trial_target: LLM call failed for %s x %s / %s; excluding trial: %s",
                    drug,
                    disease,
                    trial.nct_id,
                    exc,
                )
                return trial.nct_id, False
        data = parse_last_json_object(response)
        verdict = (
            str(data.get("verdict", "")).strip().lower()
            if isinstance(data, dict)
            else ""
        )
        if verdict not in _TRIAL_TREATS_VERDICTS:
            logger.warning(
                "trial_target: unusable verdict for %s x %s / %s; excluding trial. "
                "Response was: %s",
                drug,
                disease,
                trial.nct_id,
                response,
            )
            return trial.nct_id, False

        cache_set(TRIAL_TARGET_NS, cache_params, verdict, cache_dir, ttl=CACHE_TTL)
        return trial.nct_id, verdict == "treats"

    decisions = await asyncio.gather(*(judge_one(trial) for trial in trials))
    return dict(decisions)
