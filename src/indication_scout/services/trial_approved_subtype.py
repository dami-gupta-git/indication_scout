"""Per-trial approved-subtype check — does the trial study an approved indication of the drug, or a narrower form?

A trial of an already-approved use is not repurposing evidence for the candidate. Asked once per trial in isolation and
cached on the trial and the sorted approved-indication list, so a changed list re-asks only this check
(PLAN_per_trial_relevance). The list is label-derived in a live run and the hardcoded table as of the cutoff in a
holdout run; an empty list means no trial is an approved subtype and no call is made.
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
from indication_scout.services.trial_target import _FIELD_CAP, _format_primary_outcomes
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_APPROVED_SUBTYPE_PROMPT = (_PROMPTS_DIR / "trial_approved_subtype.txt").read_text()
APPROVED_SUBTYPE_NS = "trial_approved_subtype"
_VERDICTS = {"approved_subtype", "not_approved_subtype"}


async def judge_trials_approved_subtype(
    drug: str,
    trials: list[Trial],
    approved_indications: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, bool | None]:
    """Return {nct_id: whether the trial studies an approved indication or a narrower form}, None where no usable
    answer was obtained. An unusable answer is not cached."""
    if not trials:
        return {}
    approved = sorted({a.strip() for a in approved_indications if a.strip()}, key=str.lower)
    if not approved:
        return {t.nct_id: False for t in trials}
    semaphore = asyncio.Semaphore(_settings.rag_llm_concurrency)

    async def judge_one(trial: Trial) -> tuple[str, bool | None]:
        cache_params = {
            "nct": trial.nct_id,
            "drug": drug.strip().lower(),
            "approved": approved,
            "small_llm_model": _settings.small_llm_model,
            "logic_version": "trial_approved_subtype_v3",
        }
        cached = cache_get(APPROVED_SUBTYPE_NS, cache_params, cache_dir)
        if isinstance(cached, str) and cached in _VERDICTS:
            return trial.nct_id, cached == "approved_subtype"

        prompt = _APPROVED_SUBTYPE_PROMPT.format(
            drug=drug,
            approved="; ".join(approved),
            nct=trial.nct_id,
            title=trial.title,
            summary=(trial.brief_summary or "")[:_FIELD_CAP],
            conditions="; ".join(trial.indications) or "(none listed)",
            outcomes=_format_primary_outcomes(trial),
        )
        async with semaphore:
            try:
                response = await query_small_llm(prompt)
            except DataSourceError as exc:
                logger.warning(
                    "trial_approved_subtype: LLM call failed for %s / %s: %s",
                    drug,
                    trial.nct_id,
                    exc,
                )
                return trial.nct_id, None
        data = parse_last_json_object(response)
        verdict = (
            str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
        )
        if verdict not in _VERDICTS:
            logger.warning(
                "trial_approved_subtype: unusable verdict for %s / %s: %s",
                drug,
                trial.nct_id,
                response,
            )
            return trial.nct_id, None
        cache_set(APPROVED_SUBTYPE_NS, cache_params, verdict, cache_dir, ttl=CACHE_TTL)
        return trial.nct_id, verdict == "approved_subtype"

    return dict(await asyncio.gather(*(judge_one(t) for t in trials)))
