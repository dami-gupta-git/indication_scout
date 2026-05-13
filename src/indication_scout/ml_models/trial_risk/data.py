"""Load trial records from `_cache/ct_completed/` and `_cache/ct_terminated/`.

Each cache file groups trials by (drug, mesh_term). We flatten them into one
record per NCT ID, deduplicating across files (a trial can appear under multiple
mesh terms for the same drug). The drug name is preserved for grouped CV.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.models.model_clinical_trials import Trial

logger = logging.getLogger(__name__)


@dataclass
class LabeledTrial:
    """A single trial with its label and the drug it was cached under."""

    trial: Trial
    label: int  # 1 = terminated, 0 = completed
    drug: str  # drug name from the cache params (used for grouped CV)


def _load_dir(cache_dir: Path, namespace: str, label: int) -> list[LabeledTrial]:
    ns_dir = cache_dir / namespace
    if not ns_dir.exists():
        return []

    seen: dict[str, LabeledTrial] = {}
    for fp in ns_dir.glob("*.json"):
        entry = json.loads(fp.read_text())
        drug = entry.get("params", {}).get("drug", "")
        for raw in entry.get("data", {}).get("trials", []):
            try:
                trial = Trial(**raw)
            except Exception as exc:
                logger.warning("Skipping trial in %s: %s", fp.name, exc)
                continue
            if not trial.nct_id or trial.nct_id in seen:
                continue
            seen[trial.nct_id] = LabeledTrial(trial=trial, label=label, drug=drug)
    return list(seen.values())


def load_labeled_trials(
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[LabeledTrial]:
    """Load all labeled trials from the completed and terminated cache namespaces.

    Returns one LabeledTrial per unique NCT ID. If the same NCT ID somehow
    appears in both namespaces, terminated wins (it's the more conservative
    label for our use case — a flagged termination is more useful than a
    missed one).
    """
    completed = _load_dir(cache_dir, "ct_completed", label=0)
    terminated = _load_dir(cache_dir, "ct_terminated", label=1)

    by_nct: dict[str, LabeledTrial] = {lt.trial.nct_id: lt for lt in completed}
    for lt in terminated:
        by_nct[lt.trial.nct_id] = lt

    logger.info(
        "Loaded %d labeled trials (%d terminated, %d completed)",
        len(by_nct),
        sum(1 for lt in by_nct.values() if lt.label == 1),
        sum(1 for lt in by_nct.values() if lt.label == 0),
    )
    return list(by_nct.values())
