"""Score trials with the trained trial-risk classifier.

Run:
    python -m indication_scout.ml_models.trial_risk.score NCT00064337 NCT01234567
    python -m indication_scout.ml_models.trial_risk.score --all-terminated
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
from pathlib import Path

import numpy as np

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import get_db
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.ml_models.trial_risk.data import LabeledTrial, load_labeled_trials
from indication_scout.ml_models.trial_risk.features import build_features
from indication_scout.ml_models.trial_risk.literature import (
    LiteratureSignals,
    signals_for_trial,
)
from indication_scout.ml_models.trial_risk.train import ARTIFACT_PATH

logger = logging.getLogger(__name__)


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained artifact at {path}. Run `python -m indication_scout.ml_models.trial_risk.train` first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


async def score_trial_async(
    trial: Trial,
    drug: str,
    artifact: dict,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> float:
    """Return the predicted probability the trial will be terminated."""
    db_gen = get_db()
    db = next(db_gen)
    try:
        try:
            lit = await signals_for_trial(
                trial,
                drug=drug,
                db=db,
                cache_dir=cache_dir,
                lookback_months=artifact["lookback_months"],
            )
        except Exception as exc:
            logger.warning(
                "Lit signal failed for %s: %s — using empty signals", trial.nct_id, exc
            )
            lit = LiteratureSignals()
    finally:
        db.close()

    row = build_features(trial, lit)
    columns = artifact["columns"]
    x = np.array([[row.features.get(c, 0.0) for c in columns]], dtype=float)
    return float(artifact["model"].predict_proba(x)[0, 1])


def _resolve_targets(
    nct_ids: list[str], all_labeled: list[LabeledTrial]
) -> list[LabeledTrial]:
    by_nct = {lt.trial.nct_id: lt for lt in all_labeled}
    missing = [n for n in nct_ids if n not in by_nct]
    if missing:
        logger.warning("NCT IDs not found in cache: %s", missing)
    return [by_nct[n] for n in nct_ids if n in by_nct]


async def main_async(
    nct_ids: list[str],
    all_terminated: bool,
    cache_dir: Path,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    artifact = load_artifact()
    labeled = load_labeled_trials(cache_dir)

    if all_terminated:
        targets = [lt for lt in labeled if lt.label == 1]
    else:
        targets = _resolve_targets(nct_ids, labeled)

    if not targets:
        logger.error("No trials to score.")
        return

    print(f"{'NCT_ID':<14}  {'P(term)':>7}  {'label':>5}  {'phase':<18}  drug")
    for lt in targets:
        p = await score_trial_async(lt.trial, lt.drug, artifact, cache_dir)
        print(
            f"{lt.trial.nct_id:<14}  {p:>7.3f}  {lt.label:>5d}  {lt.trial.phase:<18}  {lt.drug}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score trials with trial-risk classifier."
    )
    parser.add_argument("nct_ids", nargs="*", help="NCT IDs to score.")
    parser.add_argument(
        "--all-terminated",
        action="store_true",
        help="Score every terminated trial in the cache (sanity check).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Path to the IndicationScout cache directory.",
    )
    args = parser.parse_args()

    if not args.nct_ids and not args.all_terminated:
        parser.error("Provide NCT IDs or --all-terminated.")

    asyncio.run(main_async(args.nct_ids, args.all_terminated, args.cache_dir))


if __name__ == "__main__":
    main()
