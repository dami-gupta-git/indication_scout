"""Train the trial-termination risk classifier.

Loads labeled trials from `cache/`, computes date-bounded literature signals
via the existing retrieval pipeline, fits a calibrated logistic regression,
runs leave-one-drug-out cross-validation, and saves the artifact + metrics
under `models/`.

Run:
    python -m indication_scout.ml_models.trial_risk.train
    python -m indication_scout.ml_models.trial_risk.train --lookback-months 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import get_db
from indication_scout.ml_models.trial_risk.data import LabeledTrial, load_labeled_trials
from indication_scout.ml_models.trial_risk.features import FeatureRow, build_features, vectorize
from indication_scout.ml_models.trial_risk.literature import (
    DEFAULT_LOOKBACK_MONTHS,
    LiteratureSignals,
    signals_for_trial,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "trial_risk_v1.pkl"
METRICS_PATH = MODELS_DIR / "trial_risk_v1.metrics.json"


CONCURRENCY = 4  # number of trials processed in parallel


def _build_estimator(model_name: str):
    """Construct one of the experimental classifiers."""
    if model_name == "lr":
        # Aggressive shrinkage for high-dim fingerprint case (n=303, p~792).
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l2",
                        C=0.01,
                        class_weight="balanced",
                        max_iter=2000,
                    ),
                ),
            ]
        )
    if model_name == "lr-pca":
        # PCA to 32 components, then LR.
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=32, random_state=0)),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                    ),
                ),
            ]
        )
    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
            ]
        )
    raise ValueError(f"unknown model: {model_name}")


async def _process_one(
    lt: LabeledTrial,
    cache_dir: Path,
    lookback_months: int,
    sem: asyncio.Semaphore,
    include_fingerprint: bool,
) -> FeatureRow:
    """Compute lit signals for one trial under a semaphore. Each task gets its own DB session."""
    async with sem:
        db_gen = get_db()
        db = next(db_gen)
        try:
            try:
                lit = await signals_for_trial(
                    lt.trial,
                    drug=lt.drug,
                    db=db,
                    cache_dir=cache_dir,
                    lookback_months=lookback_months,
                )
            except Exception as exc:
                logger.warning(
                    "Lit signal failed for %s (%s): %s — using empty signals",
                    lt.trial.nct_id,
                    lt.drug,
                    exc,
                )
                lit = LiteratureSignals()
        finally:
            db.close()
        return build_features(lt.trial, lit, include_fingerprint=include_fingerprint)


async def _build_feature_rows(
    labeled: list[LabeledTrial],
    cache_dir: Path,
    lookback_months: int,
    include_fingerprint: bool,
) -> list[FeatureRow]:
    """Compute lit signals + assemble features for every labeled trial (bounded concurrency)."""
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[FeatureRow | None] = [None] * len(labeled)
    completed = 0
    total = len(labeled)

    async def _wrapped(idx: int, lt: LabeledTrial) -> None:
        nonlocal completed
        rows[idx] = await _process_one(
            lt, cache_dir, lookback_months, sem, include_fingerprint
        )
        completed += 1
        if completed % 25 == 0:
            logger.info("Built features for %d / %d trials", completed, total)

    await asyncio.gather(*(_wrapped(i, lt) for i, lt in enumerate(labeled)))
    return [r for r in rows if r is not None]


def _grouped_cv_scores(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str
) -> dict[str, float | list[dict]]:
    """Leave-one-drug-out CV. Returns aggregate metrics + per-fold breakdown."""
    logo = LeaveOneGroupOut()
    all_y, all_p = [], []
    folds: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            # Fold has only one class in train — skip, can't fit a binary model.
            continue

        base = _build_estimator(model_name)
        # Calibration needs >=2 samples per class in train; fall back to bare pipeline if not.
        # KNN is also not great for calibration with cv=5 — skip wrapper for it.
        train_class_counts = np.bincount(y_train, minlength=2)
        if model_name != "knn" and train_class_counts.min() >= 2:
            cv_folds = min(5, int(train_class_counts.min()))
            model = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
        else:
            model = base

        model.fit(X[train_idx], y_train)
        p_test = model.predict_proba(X[test_idx])[:, 1]
        all_y.extend(y_test.tolist())
        all_p.extend(p_test.tolist())

        held_out_drug = str(groups[test_idx][0])
        folds.append(
            {
                "fold": fold_idx,
                "held_out_drug": held_out_drug,
                "n_test": int(len(y_test)),
                "n_test_terminated": int(y_test.sum()),
            }
        )

    y_arr = np.array(all_y)
    p_arr = np.array(all_p)
    metrics: dict[str, float | list[dict]] = {
        "n_trials_scored": int(len(y_arr)),
        "class_balance": float(y_arr.mean()) if len(y_arr) else 0.0,
        "roc_auc": (
            float(roc_auc_score(y_arr, p_arr))
            if len(np.unique(y_arr)) > 1
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_arr, p_arr))
            if len(np.unique(y_arr)) > 1
            else float("nan")
        ),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "folds": folds,
    }
    return metrics


def _fit_final_model(X: np.ndarray, y: np.ndarray, model_name: str):
    base = _build_estimator(model_name)
    train_class_counts = np.bincount(y, minlength=2)
    if model_name != "knn" and train_class_counts.min() >= 2:
        cv_folds = min(5, int(train_class_counts.min()))
        model = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
    else:
        model = base
    model.fit(X, y)
    return model


async def main_async(
    cache_dir: Path,
    lookback_months: int,
    drugs_filter: set[str] | None,
    exclude_phases: set[str] | None,
    dry_run: bool,
    use_fingerprint: bool,
    model_name: str,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    labeled = load_labeled_trials(cache_dir)
    if drugs_filter:
        labeled = [lt for lt in labeled if lt.drug in drugs_filter]
        logger.info(
            "Filtered to drugs %s: %d trials remain", sorted(drugs_filter), len(labeled)
        )
    if exclude_phases:
        before = len(labeled)
        labeled = [lt for lt in labeled if lt.trial.phase not in exclude_phases]
        logger.info(
            "Excluded phases %s: %d -> %d trials",
            sorted(exclude_phases),
            before,
            len(labeled),
        )
    if not labeled:
        logger.error(
            "No labeled trials found in %s (filter=%s)", cache_dir, drugs_filter
        )
        return

    rows = await _build_feature_rows(
        labeled, cache_dir, lookback_months, use_fingerprint
    )
    if dry_run:
        logger.info(
            "Dry run: built features for %d trials, exiting before fit.", len(rows)
        )
        return
    columns, matrix = vectorize(rows)
    X = np.array(matrix, dtype=float)
    y = np.array([lt.label for lt in labeled], dtype=int)
    groups = np.array([lt.drug for lt in labeled])

    logger.info(
        "Feature matrix: %d rows x %d cols. Class balance: %.3f. Drugs: %d.",
        X.shape[0],
        X.shape[1],
        y.mean(),
        len(set(groups.tolist())),
    )

    metrics = _grouped_cv_scores(X, y, groups, model_name)
    metrics["model"] = model_name
    metrics["use_fingerprint"] = use_fingerprint
    logger.info(
        "CV metrics [model=%s]: ROC-AUC=%.3f PR-AUC=%.3f Brier=%.3f baseline=%.3f",
        model_name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["brier"],
        metrics["class_balance"],
    )

    if metrics["pr_auc"] <= metrics["class_balance"]:
        logger.error(
            "PR-AUC %.3f does not beat class-balance baseline %.3f. Refusing to ship artifact.",
            metrics["pr_auc"],
            metrics["class_balance"],
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        return

    model = _fit_final_model(X, y, model_name)

    # Coefficient inspection only meaningful for linear models without PCA.
    if model_name == "lr":
        inspect_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l2",
                        C=0.01,
                        class_weight="balanced",
                        max_iter=2000,
                    ),
                ),
            ]
        )
        inspect_pipe.fit(X, y)
        inspect_lr = inspect_pipe.named_steps["lr"]
        coefs = sorted(
            zip(columns, inspect_lr.coef_[0].tolist()),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        metrics["coefficients"] = [{"feature": c, "weight": w} for c, w in coefs]
        metrics["intercept"] = float(inspect_lr.intercept_[0])
        logger.info("Top coefficients (scaled LR C=0.01, |weight| desc):")
        for feat, w in coefs[:20]:
            logger.info("  %+0.3f  %s", w, feat)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(
            {"model": model, "columns": columns, "lookback_months": lookback_months}, f
        )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote artifact %s and metrics %s", ARTIFACT_PATH, METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trial-risk classifier.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Path to the IndicationScout cache directory (default: project cache/).",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=DEFAULT_LOOKBACK_MONTHS,
        help=f"Months before completion_date to cap literature lookups (default: {DEFAULT_LOOKBACK_MONTHS}).",
    )
    parser.add_argument(
        "--drugs",
        type=str,
        default=None,
        help="Comma-separated drug names to restrict training to (matches the cache `drug` param).",
    )
    parser.add_argument(
        "--exclude-phases",
        type=str,
        default=None,
        help="Comma-separated trial phases to exclude (e.g. 'Phase 1,Early Phase 1').",
    )
    parser.add_argument(
        "--use-fingerprint",
        action="store_true",
        help="Add 768-dim mean-pooled BioLORD fingerprint as features.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "lr-pca", "knn"],
        help="Classifier: lr (L2 C=0.01), lr-pca (PCA-32 then LR), knn (cosine k=5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build features (warming the cache) but skip model fitting.",
    )
    args = parser.parse_args()
    drugs_filter = (
        {d.strip() for d in args.drugs.split(",") if d.strip()} if args.drugs else None
    )
    exclude_phases = (
        {p.strip() for p in args.exclude_phases.split(",") if p.strip()}
        if args.exclude_phases
        else None
    )
    asyncio.run(
        main_async(
            args.cache_dir,
            args.lookback_months,
            drugs_filter,
            exclude_phases,
            args.dry_run,
            args.use_fingerprint,
            args.model,
        )
    )


if __name__ == "__main__":
    main()
