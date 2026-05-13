"""Feature extraction for the trial-risk classifier.

A feature row is a flat dict[str, float]. Categorical fields (phase,
sponsor_class) are emitted as one-hot indicator columns. The trainer
discovers the full feature vocabulary by unioning keys across all training
rows; missing keys are treated as 0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.ml_models.trial_risk.literature import LiteratureSignals

PHASES = [
    "Early Phase 1",
    "Phase 1",
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3",
    "Phase 3",
    "Phase 4",
    "Not Applicable",
]

_INDUSTRY_RE = re.compile(
    r"\b(inc|corp|corporation|pharma|pharmaceuticals?|biotech|llc|ltd|gmbh|ag|sa|co)\b",
    re.IGNORECASE,
)
_ACADEMIC_RE = re.compile(
    r"\b(university|college|hospital|institute|school|medical center|clinic)\b",
    re.IGNORECASE,
)
_NIH_RE = re.compile(r"\bnational\b|\bNIH\b|\bN[CIE][A-Z]{1,3}\b", re.IGNORECASE)


def sponsor_class(sponsor: str) -> str:
    if not sponsor:
        return "unknown"
    if _NIH_RE.search(sponsor):
        return "nih"
    if _INDUSTRY_RE.search(sponsor):
        return "industry"
    if _ACADEMIC_RE.search(sponsor):
        return "academic"
    return "other"


def parse_year(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(s.split("-")[0])
    except (ValueError, IndexError):
        return None


@dataclass
class FeatureRow:
    """One flat feature row plus the NCT ID for traceability."""

    nct_id: str
    features: dict[str, float]


def build_features(
    trial: Trial,
    lit: LiteratureSignals,
    include_fingerprint: bool = False,
) -> FeatureRow:
    """Build the feature vector for one trial + its precomputed lit signals.

    If `include_fingerprint`, also emit `fp_0..fp_{N-1}` columns from the
    mean-pooled BioLORD embedding (768-dim).
    """
    f: dict[str, float] = {}

    # Phase one-hot
    for p in PHASES:
        f[f"phase__{p}"] = 1.0 if trial.phase == p else 0.0

    # Sponsor class one-hot
    sc = sponsor_class(trial.sponsor)
    for cls in ("industry", "academic", "nih", "other", "unknown"):
        f[f"sponsor__{cls}"] = 1.0 if sc == cls else 0.0

    # Enrollment
    f["has_enrollment"] = 1.0 if trial.enrollment is not None else 0.0
    f["log_enrollment"] = (
        math.log1p(trial.enrollment) if trial.enrollment is not None else 0.0
    )

    # Counts
    f["n_mesh_ancestors"] = float(len(trial.mesh_ancestors))
    f["n_interventions"] = float(len(trial.interventions))
    f["n_primary_outcomes"] = float(len(trial.primary_outcomes))

    # Start year
    yr = parse_year(trial.start_date)
    f["has_start_date"] = 1.0 if yr is not None else 0.0
    f["start_year"] = float(yr) if yr is not None else 0.0

    # Literature signals
    f["lit_failure_signal"] = lit.failure_signal
    f["lit_safety_signal"] = lit.safety_signal
    f["lit_efficacy_signal"] = lit.efficacy_signal
    f["lit_signal_available"] = 1.0 if lit.available else 0.0

    if include_fingerprint and lit.embedding_fingerprint:
        for i, v in enumerate(lit.embedding_fingerprint):
            f[f"fp_{i:03d}"] = float(v)

    return FeatureRow(nct_id=trial.nct_id, features=f)


def vectorize(rows: list[FeatureRow]) -> tuple[list[str], list[list[float]]]:
    """Convert FeatureRows to a (column_names, matrix) pair for sklearn.

    Column order is sorted alphabetically so the order is stable across runs.
    Missing keys in any row are filled with 0.0.
    """
    columns = sorted({k for r in rows for k in r.features.keys()})
    matrix = [[r.features.get(c, 0.0) for c in columns] for r in rows]
    return columns, matrix
