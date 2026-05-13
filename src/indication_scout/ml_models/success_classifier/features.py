"""Feature extraction for the success classifier.

One (target, disease) pair → one flat dict[str, float]. Features come
from non-clinical Open Targets evidence records on the pair (see
`labels.py` for the masking step that removes clinical records before
features are computed).

Per-datatype features (count, max_score, mean_score) plus a few
cross-datatype indicators. The trainer discovers the column vocabulary
by unioning keys across rows; missing keys are treated as 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# All non-clinical datatypes that appear in `_cache/target_evidences/`.
# Stable order so feature names are stable across runs.
NON_CLINICAL_DATATYPES = [
    "genetic_association",
    "genetic_literature",
    "somatic_mutation",
    "animal_model",
    "rna_expression",
    "affected_pathway",
    "literature",
]

GENETIC_DATATYPES = {
    "genetic_association",
    "genetic_literature",
    "somatic_mutation",
}


@dataclass
class FeatureRow:
    """One flat feature row plus the (target, disease) ids for traceability."""

    target_id: str
    disease_id: str
    features: dict[str, float]


def build_features(
    target_id: str,
    disease_id: str,
    non_clinical_records: list[dict],
) -> FeatureRow:
    """Build the feature vector for one pair from its non-clinical records."""
    f: dict[str, float] = {}

    # Per-datatype: count, max_score, mean_score.
    for dt in NON_CLINICAL_DATATYPES:
        rs = [r for r in non_clinical_records if r.get("datatype_id") == dt]
        scores = [float(r.get("score") or 0.0) for r in rs]
        f[f"n_{dt}"] = float(len(rs))
        f[f"max_score_{dt}"] = max(scores) if scores else 0.0
        f[f"mean_score_{dt}"] = (sum(scores) / len(scores)) if scores else 0.0

    # Cross-datatype indicators.
    present_dts = {
        r.get("datatype_id") for r in non_clinical_records if r.get("datatype_id")
    }
    f["n_distinct_datatypes"] = float(len(present_dts & set(NON_CLINICAL_DATATYPES)))
    f["has_any_genetic"] = 1.0 if (present_dts & GENETIC_DATATYPES) else 0.0
    f["log_total_evidence"] = math.log1p(len(non_clinical_records))

    # Direction signal: fraction of records with `direction_on_trait == "protect"`,
    # which proxies "loss of function helps." Computed across datatypes that carry
    # the field (animal_model, clinical — but clinical is masked here).
    directional = [
        r
        for r in non_clinical_records
        if r.get("direction_on_trait") in ("protect", "risk")
    ]
    if directional:
        n_protect = sum(
            1 for r in directional if r.get("direction_on_trait") == "protect"
        )
        f["frac_direction_protect"] = n_protect / len(directional)
        f["has_direction_signal"] = 1.0
    else:
        f["frac_direction_protect"] = 0.0
        f["has_direction_signal"] = 0.0

    return FeatureRow(target_id=target_id, disease_id=disease_id, features=f)


def vectorize(rows: list[FeatureRow]) -> tuple[list[str], list[list[float]]]:
    """Convert FeatureRows to a (column_names, matrix) pair for sklearn.

    Column order is sorted alphabetically so the order is stable across runs.
    Missing keys in any row are filled with 0.0.
    """
    columns = sorted({k for r in rows for k in r.features.keys()})
    matrix = [[r.features.get(c, 0.0) for c in columns] for r in rows]
    return columns, matrix


# Group columns by datatype so the trainer can run ablations cleanly.
def column_groups(columns: list[str]) -> dict[str, list[str]]:
    """Map ablation-group name → columns that belong to that group.

    Per-datatype groups cover the three (count, max, mean) columns for
    each datatype. The "cross" group covers cross-datatype indicators.
    """
    groups: dict[str, list[str]] = {dt: [] for dt in NON_CLINICAL_DATATYPES}
    groups["cross"] = []
    for c in columns:
        matched = False
        for dt in NON_CLINICAL_DATATYPES:
            if c.endswith(f"_{dt}") or c == f"n_{dt}":
                groups[dt].append(c)
                matched = True
                break
        if not matched:
            groups["cross"].append(c)
    return groups
