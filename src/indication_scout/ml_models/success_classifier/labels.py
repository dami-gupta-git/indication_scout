"""Label extraction for the success classifier.

Each (target_id, disease_id) pair gets a binary label derived from its
`clinical` evidence records in `cache/target_evidences/`. The clinical
records are stripped from the pair before features are computed, so the
model never sees its own answer.

Threshold: max_clinical_score >= 0.7 → label = 1 (Phase II+ activity).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLINICAL_THRESHOLD = 0.7
CLINICAL_DATATYPE = "clinical"


@dataclass
class LabeledPair:
    """One (target, disease) pair with its label and the non-clinical records."""

    target_id: str
    disease_id: str
    label: int
    max_clinical_score: float
    n_clinical_records: int
    non_clinical_records: list[dict]


def _max_clinical_score(records: list[dict]) -> tuple[float, int]:
    """Return (max_score, count) over `clinical` records. (0.0, 0) if none."""
    clinicals = [r for r in records if r.get("datatype_id") == CLINICAL_DATATYPE]
    if not clinicals:
        return 0.0, 0
    scores = [r.get("score") or 0.0 for r in clinicals]
    return float(max(scores)), len(clinicals)


def load_labeled_pairs(cache_dir: Path) -> list[LabeledPair]:
    """Walk `cache/target_evidences/*.json` and yield one LabeledPair per
    (target, disease) entry.

    The pair's `clinical` records are removed from `non_clinical_records` —
    only those non-clinical records flow into feature extraction.
    """
    target_dir = cache_dir / "target_evidences"
    if not target_dir.is_dir():
        raise FileNotFoundError(f"target_evidences cache dir not found: {target_dir}")

    pairs: list[LabeledPair] = []
    for path in sorted(target_dir.glob("*.json")):
        with path.open() as fh:
            blob = json.load(fh)
        target_id = blob.get("target_id")
        if not target_id:
            logger.warning("Skipping %s: missing target_id", path.name)
            continue
        for disease_id, payload in (blob.get("entries") or {}).items():
            records = payload.get("records") or []
            max_score, n_clinical = _max_clinical_score(records)
            label = 1 if max_score >= CLINICAL_THRESHOLD else 0
            non_clinical = [
                r for r in records if r.get("datatype_id") != CLINICAL_DATATYPE
            ]
            pairs.append(
                LabeledPair(
                    target_id=target_id,
                    disease_id=disease_id,
                    label=label,
                    max_clinical_score=max_score,
                    n_clinical_records=n_clinical,
                    non_clinical_records=non_clinical,
                )
            )
    logger.info(
        "Loaded %d (target, disease) pairs from %s; %d positive (label=1).",
        len(pairs),
        target_dir,
        sum(p.label for p in pairs),
    )
    return pairs
