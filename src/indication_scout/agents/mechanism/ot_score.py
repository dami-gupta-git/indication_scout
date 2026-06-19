"""Local recomputation of Open Targets' overall association score.

Pure functions, no I/O, no OT client import. OT's overall score is a weighted harmonic sum over
per-datasource scores, normalized by the harmonic sum of a fixed number of ones. We reproduce it so
we can drop the `clinical_precedence` datasource in holdout mode — that channel encodes current
trials/approvals and would leak post-cutoff information into the mechanism ranking.

Reproduces the published overall_score to ~0.01 mean abs error (verified against JAK1/JAK2).
"""

from indication_scout.constants import (
    OT_DEFAULT_DATASOURCE_WEIGHT,
    OT_DATASOURCE_WEIGHTS,
    OT_PLATFORM_DATASOURCE_COUNT,
)


def _harmonic_sum(values: list[float]) -> float:
    """Sum of values sorted descending, each divided by (rank+1)**2."""
    ordered = sorted(values, reverse=True)
    return sum(v / ((i + 1) ** 2) for i, v in enumerate(ordered))


def recompute_overall(
    datasource_scores: dict[str, float],
    exclude: set[str],
) -> float:
    """Recompute OT's overall association score from per-datasource scores, dropping `exclude` ids.

    Weights each datasource by OT_DATASOURCE_WEIGHTS (default OT_DEFAULT_DATASOURCE_WEIGHT), takes the
    harmonic sum, and normalizes by the harmonic sum of OT_PLATFORM_DATASOURCE_COUNT ones. Returns 0.0
    when no datasources remain after exclusion.
    """
    weighted = [
        OT_DATASOURCE_WEIGHTS.get(ds_id, OT_DEFAULT_DATASOURCE_WEIGHT) * score
        for ds_id, score in datasource_scores.items()
        if ds_id not in exclude
    ]
    if not weighted:
        return 0.0
    normalizer = _harmonic_sum([1.0] * OT_PLATFORM_DATASOURCE_COUNT)
    return _harmonic_sum(weighted) / normalizer
