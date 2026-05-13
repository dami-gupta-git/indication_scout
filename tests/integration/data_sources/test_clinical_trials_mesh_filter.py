"""Regression: MeSH resolver -> CT.gov server-side condition filter.

For a small fixed set of drug-indication pairs, resolves the indication to
its MeSH preferred term via the resolver, then calls `search_trials` with
that term. The client passes it as `AREA[ConditionMeshTerm]"<term>"` so
filtering happens server-side; there is no longer a pre/post comparison.

Assertions are sanity checks on the resolver + client pipeline:
  - resolver returns a (descriptor_id, preferred_term) tuple for each
    indication
  - `len(result.trials) <= result.total_count` (top-50 fetch never exceeds
    total)
  - at least len(_PAIRS) - 1 pairs return >0 trials (the pipeline isn't
    catastrophically empty for known-active pairs). Threshold scales so
    pairs can be commented out locally; minimum of 1.

A `date_before=2025-01-01` cutoff keeps counts roughly stable; counts may
still drift as CT.gov re-tags trials.
"""

import logging
from datetime import date

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.services.disease_helper import resolve_mesh_id

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)

_PAIRS: list[tuple[str, str]] = [
    ("metformin", "hypertension"),
    ("aspirin", "diabetes mellitus"),
    ("semaglutide", "hypertension"),
]


async def test_search_trials_with_resolved_mesh_term(
    clinical_trials_client: ClinicalTrialsClient,
):
    results: list[tuple[str, str, str, str, int, int]] = []

    for drug, indication in _PAIRS:
        resolved = await resolve_mesh_id(indication)
        assert resolved is not None, f"resolver returned None for {indication!r}"
        descriptor_id, preferred_term = resolved

        result = await clinical_trials_client.search_trials(
            drug=drug, mesh_term=preferred_term, date_before=_CUTOFF
        )

        total, fetched = result.total_count, len(result.trials)
        results.append(
            (drug, indication, descriptor_id, preferred_term, total, fetched)
        )

        logger.info(
            "search_trials: %s x %s (mesh=%s/%r) — total=%d fetched=%d",
            drug,
            indication,
            descriptor_id,
            preferred_term,
            total,
            fetched,
        )

        # Fetch is capped by CLINICAL_TRIALS_FETCH_MAX; never exceeds total.
        assert fetched <= total, (
            f"{drug} x {indication}: fetched {fetched} exceeds total {total}"
        )

    # Allow at most one pair to return zero trials (e.g. resolver mis-pick
    # on an ambiguous indication). If more than one is empty, the pipeline
    # is broken across the board.
    nonempty = sum(1 for *_, total, _ in results if total > 0)
    min_nonempty = max(1, len(results) - 1)
    assert nonempty >= min_nonempty, (
        f"only {nonempty}/{len(results)} pairs returned trials: {results}"
    )
