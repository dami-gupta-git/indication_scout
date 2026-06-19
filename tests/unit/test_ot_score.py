"""Unit tests for the leak-free OT overall-score recomputation.

These use a frozen offline fixture (no network). A live-OT drift check — which catches OT changing
its datasource count or weights — lives in tests/integration/test_open_targets.py.
"""

from indication_scout.agents.mechanism.ot_score import recompute_overall

# Real JAK1 × rheumatoid arthritis datasource scores (OT public API). Published overall_score for
# this row is 0.6871; recompute_overall with no exclusions should land within ~0.02 of it.
_RA_DATASOURCE_SCORES: dict[str, float] = {
    "clinical_precedence": 0.9907763938498343,
    "gwas_credible_sets": 0.524860902016387,
    "europepmc": 0.3724699275331018,
}
_RA_PUBLISHED_OVERALL = 0.6871251698480906


def test_recompute_reproduces_published_overall() -> None:
    recomputed = recompute_overall(_RA_DATASOURCE_SCORES, exclude=set())
    assert abs(recomputed - _RA_PUBLISHED_OVERALL) < 0.02


def test_excluding_clinical_precedence_lowers_clinical_dominated_score() -> None:
    full = recompute_overall(_RA_DATASOURCE_SCORES, exclude=set())
    leak_free = recompute_overall(
        _RA_DATASOURCE_SCORES, exclude={"clinical_precedence"}
    )
    # clinical_precedence is the dominant datasource here, so dropping it must lower the score.
    assert leak_free < full


def test_empty_after_exclusion_returns_zero() -> None:
    assert (
        recompute_overall(
            {"clinical_precedence": 0.99}, exclude={"clinical_precedence"}
        )
        == 0.0
    )


def test_downweighted_datasource_uses_published_weight() -> None:
    # europepmc has weight 0.2; a lone europepmc row should score far below its raw value.
    raw = 0.8
    scored = recompute_overall({"europepmc": raw}, exclude=set())
    assert scored < raw * 0.2  # weighted then normalized by the harmonic-of-ones constant
