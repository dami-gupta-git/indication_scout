"""Integration test for mechanism_candidates.classify_positive against real
Open Targets direction evidence.

Verifies: for known (target, disease) pairs with known direction evidence in
OT, combining that evidence with the canonical drug action returns the
expected verdict. Unit tests cover the full truth table with synthetic inputs;
this test confirms the same logic holds when fed evidence fetched live from OT.
"""

import pytest

from indication_scout.agents.mechanism.mechanism_candidates import (
    aggregate_directions,
    classify_positive,
)


@pytest.mark.parametrize(
    "target_id,efo_id,action,expected,note",
    [
        # Semaglutide / GLP1R / T2D — evidence is GoF/protect, agonist aligns
        # with the implied LoF-driven disease mechanism → POSITIVE.
        (
            "ENSG00000112164",
            "MONDO_0005148",
            "AGONIST",
            True,
            "agonist vs. LoF-driven T2D (dirT=GoF, dirTrait=protect)",
        ),
        # Same pair but pretend the drug is an inhibitor — same direction as
        # the disease mechanism (both reduce GLP1R activity) → NEGATIVE.
        (
            "ENSG00000112164",
            "MONDO_0005148",
            "INHIBITOR",
            False,
            "inhibitor vs. LoF-driven T2D = contraindication",
        ),
        # Bupropion / SLC6A3 / infantile dystonia-parkinsonism — evidence is
        # LoF/risk. Bupropion inhibits SLC6A3 → matches disease mechanism →
        # NEGATIVE (contraindication).
        (
            "ENSG00000142319",
            "Orphanet_238455",
            "INHIBITOR",
            False,
            "inhibitor vs. LoF-driven infantile dystonia-parkinsonism = contraindication",
        ),
        # Same pair with an agonist — opposite direction of the disease
        # mechanism → POSITIVE (hypothetical, but the classifier should
        # call it correctly).
        (
            "ENSG00000142319",
            "Orphanet_238455",
            "AGONIST",
            True,
            "agonist vs. LoF-driven infantile dystonia-parkinsonism = aligned",
        ),
    ],
)
async def test_classify_positive_against_live_evidence(
    open_targets_client, target_id, efo_id, action, expected, note
):
    """Fetch real evidence for (target, disease), aggregate direction fields,
    feed into classify_positive with the given action, assert the verdict."""
    ev_map = await open_targets_client.get_target_evidences(target_id, [efo_id])
    records = ev_map[efo_id]
    assert records, f"expected evidence records for {target_id} / {efo_id}"

    # Majority-vote aggregation — robust to outlier records contradicting
    # a dominant direction (e.g. 146 GoF/protect + 1 LoF/risk on GLP1R/T2D).
    dir_targets, dir_traits = aggregate_directions(records)
    assert dir_targets and dir_traits, (
        f"expected consensus direction for {target_id} / {efo_id}; "
        f"got dir_targets={dir_targets}, dir_traits={dir_traits}"
    )
    assert classify_positive({action}, dir_targets, dir_traits) is expected, note
