"""Integration tests for mechanism_candidates against real Open Targets data.

Hits the live OT client to build `rows` and then runs the pure-function
classifier against them. Verifies the classifier produces sensible POSITIVE
candidates on known agonist / inhibitor cases and correctly excludes
LoF-syndrome contraindications where the drug's direction matches the
disease mechanism.
"""

import pytest

from indication_scout.agents.mechanism.mechanism_candidates import (
    aggregate_directions,
    select_top_candidates,
)

# OT's GraphQL endpoint intermittently returns "Internal server error" 500s.
# Rerun only on that transient transport failure, not on real assertion failures.
pytestmark = pytest.mark.flaky(reruns=2, reruns_delay=1, only_rerun=["DataSourceError"])


async def _build_rows(
    ot_client,
    target_id: str,
    action_types: set[str],
    top_n: int = 15,
) -> list[dict]:
    """Fetch associations + evidences for a single target and assemble the
    row dicts `select_top_candidates` expects. Kept local so the classifier
    stays decoupled from OT in production code — agent wiring in the next
    step will own this assembly."""
    target = await ot_client.get_target_data(target_id)
    target_function = (
        target.function_descriptions[0] if target.function_descriptions else ""
    )
    top = sorted(
        target.associations, key=lambda a: a.overall_score or 0.0, reverse=True
    )[:top_n]
    efo_ids = [a.disease_id for a in top]
    ev_map = await ot_client.get_target_evidences(target_id, efo_ids)

    rows = []
    for a in top:
        rows.append(
            {
                "target_symbol": target.symbol,
                "action_types": action_types,
                "disease_name": a.disease_name,
                "overall_score": a.overall_score,
                "evidences": ev_map.get(a.disease_id, []),
                "disease_description": a.disease_description,
                "target_function": target_function,
            }
        )
    return rows


@pytest.mark.parametrize(
    "target_id,action,approved",
    [
        # Semaglutide / GLP1R (AGONIST) — T2D approved, repurposing
        # candidates (e.g. obesity-adjacent) should survive the filter.
        ("ENSG00000112164", "AGONIST", {"type 2 diabetes mellitus"}),
        # Bupropion / SLC6A3 (INHIBITOR) — MDD approved, off-label
        # candidates (ADHD, etc.) should surface.
        ("ENSG00000142319", "INHIBITOR", {"major depressive disorder"}),
        # Atorvastatin / HMGCR (INHIBITOR) — hypercholesterolemia approved,
        # non-approved GoF-driven disease candidates should survive.
        ("ENSG00000113161", "INHIBITOR", {"hypercholesterolemia", "hyperlipidemia"}),
        # Adalimumab / TNF (INHIBITOR) — RA approved, psoriasis / IBD /
        # other inflammatory candidates should surface.
        ("ENSG00000232810", "INHIBITOR", {"rheumatoid arthritis"}),
        # Tofacitinib / TYK2 (INHIBITOR) — RA approved, psoriasis and
        # other JAK-family-responsive inflammatory diseases should surface.
        ("ENSG00000105397", "INHIBITOR", {"rheumatoid arthritis"}),
        # Losartan / AGTR1 (ANTAGONIST) — hypertension approved, other
        # LoF/protect cardiovascular / renal candidates should survive.
        ("ENSG00000144891", "ANTAGONIST", {"hypertension"}),
    ],
)
async def test_positive_candidates_surface_and_exclude_approved(
    open_targets_client, target_id, action, approved
):
    """For each (target, action) pair: at least one POSITIVE candidate
    survives; no candidate disease-name exactly matches the approved set
    (case-insensitive); and each candidate carries the expected action
    plus non-empty target_function text.

    Exact-match only — synonym / parent-term expansion is the caller's
    problem (services.approval_check.get_fda_approved_disease_mapping). So
    'familial hypercholesterolemia' can survive even when
    'hypercholesterolemia' is in the approved set — that's the contract.
    """
    rows = await _build_rows(open_targets_client, target_id, {action}, top_n=15)

    candidates = select_top_candidates(rows, approved_diseases=approved, limit=5)
    assert candidates, f"expected at least one POSITIVE candidate for {target_id}"

    names_lower = {c.disease_name.lower() for c in candidates}
    approved_lower = {a.lower() for a in approved}
    leaked_exact = names_lower & approved_lower
    assert (
        not leaked_exact
    ), f"approved terms leaked into candidates via exact match: {leaked_exact}"

    for c in candidates:
        assert c.action_type == action
        assert c.target_function  # same target → same function text on every row

    # Not every EFO node has a description in OT, so don't require it on
    # every candidate — just confirm text plumbing works on at least one.
    assert any(c.disease_description for c in candidates)


@pytest.mark.parametrize(
    "target_id,action,lof_disease_substring",
    [
        # Bupropion / SLC6A3 (INHIBITOR) — LoF syndrome = infantile
        # dystonia-parkinsonism. Inhibitor matches the disease mechanism =>
        # contraindication, must not appear as a candidate.
        ("ENSG00000142319", "INHIBITOR", "infantile dystonia-parkinsonism"),
        # Tofacitinib / JAK3 (INHIBITOR) — LoF syndrome = JAK3-deficient SCID.
        ("ENSG00000105639", "INHIBITOR", "severe combined immunodeficiency"),
        # Imatinib / KIT (INHIBITOR) — LoF syndrome = piebaldism, caused by
        # LoF KIT variants. Inhibiting KIT would mimic the disease mechanism.
        ("ENSG00000157404", "INHIBITOR", "piebaldism"),
        # Hypothetical propranolol-adjacent / AGTR1 (ANTAGONIST) — LoF
        # syndrome = renal tubular dysgenesis, caused by LoF AGTR1 variants.
        # Antagonizing AGTR1 matches the disease mechanism and is a known
        # teratogen in pregnancy for this reason.
        ("ENSG00000144891", "ANTAGONIST", "renal tubular dysgenesis"),
    ],
)
async def test_lof_syndromes_excluded_for_inhibitor_like_drugs(
    open_targets_client, target_id, action, lof_disease_substring
):
    """For each LoF-class drug + target with a known LoF/risk Mendelian
    syndrome, confirm the syndrome appears in the raw rows with LoF/risk
    evidence but does NOT appear in the POSITIVE candidates (classifier
    correctly flags it as a contraindication)."""
    rows = await _build_rows(open_targets_client, target_id, {action}, top_n=15)

    lof_row = next(
        (r for r in rows if lof_disease_substring in r["disease_name"].lower()),
        None,
    )
    assert (
        lof_row is not None
    ), f"expected {lof_disease_substring!r} in top rows for {target_id}"
    # Majority-voted direction on the LoF syndrome's evidence must be LoF/risk.
    dir_targets, dir_traits = aggregate_directions(lof_row["evidences"])
    assert dir_targets == {"LoF"} and dir_traits == {"risk"}, (
        f"expected LoF/risk majority on {lof_disease_substring!r}, "
        f"got dir_targets={dir_targets}, dir_traits={dir_traits}"
    )

    candidates = select_top_candidates(rows, approved_diseases=set(), limit=5)
    names_lower = {c.disease_name.lower() for c in candidates}
    assert not any(
        lof_disease_substring in n for n in names_lower
    ), f"contraindication {lof_disease_substring!r} leaked into candidates: {names_lower}"
