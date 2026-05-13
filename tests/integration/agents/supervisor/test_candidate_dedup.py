"""Integration test for the hierarchical-dedup LLM helper — hits real Anthropic.

Verifies the rosiglitazone regression: UC ⊂ IBD and T2DM ⊂ DM each collapse
into a single survivor when the helper is asked to dedup the full merged
candidate list.
"""

import logging

from indication_scout.agents.supervisor.candidate_dedup import (
    run_hierarchical_dedup,
)

logger = logging.getLogger(__name__)


_ROSIGLITAZONE_CANDIDATES: list[tuple[str, str, str | None]] = [
    ("inflammatory bowel disease", "competitor", "EFO_0003767"),
    ("non-alcoholic steatohepatitis", "competitor", "EFO_1001249"),
    ("non-alcoholic fatty liver disease", "competitor", "EFO_0003095"),
    ("type 1 diabetes mellitus", "competitor", "EFO_0001359"),
    ("liver disease", "competitor", "EFO_0001421"),
    ("type 2 diabetes mellitus", "mechanism", "EFO_0001360"),
    ("pparg-related familial partial lipodystrophy", "mechanism", "Orphanet_79083"),
    ("diabetes mellitus", "mechanism", "EFO_0000400"),
    ("ulcerative colitis", "mechanism", "EFO_0000729"),
    ("morbid obesity", "mechanism", "EFO_0001074"),
]


async def test_rosiglitazone_hierarchical_dedup_collapses_uc_and_t2dm():
    """Real LLM must collapse UC + IBD into one entry, and T2DM + DM into one.

    Also expects NASH + NAFLD to collapse (NASH ⊂ NAFLD). Distinct subtypes
    of different parents (e.g. T1DM is a sibling of T2DM, not a sub/super
    of either) must NOT be grouped.

    The exact survivor each decision picks depends on the LLM's clinical
    judgment about which level is most actionable for a PPARγ agonist. The
    test asserts on the SET of dropped names per overlap rather than pinning
    the survivor, so the test stays meaningful as the prompt evolves.
    """
    output = await run_hierarchical_dedup(
        drug_name="rosiglitazone",
        mechanism_targets=[("PPARG", "AGONIST")],
        candidates=_ROSIGLITAZONE_CANDIDATES,
    )

    all_names = {name for name, _, _ in _ROSIGLITAZONE_CANDIDATES}
    all_dropped: set[str] = set()
    all_survivors: set[str] = set()
    for d in output.decisions:
        assert d.survivor in all_names, (
            f"survivor {d.survivor!r} not in candidate list — dedup helper "
            f"should have filtered this. decisions={output.decisions!r}"
        )
        for name in d.dropped:
            assert name in all_names, (
                f"dropped {name!r} not in candidate list — dedup helper "
                f"should have filtered this"
            )
            assert name != d.survivor, "self-reference in dropped should be filtered"
        all_survivors.add(d.survivor)
        all_dropped.update(d.dropped)

    # UC + IBD must collapse — exactly one of the two should be dropped.
    uc_ibd_dropped = all_dropped & {"ulcerative colitis", "inflammatory bowel disease"}
    assert len(uc_ibd_dropped) == 1, (
        f"expected exactly one of UC/IBD dropped, got {uc_ibd_dropped!r}; "
        f"full decisions={output.decisions!r}"
    )

    # T2DM + DM must collapse — exactly one of the two should be dropped.
    dm_dropped = all_dropped & {"type 2 diabetes mellitus", "diabetes mellitus"}
    assert len(dm_dropped) == 1, (
        f"expected exactly one of T2DM/DM dropped, got {dm_dropped!r}; "
        f"full decisions={output.decisions!r}"
    )

    # NASH + NAFLD should also collapse (NASH ⊂ NAFLD).
    nash_dropped = all_dropped & {
        "non-alcoholic steatohepatitis",
        "non-alcoholic fatty liver disease",
    }
    assert len(nash_dropped) == 1, (
        f"expected exactly one of NASH/NAFLD dropped, got {nash_dropped!r}; "
        f"full decisions={output.decisions!r}"
    )

    # T1DM is a sibling of T2DM, not a super/subtype — must NOT be dropped.
    assert "type 1 diabetes mellitus" not in all_dropped, (
        "T1DM is a sibling of T2DM, not a sub/super type — should not be "
        f"dropped. decisions={output.decisions!r}"
    )

    # PPARG-related familial partial lipodystrophy is a distinct entity from
    # the other lipid/glucose candidates and must not be folded into any
    # broader category by the LLM.
    assert (
        "pparg-related familial partial lipodystrophy" not in all_dropped
    ), f"unrelated rare lipodystrophy must not be dropped. decisions={output.decisions!r}"


async def test_no_overlaps_returns_no_decisions():
    """When the candidate list has no super/subtype pairs, the LLM should
    return an empty decisions list."""
    candidates: list[tuple[str, str, str | None]] = [
        ("psoriasis", "competitor", "EFO_0000676"),
        ("rheumatoid arthritis", "competitor", "EFO_0000685"),
        ("alzheimer disease", "mechanism", "EFO_0000249"),
    ]
    output = await run_hierarchical_dedup(
        drug_name="placebo",
        mechanism_targets=[],
        candidates=candidates,
    )
    assert output.decisions == [], (
        f"expected no overlaps in this candidate list, got {output.decisions!r}"
    )
