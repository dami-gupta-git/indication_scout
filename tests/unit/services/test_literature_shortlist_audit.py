from indication_scout.services.literature_shortlist_audit import (
    ShortlistPaper,
    has_explicit_controlled_design,
    publication_type_boost,
    rank_of,
    rank_shortlist,
    reserve_ranked_papers,
)


def _paper(
    pmid: str,
    similarity: float,
    *,
    abstract: str = "",
    pubtypes: tuple[str, ...] = ("Journal Article",),
) -> ShortlistPaper:
    return ShortlistPaper(
        pmid=pmid,
        title=f"Paper {pmid}",
        abstract=abstract,
        similarity=similarity,
        pubtypes=pubtypes,
        publication_date="2025",
        trial_reference=False,
    )


def test_publication_type_boost_matches_production_values():
    assert publication_type_boost(("Journal Article",)) == 1.0
    assert publication_type_boost(("Journal Article", "Randomized Controlled Trial")) == 2.0
    assert publication_type_boost(("Review",)) == 0.6
    assert publication_type_boost(()) == 1.0


def test_text_controlled_signal_recognizes_unindexed_randomized_trial():
    paper = _paper(
        "40487775",
        0.4,
        abstract="Participants were randomly assigned in a double-blind, placebo-controlled trial.",
    )

    assert has_explicit_controlled_design(paper) is True


def test_counterfactual_can_promote_unindexed_controlled_paper():
    papers = [
        _paper("1", 0.70),
        _paper(
            "40487775",
            0.40,
            abstract="This randomized, double-blind, placebo-controlled trial enrolled adults.",
        ),
        _paper("3", 0.60),
    ]

    baseline = rank_shortlist(papers, 2, use_text_controlled_signal=False)
    counterfactual = rank_shortlist(papers, 2, use_text_controlled_signal=True)

    assert [item.paper.pmid for item in baseline] == ["1", "3"]
    assert [item.paper.pmid for item in counterfactual] == ["40487775", "1"]
    assert counterfactual[0].boost == 2.0
    assert counterfactual[0].score == 0.8


def test_rank_of_returns_rank_or_none():
    ranked = rank_shortlist(
        [_paper("2", 0.5), _paper("1", 0.5)],
        2,
        use_text_controlled_signal=False,
    )

    assert [item.paper.pmid for item in ranked] == ["1", "2"]
    assert rank_of(ranked, "1") == 1
    assert rank_of(ranked, "2") == 2
    assert rank_of(ranked, "3") is None


def test_reserve_ranked_papers_promotes_only_eligible_papers():
    baseline = rank_shortlist(
        [_paper(str(index), 1.0 - index / 100) for index in range(1, 7)],
        6,
        use_text_controlled_signal=False,
    )

    selected = reserve_ranked_papers(baseline, {"5", "6"}, reserve=1, limit=3)

    assert [item.paper.pmid for item in selected] == ["5", "1", "2"]
    assert [(item.boost, item.text_controlled) for item in selected] == [
        (1.0, False),
        (1.0, False),
        (1.0, False),
    ]
