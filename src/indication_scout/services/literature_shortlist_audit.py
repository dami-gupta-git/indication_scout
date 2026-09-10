"""Deterministic selectors used by the literature-shortlist validation harness."""

from dataclasses import dataclass

from indication_scout.services.retrieval import (
    _CONTROLLED_DESIGN_PATTERN,
    _CONTROLLED_PUBTYPES,
    PUBTYPE_BOOST_DEFAULT,
    PUBTYPE_BOOSTS,
)


@dataclass(frozen=True)
class ShortlistPaper:
    """One paper in the semantic rerank candidate set."""

    pmid: str
    title: str
    abstract: str
    similarity: float
    pubtypes: tuple[str, ...]
    publication_date: str | None
    trial_reference: bool


@dataclass(frozen=True)
class RankedPaper:
    """A paper plus the score and evidence used by one selector."""

    paper: ShortlistPaper
    boost: float
    score: float
    text_controlled: bool


def has_explicit_controlled_design(paper: ShortlistPaper) -> bool:
    """Whether PubMed metadata or article text explicitly identifies a controlled design."""
    return bool(_CONTROLLED_PUBTYPES.intersection(paper.pubtypes)) or bool(
        _CONTROLLED_DESIGN_PATTERN.search(f"{paper.title} {paper.abstract}")
    )


def publication_type_boost(pubtypes: tuple[str, ...]) -> float:
    """Return the production publication-type multiplier for one paper."""
    return max(
        (PUBTYPE_BOOSTS.get(pubtype, PUBTYPE_BOOST_DEFAULT) for pubtype in pubtypes),
        default=PUBTYPE_BOOST_DEFAULT,
    )


def rank_shortlist(
    papers: list[ShortlistPaper],
    limit: int,
    *,
    use_text_controlled_signal: bool,
) -> list[RankedPaper]:
    """Rank a frozen candidate set using production scoring or the text-design counterfactual."""
    ranked: list[RankedPaper] = []
    controlled_boost = PUBTYPE_BOOSTS["Randomized Controlled Trial"]
    for paper in papers:
        metadata_boost = publication_type_boost(paper.pubtypes)
        text_controlled = has_explicit_controlled_design(paper)
        boost = (
            max(metadata_boost, controlled_boost)
            if use_text_controlled_signal and text_controlled
            else metadata_boost
        )
        ranked.append(
            RankedPaper(
                paper=paper,
                boost=boost,
                score=paper.similarity * boost,
                text_controlled=text_controlled,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (-item.score, item.paper.pmid),
    )[:limit]


def rank_of(ranked: list[RankedPaper], pmid: str) -> int | None:
    """Return the one-based rank of a PMID, or None when it is absent."""
    return next(
        (index for index, item in enumerate(ranked, start=1) if item.paper.pmid == pmid),
        None,
    )


def reserve_ranked_papers(
    baseline: list[RankedPaper],
    eligible_pmids: set[str],
    reserve: int,
    limit: int,
) -> list[RankedPaper]:
    """Reserve positions for eligible papers while preserving baseline order."""
    if reserve < 0:
        raise ValueError("reserve must be non-negative")
    reserved = [item for item in baseline if item.paper.pmid in eligible_pmids][:reserve]
    reserved_pmids = {item.paper.pmid for item in reserved}
    remainder = [item for item in baseline if item.paper.pmid not in reserved_pmids]
    return (reserved + remainder)[:limit]
