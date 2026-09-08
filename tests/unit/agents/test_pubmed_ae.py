"""Unit tests for disease-scoped adverse-event PubMed queries."""

from indication_scout.agents.literature.pubmed_ae import _disease_scoped_query


def test_disease_scoped_query_includes_deduplicated_aliases() -> None:
    query = _disease_scoped_query(
        "bupropion",
        "cocaine use disorder",
        ["cocaine dependence", "Cocaine Use Disorder"],
    )

    assert query.count('"cocaine use disorder"[mh]') == 1
    assert query.count('"cocaine use disorder"[tiab]') == 1
    assert query.count('"cocaine dependence"[mh]') == 1
    assert query.count('"cocaine dependence"[tiab]') == 1
    assert '"bupropion"[nm]' in query
    assert ' OR ("cocaine dependence"[mh]' in query
