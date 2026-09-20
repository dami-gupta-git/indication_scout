"""PubMed client contract: search returns the recorded PMIDs, and XML parses into an abstract."""

from __future__ import annotations

import pytest

from indication_scout.data_sources.pubmed import PubMedClient
from tests.regression.layer0_contracts.conftest import ContractClient

pytestmark = pytest.mark.contract


async def test_search_returns_recorded_pmids(contract_client: ContractClient) -> None:
    async with contract_client(PubMedClient, "pubmed_search") as client:
        pmids = await client.search("semaglutide alzheimer", max_results=10)

    # PubMed's "relevance" sort re-ranks as new articles are indexed, so a re-recorded
    # cassette can return the same PMIDs in a different order. Assert membership, not order.
    assert len(pmids) == 10
    assert len(set(pmids)) == 10


async def test_fetch_abstracts_parses_every_field(
    contract_client: ContractClient,
) -> None:
    async with contract_client(PubMedClient, "pubmed_abstracts") as client:
        articles = await client.fetch_abstracts(["39215927"])

    assert len(articles) == 1
    article = articles[0]
    assert article.pmid == "39215927"
    assert article.title.startswith("Metformin protects against small intestine damage")
    assert article.abstract is not None
    assert article.abstract.startswith("The oral biguanide metformin is used to treat")
    assert article.authors == [
        "Dagsuyu, Eda",
        "Koroglu, Pinar",
        "Bulan, Omur Karabulut",
        "Gul, Ilknur Bugan",
        "Yanardag, Refiye",
    ]
    assert article.journal == "Journal of molecular histology"
    assert article.pub_date == "2024-Dec"
    assert article.mesh_terms[:3] == ["Metformin", "Animals", "Male"]
    assert len(article.mesh_terms) == 13
    assert article.keywords == [
        "Cancer",
        "Diabetes",
        "Metformin",
        "Rat",
        "Small intestine",
    ]
