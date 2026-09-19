"""PubMed client contract: search returns the recorded PMIDs, and XML parses into an abstract."""

from __future__ import annotations

import pytest

from indication_scout.data_sources.pubmed import PubMedClient
from tests.regression.layer0_contracts.conftest import ContractClient

pytestmark = pytest.mark.contract


async def test_search_returns_recorded_pmids(contract_client: ContractClient) -> None:
    async with contract_client(PubMedClient, "pubmed_search") as client:
        pmids = await client.search("semaglutide alzheimer", max_results=10)

    assert pmids == [
        "39780249",
        "39445596",
        "39405916",
        "40156843",
        "37730113",
        "39976940",
        "40552638",
        "38639975",
        "41865758",
        "36989942",
    ]


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
