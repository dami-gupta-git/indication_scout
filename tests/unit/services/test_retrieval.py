"""Unit tests for services/retrieval — no network, no LLM calls."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    RichDrugData,
    TargetData,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import (
    AbstractResult,
    RetrievalService,
    _key_finding_is_negative,
    _key_finding_is_positive,
)

# --- Fixtures ---


@pytest.fixture
def atc_metformin() -> ATCDescription:
    return ATCDescription(
        level1="A",
        level1_description="ALIMENTARY TRACT AND METABOLISM",
        level2="A10",
        level2_description="DRUGS USED IN DIABETES",
        level3="A10B",
        level3_description="BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        level4="A10BA",
        level4_description="Biguanides",
        level5="A10BA02",
        who_name="metformin",
    )


@pytest.fixture
def rich_metformin(atc_metformin) -> RichDrugData:
    drug = DrugData(
        chembl_id="CHEMBL1431",
        drug_type="Small molecule",
        maximum_clinical_stage="APPROVAL",
        atc_classifications=["A10BA02"],
        targets=[
            DrugTarget(
                target_id="ENSG00000132356",
                target_symbol="PRKAA1",
                mechanism_of_action="AMP-activated protein kinase activator",
                action_type="ACTIVATOR",
            ),
            DrugTarget(
                target_id="ENSG00000162409",
                target_symbol="PRKAA2",
                mechanism_of_action="AMP-activated protein kinase activator",  # duplicate MoA
                action_type="ACTIVATOR",
            ),
        ],
    )
    targets = [
        TargetData(
            target_id="ENSG00000132356",
            symbol="PRKAA1",
            name="Protein kinase AMP-activated alpha 1",
        ),
        TargetData(
            target_id="ENSG00000162409",
            symbol="PRKAA2",
            name="Protein kinase AMP-activated alpha 2",
        ),
    ]
    return RichDrugData(drug=drug, targets=targets)


@pytest.fixture
def metformin_profile() -> DrugProfile:
    return DrugProfile(
        chembl_id="CHEMBL1431",
        target_gene_symbols=["PRKAA1", "PRKAA2", "STK11"],
        mechanisms_of_action=[
            "AMP-activated protein kinase activator",
            "mTOR inhibitor",
        ],
        atc_codes=["A10BA02"],
        atc_descriptions=["BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS", "Biguanides"],
        drug_type="Small molecule",
    )


@pytest.fixture
def svc(tmp_path):
    return RetrievalService(tmp_path)


# --- DrugProfile.from_rich_drug_data ---


def test_drug_profile_from_rich_drug_data(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.chembl_id == "CHEMBL1431"
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]
    assert profile.atc_codes == ["A10BA02"]
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]
    assert profile.drug_type == "Small molecule"


def test_drug_profile_from_rich_drug_data_target_gene_symbols(
    rich_metformin, atc_metformin
):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]


def test_drug_profile_from_rich_drug_data_mechanisms_deduped(
    rich_metformin, atc_metformin
):
    """Duplicate MoA strings across targets are collapsed to one."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]


def test_drug_profile_from_rich_drug_data_atc_codes(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.atc_codes == ["A10BA02"]


def test_drug_profile_from_rich_drug_data_atc_descriptions(
    rich_metformin, atc_metformin
):
    """level3_description then level4_description, deduplicated."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]


def test_drug_profile_from_rich_drug_data_drug_type(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.drug_type == "Small molecule"


# --- extract_organ_term ---


async def test_extract_organ_term_returns_stripped_string(svc):
    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(return_value="  colon  "),
    ):
        result = await svc.extract_organ_term("colorectal cancer")
    assert result == "colon"


async def test_extract_organ_term_returns_cached_result(tmp_path):
    from indication_scout.config import get_settings
    from indication_scout.utils.cache import cache_set

    cache_set(
        "organ_term",
        {
            "disease_name": "colorectal cancer",
            "small_llm_model": get_settings().small_llm_model,
        },
        "colon",
        tmp_path,
    )

    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(),
    ) as mock_llm:
        result = await RetrievalService(tmp_path).extract_organ_term(
            "colorectal cancer"
        )

    assert result == "colon"
    mock_llm.assert_not_called()


# --- expand_search_terms ---


async def test_expand_search_terms_returns_list(tmp_path, metformin_profile):
    llm_response = '["metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert result == ["metformin AND colorectal cancer", "biguanides AND colon"]


async def test_expand_search_terms_prompt_contains_drug_name(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert "metformin" in captured["prompt"]
    assert "colorectal cancer" in captured["prompt"]


async def test_expand_search_terms_prompt_sources_names_from_chembl(
    tmp_path, metformin_profile
):
    """Both `{drug_name}` and `{synonyms}` come from get_all_drug_names(chembl_id):
    index 0 is the pref_name, the rest are synonyms.
    """
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["x"]'

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["CHEMBL_PREF", "CHEMBL_SYN_1", "CHEMBL_SYN_2"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    prompt = captured["prompt"]
    assert "Drug name: CHEMBL_PREF" in prompt
    assert "Synonyms and trade names: CHEMBL_SYN_1, CHEMBL_SYN_2" in prompt


async def test_expand_search_terms_prompt_contains_targets(tmp_path, metformin_profile):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert "PRKAA1" in captured["prompt"]
    assert "PRKAA2" in captured["prompt"]
    assert "STK11" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_atc_descriptions(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert "Biguanides" in captured["prompt"]
    assert "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_organ_term(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert "colon" in captured["prompt"]


async def test_expand_search_terms_deduplicates_output(tmp_path, metformin_profile):
    """Case-duplicate entries in LLM output are deduped; first occurrence casing is preserved."""
    llm_response = '["Metformin AND colorectal cancer", "metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage", "fortamet"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert result == ["Metformin AND colorectal cancer", "biguanides AND colon"]


async def test_expand_search_terms_returns_cached_result(tmp_path, metformin_profile):
    from indication_scout.config import get_settings
    from indication_scout.utils.cache import cache_set

    cached_queries = ["metformin AND colorectal cancer", "biguanides AND colon"]
    cache_set(
        "expand_search_terms",
        {
            "chembl_id": "CHEMBL1431",
            "disease_name": "colorectal cancer",
            "small_llm_model": get_settings().small_llm_model,
        },
        cached_queries,
        tmp_path,
    )

    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(),
    ) as mock_llm:
        result = await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "colorectal cancer", metformin_profile
        )

    assert result == cached_queries
    mock_llm.assert_not_called()


# --- build_drug_profile ---


async def test_build_drug_profile_returns_profile(svc, rich_metformin, atc_metformin):
    """build_drug_profile fetches RichDrugData and ATC descriptions, returns a DrugProfile."""
    mock_open_targets = AsyncMock()
    mock_open_targets.__aenter__ = AsyncMock(return_value=mock_open_targets)
    mock_open_targets.__aexit__ = AsyncMock(return_value=None)
    mock_open_targets.get_rich_drug_data = AsyncMock(return_value=rich_metformin)

    mock_chembl = AsyncMock()
    mock_chembl.__aenter__ = AsyncMock(return_value=mock_chembl)
    mock_chembl.__aexit__ = AsyncMock(return_value=None)
    mock_chembl.get_atc_description = AsyncMock(return_value=atc_metformin)

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_open_targets,
        ),
        patch(
            "indication_scout.services.retrieval.ChEMBLClient", return_value=mock_chembl
        ),
    ):
        profile = await svc.build_drug_profile("CHEMBL1431")

    assert profile.chembl_id == "CHEMBL1431"
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]
    assert profile.atc_codes == ["A10BA02"]
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]
    assert profile.drug_type == "Small molecule"


async def test_build_drug_profile_fetches_atc_per_code(
    svc, rich_metformin, atc_metformin
):
    """get_atc_description is called once per ATC code on the drug."""
    mock_open_targets = AsyncMock()
    mock_open_targets.__aenter__ = AsyncMock(return_value=mock_open_targets)
    mock_open_targets.__aexit__ = AsyncMock(return_value=None)
    mock_open_targets.get_rich_drug_data = AsyncMock(return_value=rich_metformin)

    mock_chembl = AsyncMock()
    mock_chembl.__aenter__ = AsyncMock(return_value=mock_chembl)
    mock_chembl.__aexit__ = AsyncMock(return_value=None)
    mock_chembl.get_atc_description = AsyncMock(return_value=atc_metformin)

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_open_targets,
        ),
        patch(
            "indication_scout.services.retrieval.ChEMBLClient", return_value=mock_chembl
        ),
    ):
        await svc.build_drug_profile("CHEMBL1431")

    # rich_metformin has one ATC code: "A10BA02"
    assert mock_chembl.get_atc_description.call_count == 1
    mock_chembl.get_atc_description.assert_called_once_with("A10BA02")


async def test_build_drug_profile_no_atc_codes(svc, rich_metformin):
    """If the drug has no ATC codes, ChEMBLClient is never opened and atc_descriptions is []."""
    rich_metformin.drug.atc_classifications = []

    mock_open_targets = AsyncMock()
    mock_open_targets.__aenter__ = AsyncMock(return_value=mock_open_targets)
    mock_open_targets.__aexit__ = AsyncMock(return_value=None)
    mock_open_targets.get_rich_drug_data = AsyncMock(return_value=rich_metformin)

    mock_chembl = AsyncMock()

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_open_targets,
        ),
        patch(
            "indication_scout.services.retrieval.ChEMBLClient", return_value=mock_chembl
        ),
    ):
        profile = await svc.build_drug_profile("CHEMBL1431")

    assert profile.atc_codes == []
    assert profile.atc_descriptions == []
    mock_chembl.__aenter__.assert_not_called()


# --- get_stored_pmids ---


def _make_db_session(returned_pmids: list[str]) -> MagicMock:
    """Return a mock Session whose execute().fetchall() yields the given PMIDs as row tuples."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [(pmid,) for pmid in returned_pmids]
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_result
    return mock_db


def test_get_stored_pmids_returns_present_pmids(svc):
    """Only PMIDs that the DB reports as present are returned.

    The DB mock returns ["111", "222"] as existing rows. The third PMID "333"
    is not in the mock result, so it must not appear in the output.
    """
    mock_db = _make_db_session(["111", "222"])

    result = svc.get_stored_pmids(["111", "222", "333"], mock_db)

    assert result == {"111", "222"}


def test_get_stored_pmids_empty_input_returns_empty_set(svc):
    """Empty input short-circuits before hitting the DB.

    No DB query should be made when there are no PMIDs to check — avoids
    sending a vacuous ANY(ARRAY[]) query to Postgres.
    """
    mock_db = _make_db_session([])

    result = svc.get_stored_pmids([], mock_db)

    assert result == set()
    mock_db.execute.assert_not_called()


def test_get_stored_pmids_all_present(svc):
    """All input PMIDs present in DB → full set returned."""
    mock_db = _make_db_session(["111", "222"])

    result = svc.get_stored_pmids(["111", "222"], mock_db)

    assert result == {"111", "222"}


def test_get_stored_pmids_none_present(svc):
    """No input PMIDs present in DB → empty set returned."""
    mock_db = _make_db_session([])

    result = svc.get_stored_pmids(["111", "222"], mock_db)

    assert result == set()


def test_get_stored_pmids_passes_pmids_to_query(svc):
    """The pmids list is passed as a bind parameter to the SQL query."""
    mock_db = _make_db_session([])
    pmids = ["111", "222", "333"]

    svc.get_stored_pmids(pmids, mock_db)

    call_kwargs = mock_db.execute.call_args
    # Second positional arg is the params dict
    params = call_kwargs[0][1]
    assert params["pmids"] == pmids


# --- fetch_new_abstracts ---


def _make_pubmed_abstract(pmid: str) -> MagicMock:
    """Return a mock PubmedAbstract with a known pmid."""
    m = MagicMock()
    m.pmid = pmid
    return m


@pytest.mark.parametrize(
    "all_pmids, stored_pmids, expected_fetched",
    [
        # Case 1: all PMIDs are new → fetch all
        (
            ["111", "222", "333"],
            set(),
            ["111", "222", "333"],
        ),
        # Case 2: all PMIDs already stored → no fetch
        (
            ["111", "222"],
            {"111", "222"},
            [],
        ),
        # Case 3: mixed → fetch only the new subset
        (
            ["111", "222", "333"],
            {"111"},
            ["222", "333"],
        ),
    ],
)
async def test_fetch_new_abstracts(svc, all_pmids, stored_pmids, expected_fetched):
    """fetch_new_abstracts calls fetch_abstracts with only the new PMIDs.

    Three cases: all new, all stored, mixed. When stored_pmids covers everything
    the network call is skipped entirely (fetch_abstracts not called).
    """
    mock_client = AsyncMock()
    mock_client.fetch_abstracts = AsyncMock(
        return_value=[_make_pubmed_abstract(p) for p in expected_fetched]
    )

    result = await svc.fetch_new_abstracts(all_pmids, stored_pmids, mock_client)

    if not expected_fetched:
        mock_client.fetch_abstracts.assert_not_called()
        assert result == []
    else:
        mock_client.fetch_abstracts.assert_called_once_with(expected_fetched)
        assert [r.pmid for r in result] == expected_fetched


# --- embed_abstracts ---


def _make_abstract(pmid: str, title: str, abstract: str | None) -> PubmedAbstract:
    return PubmedAbstract(pmid=pmid, title=title, abstract=abstract)


async def test_embed_abstracts_texts_contain_title_and_abstract(svc):
    """embed_async() is called with '<title>. <abstract>' for each abstract."""
    abstracts = [_make_abstract("1", "My Title", "My abstract text.")]
    mock_vectors = [[0.1] * 768]

    with patch(
        "indication_scout.services.retrieval.embed_async", return_value=mock_vectors
    ) as mock_embed:
        result = await svc.embed_abstracts(abstracts)

    mock_embed.assert_called_once_with(["My Title. My abstract text."])
    assert len(result) == 1
    assert result[0][0].pmid == "1"
    assert result[0][1] == mock_vectors[0]


async def test_embed_abstracts_none_abstract_produces_title_dot_space(svc):
    """An abstract of None produces '<title>. ' without crashing."""
    abstracts = [_make_abstract("2", "Only Title", None)]
    mock_vectors = [[0.2] * 768]

    with patch(
        "indication_scout.services.retrieval.embed_async", return_value=mock_vectors
    ) as mock_embed:
        result = await svc.embed_abstracts(abstracts)

    mock_embed.assert_called_once_with(["Only Title. "])
    assert result[0][0].pmid == "2"


async def test_embed_abstracts_empty_input_skips_embed(svc):
    """Empty input returns [] without calling embed_async()."""
    with patch("indication_scout.services.retrieval.embed_async") as mock_embed:
        result = await svc.embed_abstracts([])

    mock_embed.assert_not_called()
    assert result == []


async def test_embed_abstracts_vectors_align_to_abstracts_by_index(svc):
    """Each abstract is paired with the vector at the same index."""
    abstracts = [
        _make_abstract("10", "Title A", "Abstract A"),
        _make_abstract("20", "Title B", "Abstract B"),
        _make_abstract("30", "Title C", "Abstract C"),
    ]
    mock_vectors = [[float(i)] * 768 for i in range(3)]

    with patch(
        "indication_scout.services.retrieval.embed_async", return_value=mock_vectors
    ):
        result = await svc.embed_abstracts(abstracts)

    assert len(result) == 3
    for i, (abstract, vector) in enumerate(result):
        assert abstract.pmid == abstracts[i].pmid
        assert vector == mock_vectors[i]


# --- insert_abstracts ---


def _make_pair(pmid: str) -> tuple[PubmedAbstract, list[float]]:
    abstract = PubmedAbstract(
        pmid=pmid,
        title="Title",
        abstract="Abstract text",
        authors=["Author A"],
        journal="Journal X",
        pub_date="2024",
        mesh_terms=["MeSH term"],
    )
    vector = [0.1] * 768
    return abstract, vector


def test_insert_abstracts_calls_execute_and_commit(svc):
    """session.execute() and session.commit() are called when pairs is non-empty."""
    mock_db = MagicMock()
    pairs = [_make_pair("111"), _make_pair("222")]

    with patch("indication_scout.services.retrieval.insert") as mock_insert:
        mock_stmt = MagicMock()
        mock_insert.return_value.values.return_value.on_conflict_do_nothing.return_value = (
            mock_stmt
        )
        svc.insert_abstracts(pairs, mock_db)

    mock_db.execute.assert_called_once_with(mock_stmt)
    mock_db.commit.assert_called_once()


def test_insert_abstracts_empty_pairs_skips_db(svc):
    """Empty pairs list does not touch the DB."""
    mock_db = MagicMock()

    svc.insert_abstracts([], mock_db)

    mock_db.execute.assert_not_called()
    mock_db.commit.assert_not_called()


def test_insert_abstracts_rows_contain_all_fields(svc):
    """Each row passed to insert() contains all expected fields including embedding."""
    mock_db = MagicMock()
    abstract = PubmedAbstract(
        pmid="999",
        title="My Title",
        abstract="My abstract",
        authors=["Author A"],
        journal="Nature",
        pub_date="2023",
        mesh_terms=["Diabetes"],
    )
    vector = [0.5] * 768
    captured_rows = {}

    def capture_values(rows):
        captured_rows["rows"] = rows
        stmt = MagicMock()
        stmt.on_conflict_do_nothing.return_value = MagicMock()
        return stmt

    with patch("indication_scout.services.retrieval.insert") as mock_insert:
        mock_insert.return_value.values.side_effect = capture_values
        svc.insert_abstracts([(abstract, vector)], mock_db)

    row = captured_rows["rows"][0]
    assert row["pmid"] == "999"
    assert row["title"] == "My Title"
    assert row["abstract"] == "My abstract"
    assert row["authors"] == ["Author A"]
    assert row["journal"] == "Nature"
    assert row["pub_date"] == "2023"
    assert row["mesh_terms"] == ["Diabetes"]
    assert row["embedding"] == vector


# --- fetch_and_cache ---


async def test_fetch_and_cache_returns_deduped_pmids(svc):
    """PMIDs shared across queries appear exactly once in the result."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    # Two queries share PMID "111"
    mock_client.search = AsyncMock(side_effect=[["111", "222"], ["111", "333"]])
    mock_client.fetch_abstracts = AsyncMock(return_value=[])

    with (
        patch(
            "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
        ),
        patch("indication_scout.services.retrieval.embed_async", return_value=[]),
        patch("indication_scout.services.retrieval.insert"),
    ):
        result = await svc.fetch_and_cache(["query1", "query2"], mock_db)

    assert result == ["111", "222", "333"]
    assert len(result) == len(set(result))


async def test_fetch_and_cache_calls_search_per_query(svc):
    """search() is called once for each query string."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(return_value=[])
    mock_client.fetch_abstracts = AsyncMock(return_value=[])

    with (
        patch(
            "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
        ),
        patch("indication_scout.services.retrieval.insert"),
    ):
        await svc.fetch_and_cache(["q1", "q2", "q3"], mock_db)

    assert mock_client.search.call_count == 3
    from indication_scout.config import get_settings

    _expected_max = get_settings().pubmed_max_results
    mock_client.search.assert_any_call(
        "q1", max_results=_expected_max, date_before=None
    )
    mock_client.search.assert_any_call(
        "q2", max_results=_expected_max, date_before=None
    )
    mock_client.search.assert_any_call(
        "q3", max_results=_expected_max, date_before=None
    )


async def test_fetch_and_cache_empty_queries_returns_empty(svc):
    """Empty query list returns [] without hitting PubMed."""
    mock_db = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(return_value=[])

    with patch(
        "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
    ):
        result = await svc.fetch_and_cache([], mock_db)

    assert result == []
    mock_client.search.assert_not_called()


# --- semantic_search ---


def _make_db_with_rows(rows: list[tuple]) -> MagicMock:
    """Return a mock Session whose execute().fetchall() yields the given rows."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_result
    return mock_db


@pytest.fixture
def mock_pubtypes_empty():
    """Patch PubMedClient so semantic_search's pubtype fetch is a no-op (empty map).
    With no pubtypes, every record gets PUBTYPE_BOOST_DEFAULT (1.0), so the
    rerank is a no-op and ordering follows pure similarity — matching the
    pre-rerank behaviour these unit tests were written against.
    """
    mock_client = AsyncMock()
    mock_client.fetch_pubtypes = AsyncMock(return_value={})
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch(
        "indication_scout.services.retrieval.PubMedClient",
        return_value=mock_client,
    ):
        yield mock_client


async def test_semantic_search_returns_ranked_dicts(svc, mock_pubtypes_empty):
    """Returns list of dicts with pmid, title, abstract, similarity for each DB row."""
    db_rows = [
        ("111", "Title A", "Abstract A", 0.92),
        ("222", "Title B", "Abstract B", 0.85),
    ]
    mock_db = _make_db_with_rows(db_rows)
    mock_vector = [0.1] * 768

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
    ):
        result = await svc.semantic_search(
            "colorectal cancer", "CHEMBL1431", ["111", "222"], mock_db
        )

    assert len(result) == 2
    assert result[0].pmid == "111"
    assert result[0].title == "Title A"
    assert result[0].abstract == "Abstract A"
    assert result[0].similarity == 0.92
    assert result[1].pmid == "222"
    assert result[1].title == "Title B"
    assert result[1].abstract == "Abstract B"
    assert result[1].similarity == 0.85


async def test_semantic_search_embeds_therapeutic_query(svc, mock_pubtypes_empty):
    """embed() is called with the therapeutic intent query string."""
    mock_db = _make_db_with_rows([])
    mock_vector = [0.1] * 768
    captured = {}

    def capture_embed(texts: list[str]) -> list[list[float]]:
        captured["texts"] = texts
        return [mock_vector]

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion", "wellbutrin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async", side_effect=capture_embed
        ),
    ):
        await svc.semantic_search("obesity", "CHEMBL894", ["111"], mock_db)

    assert len(captured["texts"]) == 1
    assert "bupropion" in captured["texts"][0]
    assert "obesity" in captured["texts"][0]


async def test_semantic_search_uses_pref_name_not_chembl_id(svc, mock_pubtypes_empty):
    """The embedded query string uses pref_name (first element of get_all_drug_names),
    not the ChEMBL ID or any other identifier. Sentinel values guarantee correct routing.
    """
    mock_db = _make_db_with_rows([])
    mock_vector = [0.1] * 768
    captured = {}

    def capture_embed(texts: list[str]) -> list[list[float]]:
        captured["texts"] = texts
        return [mock_vector]

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["SENTINEL_PREF", "SENTINEL_SYN_A"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async", side_effect=capture_embed
        ),
    ):
        await svc.semantic_search("obesity", "CHEMBL999", ["111"], mock_db)

    query = captured["texts"][0]
    assert "SENTINEL_PREF" in query
    # ChEMBL ID must not leak into the embedded query
    assert "CHEMBL999" not in query
    # Non-pref synonyms must not be used as the drug identifier
    assert "SENTINEL_SYN_A" not in query


async def test_semantic_search_passes_pmids_to_query(svc, mock_pubtypes_empty):
    """The pmids list is passed as a bind parameter to the SQL query."""
    mock_db = _make_db_with_rows([])
    mock_vector = [0.1] * 768
    pmids = ["111", "222", "333"]

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
    ):
        await svc.semantic_search("diabetes", "CHEMBL1431", pmids, mock_db)

    call_kwargs = mock_db.execute.call_args
    params = call_kwargs[0][1]
    assert params["pmids"] == pmids


async def test_semantic_search_respects_top_k_from_settings(svc, mock_pubtypes_empty):
    """Returned list length is capped at settings.semantic_search_top_k after rerank."""
    from indication_scout.config import get_settings

    top_k = get_settings().semantic_search_top_k
    # Build more rows than top_k so the slice has work to do.
    db_rows = [
        (f"{i}", f"Title {i}", f"Abstract {i}", 0.9 - 0.01 * i)
        for i in range(top_k + 3)
    ]
    mock_db = _make_db_with_rows(db_rows)
    mock_vector = [0.1] * 768

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
    ):
        result = await svc.semantic_search("diabetes", "CHEMBL1431", ["111"], mock_db)

    assert len(result) == top_k


async def test_semantic_search_similarity_is_float(svc, mock_pubtypes_empty):
    """similarity values in returned dicts are plain Python floats."""
    from decimal import Decimal

    db_rows = [("111", "Title", "Abstract", Decimal("0.8765"))]
    mock_db = _make_db_with_rows(db_rows)
    mock_vector = [0.1] * 768

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
    ):
        result = await svc.semantic_search("diabetes", "CHEMBL1431", ["111"], mock_db)

    assert isinstance(result[0].similarity, float)
    assert result[0].similarity == float(Decimal("0.8765"))


@pytest.mark.skip(reason="wandb logging is currently commented out in retrieval.py")
async def test_semantic_search_logs_wandb_table_when_run_active(svc):
    """When wandb.run is active, wandb.log is called with a Table and scalar metrics."""
    db_rows = [
        ("111", "Title A", "Abstract A", 0.92),
        ("222", "Title B", "Abstract B", 0.85),
    ]
    mock_db = _make_db_with_rows(db_rows)
    mock_vector = [0.1] * 768
    logged = {}

    mock_table = MagicMock()
    mock_run = MagicMock()

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
        patch("indication_scout.services.retrieval.wandb.run", mock_run),
        patch(
            "indication_scout.services.retrieval.wandb.Table", return_value=mock_table
        ),
        patch(
            "indication_scout.services.retrieval.wandb.log",
            side_effect=lambda d: logged.update(d),
        ),
    ):
        await svc.semantic_search(
            "colorectal cancer", "CHEMBL1431", ["111", "222"], mock_db
        )

    assert logged["semantic_search/colorectal_cancer/candidate_pmids"] == 2
    assert logged["semantic_search/colorectal_cancer/results_returned"] == 2
    assert logged["semantic_search/colorectal_cancer/top_similarity"] == 0.92
    assert "metformin" in logged["semantic_search/colorectal_cancer/query"]
    assert "colorectal cancer" in logged["semantic_search/colorectal_cancer/query"]
    assert logged["semantic_search/colorectal_cancer/results_table"] is mock_table
    assert mock_table.add_data.call_count == 2
    mock_table.add_data.assert_any_call("111", "Title A", 0.92)
    mock_table.add_data.assert_any_call("222", "Title B", 0.85)


async def test_semantic_search_skips_wandb_log_when_no_run(svc, mock_pubtypes_empty):
    """When wandb.run is None, wandb.log is never called."""
    mock_db = _make_db_with_rows([("111", "Title A", "Abstract A", 0.92)])
    mock_vector = [0.1] * 768

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[mock_vector],
        ),
        patch("indication_scout.services.retrieval.wandb.run", None),
        patch("indication_scout.services.retrieval.wandb.log") as mock_log,
    ):
        await svc.semantic_search("colorectal cancer", "CHEMBL1431", ["111"], mock_db)

    mock_log.assert_not_called()


# --- synthesize ---

_SAMPLE_ABSTRACTS = [
    AbstractResult(
        pmid="11111111",
        title="Metformin reduces colorectal cancer risk",
        abstract="This RCT showed significant reduction in CRC incidence.",
        similarity=0.95,
    ),
    AbstractResult(
        pmid="22222222",
        title="AMPK activation and colon cancer",
        abstract="Preclinical data demonstrating AMPK-mediated apoptosis.",
        similarity=0.88,
    ),
]

# Directional per-PMID verdicts (the new schema): supporting/contradicting/mixed/contaminated.
# supporting/contradicting/relevant lists + study_count + direction are DERIVED in code from this.
_SAMPLE_LLM_RESPONSE = json.dumps(
    {
        "verdicts": {"11111111": "supporting", "22222222": "supporting"},
        "evidence_basis": "drug_specific",
        "summary": "Two studies support metformin for colorectal cancer.",
        "strength": "moderate",
        "is_observational": False,
        "key_findings": [
            "CRC risk reduction in RCT (PMID: 11111111)",
            "AMPK-mediated apoptosis in colon cancer cells (PMID: 22222222)",
        ],
    }
)


async def test_synthesize_calls_llm_with_correct_prompt(svc):
    """query_llm is called with a prompt containing drug, disease, and abstract content."""
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
    ):
        await svc.synthesize("CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS)

    assert "metformin" in captured["prompt"]
    assert "colorectal cancer" in captured["prompt"]
    assert "PMID: 11111111" in captured["prompt"]
    assert "Metformin reduces colorectal cancer risk" in captured["prompt"]
    assert (
        "This RCT showed significant reduction in CRC incidence." in captured["prompt"]
    )


async def test_synthesize_prompt_uses_pref_name(svc):
    """`{drug_name}` in the synthesize prompt must come from get_all_drug_names[0]
    (pref_name), not from the ChEMBL ID. Uses a distinct sentinel to guarantee routing.
    """
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["SENTINEL_PREF", "SENTINEL_SYN"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
    ):
        await svc.synthesize("CHEMBL999", "colorectal cancer", _SAMPLE_ABSTRACTS)

    prompt = captured["prompt"]
    assert "SENTINEL_PREF" in prompt
    # ChEMBL ID must not leak into the prompt
    assert "CHEMBL999" not in prompt
    # Non-pref synonyms must not appear (synthesize only uses pref_name)
    assert "SENTINEL_SYN" not in prompt


async def test_synthesize_strips_markdown_fences(svc):
    """synthesize handles LLM responses wrapped in ```json ... ``` code fences."""
    fenced = f"```json\n{_SAMPLE_LLM_RESPONSE}\n```"
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=fenced),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )

    assert result.strength == "moderate"
    assert result.supporting_pmids == ["11111111", "22222222"]


async def test_synthesize_parses_llm_response(svc):
    """synthesize returns an EvidenceSummary with all fields matching the LLM JSON output,
    including the per-abstract relevance split derived from `verdicts`."""
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=_SAMPLE_LLM_RESPONSE),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )

    assert isinstance(result, EvidenceSummary)
    assert result.summary == "Two studies support metformin for colorectal cancer."
    assert result.study_count == 2
    assert result.strength == "moderate"
    assert result.direction == "supports"
    assert result.evidence_basis == "drug_specific"
    assert result.is_observational is False
    assert result.key_findings == [
        "CRC risk reduction in RCT (PMID: 11111111)",
        "AMPK-mediated apoptosis in colon cancer cells (PMID: 22222222)",
    ]
    assert result.supporting_pmids == ["11111111", "22222222"]
    assert result.relevant_pmids == ["11111111", "22222222"]
    assert result.contaminated_pmids == []


async def test_synthesize_degrades_to_safe_floor_on_invalid_json(svc):
    """Unparseable LLM JSON degrades to a safe untested floor (no raise): basis none, strength
    none, all abstracts contaminated. The tolerant parse + floor keeps the pipeline alive."""
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value="not valid json at all"),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )
    assert result.evidence_basis == "none"
    assert result.strength == "none"
    assert result.direction == "none"
    assert result.relevant_pmids == []
    assert result.study_count == 0
    assert set(result.contaminated_pmids) == {"11111111", "22222222"}


async def test_synthesize_strength_cap_forces_none_for_class_level(svc):
    """The deterministic strength cap: when evidence_basis != drug_specific, strength/direction
    are forced to none even if the LLM emitted a non-none grade — the Parkinson class-level fix.
    The contaminated PMIDs (verdicts) are split out of the relevant set."""
    leaky_response = json.dumps(
        {
            "verdicts": {"11111111": "contaminated", "22222222": "contaminated"},
            "evidence_basis": "class_level",
            "summary": "Class-level GLP-1 RCTs; no direct drug evidence (PMID: 11111111).",
            "study_count": 2,
            "strength": "strong",
            "direction": "supports",
            "is_observational": False,
            "supporting_pmids": ["11111111", "22222222"],
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=leaky_response),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )

    # strength cap forces these to none for a non-drug_specific basis
    assert result.strength == "none"
    assert result.direction == "none"
    assert result.evidence_basis == "class_level"
    # prose untouched
    assert result.summary.startswith("Class-level GLP-1 RCTs")
    # per-abstract split from verdicts
    assert result.relevant_pmids == []
    assert result.contaminated_pmids == ["11111111", "22222222"]


async def test_synthesize_holdout_skips_strength_cap(svc):
    """In holdout_mode the relaxed rubric intentionally scores class-level evidence, so the
    deterministic strength cap must NOT run — a class_level basis keeps its non-none strength.
    """
    holdout_response = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "contaminated"},
            "evidence_basis": "class_level",
            "summary": "Class-level evidence (PMID: 11111111).",
            "strength": "moderate",
            "is_observational": None,
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=holdout_response),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS, holdout_mode=True
        )

    # cap skipped: class_level keeps its moderate strength in holdout
    assert result.strength == "moderate"
    assert result.direction == "supports"
    assert result.evidence_basis == "class_level"
    assert result.relevant_pmids == ["11111111"]
    assert result.contaminated_pmids == ["22222222"]


# --- _key_finding_is_positive guard (BRAVE-I/II regression) ---


def test_key_finding_positive_brave_i_endpoint_met():
    """BRAVE-I met its primary endpoint (missed secondaries) → positive → upgrades the per-PMID
    verdict contradicting→mixed so it appears in supporting too."""
    kf = [
        "baricitinib 4 mg achieved SRI-4 in 57% vs 46% (OR 1.57, p=0.016), meeting the "
        "primary endpoint, but no major secondary endpoints were met (PMID: 36848918)"
    ]
    assert _key_finding_is_positive("36848918", kf) is True


def test_key_finding_positive_brave_ii_failure_not_positive():
    """BRAVE-II 'achieved 47% vs 46% ... failing to meet the primary endpoint' is a clean
    NEGATIVE — the guard must NOT read it as positive (the 'failing to' negation gap)."""
    kf = [
        "baricitinib 4 mg achieved SRI-4 in 47% vs 46% (OR 1.07, 95% CI 0.75-1.53), "
        "failing to meet the primary endpoint (PMID: 36848919)"
    ]
    assert _key_finding_is_positive("36848919", kf) is False


def test_key_finding_positive_significant_benefit():
    assert _key_finding_is_positive(
        "1", ["significantly reduced disease activity vs placebo (PMID: 1)"]
    ) is True


def test_key_finding_negation_variants_not_positive():
    for text in (
        "achieved 30% response but did not meet the primary endpoint (PMID: 2)",
        "no significant difference vs placebo (PMID: 2)",
        "nonsignificant improvement (PMID: 2)",
        "did not meet the primary endpoint (PMID: 2)",
    ):
        assert _key_finding_is_positive("2", [text]) is False, text


def test_key_finding_only_matches_own_pmid():
    assert _key_finding_is_positive(
        "999", ["met its primary endpoint (PMID: 111)"]
    ) is False


# --- _key_finding_is_negative guard (imatinib × GBM regression) ---


def test_key_finding_negative_gbm_primary_endpoint_not_met():
    """The GBM bug: a flat-negative Phase 3 'primary endpoint ... was not met' was over-labeled
    mixed and leaked into supporting. The negative guard must flag it (and the positive guard must
    NOT — [^.]* must not span the negation between 'endpoint' and 'met')."""
    kf = [
        "In a randomized phase III trial of recurrent GBM, the primary endpoint of "
        "progression-free survival was not met; median PFS was 6 weeks in both arms "
        "(PMID: 19688297)"
    ]
    assert _key_finding_is_negative("19688297", kf) is True
    assert _key_finding_is_positive("19688297", kf) is False


def test_key_finding_negative_no_meaningful_activity():
    for text in (
        "the combination did not show clinically meaningful anti-tumour activity (PMID: 1)",
        "imatinib showed no measurable activity (PMID: 1)",
        "no significant difference vs placebo (PMID: 1)",
        "limited antitumor activity was observed (PMID: 1)",
    ):
        assert _key_finding_is_negative("1", [text]) is True, text


def test_key_finding_negative_leaves_genuine_positive_alone():
    """A clear positive must NOT be flagged negative (no spurious downgrade)."""
    assert _key_finding_is_negative(
        "1", ["significantly reduced disease activity vs placebo (PMID: 1)"]
    ) is False
    assert _key_finding_is_negative(
        "1", ["met its primary endpoint (PMID: 1)"]
    ) is False


def test_key_finding_negative_within_trial_mixed_not_downgraded():
    """A finding with BOTH a positive cue and a failure cue (genuinely within-trial mixed, e.g.
    BRAVE-I: met primary, missed secondaries) must NOT be flagged for downgrade."""
    kf = [
        "baricitinib met the primary endpoint (57% vs 46%, p=0.016) but no major secondary "
        "endpoints were met (PMID: 36848918)"
    ]
    assert _key_finding_is_negative("36848918", kf) is False


async def test_synthesize_missing_verdicts_treats_all_contaminated(svc):
    """No usable `verdicts` in the response → all abstracts contaminated (conservative), and the
    strength cap then forces a non-drug_specific basis to none."""
    no_verdicts = json.dumps(
        {
            "evidence_basis": "drug_specific",
            "summary": "Some evidence (PMID: 11111111).",
            "study_count": 2,
            "strength": "strong",
            "direction": "supports",
            "supporting_pmids": ["11111111"],
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=no_verdicts),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )

    assert result.relevant_pmids == []
    assert result.contaminated_pmids == ["11111111", "22222222"]


# --- get_drug_competitors ---


def _make_open_targets_mock(raw: dict) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get_drug_competitors = AsyncMock(return_value=raw)
    return mock_client


async def _passthrough_normalize_batch(terms: list[str]) -> dict[str, str]:
    """Stub for llm_normalize_disease_batch that returns each term unchanged."""
    return {term: term for term in terms}


async def test_get_drug_competitors_alias_in_removed_not_merged(tmp_path):
    """When an alias appears in both merge values and remove, its data must not be merged in."""
    raw = {
        "diseases": {
            "narcolepsy": {"competitor_a"},
            "narcolepsy-cataplexy syndrome": {"competitor_b"},
        },
        "drug_indications": [],
    }
    merge_result = {
        "merge": {"narcolepsy": ["narcolepsy-cataplexy syndrome"]},
        "remove": ["narcolepsy-cataplexy syndrome"],
    }
    mock_client = _make_open_targets_mock(raw)

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_client,
        ),
        patch(
            "indication_scout.services.retrieval.llm_normalize_disease_batch",
            new=_passthrough_normalize_batch,
        ),
        patch(
            "indication_scout.services.retrieval.merge_duplicate_diseases",
            new=AsyncMock(return_value=merge_result),
        ),
    ):
        result = await RetrievalService(tmp_path).get_drug_competitors("CHEMBL1")

    assert "narcolepsy-cataplexy syndrome" not in result
    assert "narcolepsy" in result
    assert result["narcolepsy"] == {"competitor_a"}


async def test_get_drug_competitors_returns_cached(tmp_path):
    """When a cache entry exists, the client and LLM are not called."""
    from indication_scout.config import get_settings
    from indication_scout.utils.cache import cache_set

    cached = {"depression": ["competitor_a"]}
    cache_set(
        "competitors_merged",
        {
            "chembl_id": "CHEMBL1",
            "date_before": None,
            "top_k": get_settings().literature_top_k,
        },
        cached,
        tmp_path,
    )

    mock_client = AsyncMock()
    with patch(
        "indication_scout.services.retrieval.OpenTargetsClient",
        return_value=mock_client,
    ):
        result = await RetrievalService(tmp_path).get_drug_competitors("CHEMBL1")

    assert result == {"depression": {"competitor_a"}}
    mock_client.__aenter__.assert_not_called()
