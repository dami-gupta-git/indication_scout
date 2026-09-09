"""Unit tests for services/retrieval — no network, no LLM calls."""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    RichDrugData,
    TargetData,
)
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.services.retrieval import (
    AbstractResult,
    RetrievalService,
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
    async def all_studied(chembl_id, drug_names, abstracts, cache_dir):
        return {abstract.pmid: "studied" for abstract in abstracts}

    async def all_on_topic(chembl_id, drug, disease, abstracts, cache_dir):
        return {abstract.pmid: True for abstract in abstracts}

    with (
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=all_studied,
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease",
            new=all_on_topic,
        ),
    ):
        yield RetrievalService(tmp_path)


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
    llm_response = '["metformin AND <DISEASE>", "biguanides AND colon"]'
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

    assert result == [
        'metformin AND "colorectal cancer"',
        "biguanides AND colon",
    ]


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
    llm_response = (
        '["metformin AND <DISEASE>", "METFORMIN AND <DISEASE>", "biguanides AND colon"]'
    )
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

    assert result == [
        'metformin AND "colorectal cancer"',
        "biguanides AND colon",
    ]


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
            "logic_version": "deterministic_direct_v1",
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


async def test_filter_pmids_by_date_releases_db_before_esummary(svc):
    """The publication-date read transaction ends before the PubMed fallback."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = [("111", "2020-01-01")]
    mock_client = AsyncMock()

    async def filter_after_release(pmids, date_before):
        mock_db.rollback.assert_called_once_with()
        return ["222"]

    mock_client._filter_pmids_by_date = AsyncMock(side_effect=filter_after_release)

    result = await svc._filter_pmids_by_date(
        ["111", "222"], date(2021, 1, 1), mock_db, mock_client
    )

    assert result == ["111", "222"]
    mock_client._filter_pmids_by_date.assert_awaited_once_with(
        ["222"], date(2021, 1, 1)
    )


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
    """The direct query is complete while supplemental searches stay capped."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(side_effect=[["222", "333"], ["444"]])
    mock_client.search_complete = AsyncMock(return_value=["111", "222"])
    mock_client.fetch_abstracts = AsyncMock(return_value=[])

    with (
        patch(
            "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
        ),
        patch("indication_scout.services.retrieval.insert"),
    ):
        result = await svc.fetch_and_cache(
            ["q1", "q2", "q3"], mock_db, direct_query="q1"
        )

    assert result == ["111", "222", "333", "444"]
    assert mock_client.search.call_count == 2
    from indication_scout.config import get_settings

    _expected_max = get_settings().pubmed_max_results
    mock_client.search_complete.assert_awaited_once_with(
        "q1", page_size=_expected_max, date_before=None
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


async def test_fetch_and_cache_releases_db_before_fetching_abstracts(svc):
    """The stored-PMID read transaction ends before the PubMed fetch starts."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(return_value=["111"])

    async def fetch_after_release(pmids):
        mock_db.rollback.assert_called_once_with()
        return []

    mock_client.fetch_abstracts = AsyncMock(side_effect=fetch_after_release)

    with patch(
        "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
    ):
        result = await svc.fetch_and_cache(["query"], mock_db)

    assert result == ["111"]
    mock_client.fetch_abstracts.assert_awaited_once_with(["111"])


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
    assert top_k == 15
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


async def test_semantic_search_releases_db_before_fetching_pubtypes(svc):
    """The pgvector read transaction ends before the PubMed request starts."""
    db_rows = [("111", "Title", "Abstract", 0.9)]
    mock_db = _make_db_with_rows(db_rows)
    mock_vector = [0.1] * 768

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def fetch_after_release(pmids):
        mock_db.rollback.assert_called_once_with()
        return {}

    mock_client.fetch_pubtypes = AsyncMock(side_effect=fetch_after_release)

    with (
        patch(
            "indication_scout.services.retrieval.PubMedClient",
            return_value=mock_client,
        ),
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

    assert result == [
        AbstractResult(
            pmid="111",
            title="Title",
            abstract="Abstract",
            similarity=0.9,
            pubtype=[],
        )
    ]
    mock_client.fetch_pubtypes.assert_awaited_once_with(["111"])


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
# Supporting/contradicting/relevant lists and study_count are derived in code from this.
# Overall direction is preserved from the synthesis model when paper-level verdicts disagree.
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
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(
                return_value={"11111111": "supporting", "22222222": "supporting"}
            ),
        ),
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
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(
                return_value={"11111111": "supporting", "22222222": "supporting"}
            ),
        ),
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
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value="{}"),
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
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value="{}"),
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


async def test_synthesize_downgrades_unsupported_controlled_design_claim(svc):
    """An uncontrolled Parkinson study cannot render as RCT-backed in the candidate card."""
    abstract = AbstractResult(
        pmid="6431314",
        title="Bupropion in Parkinson's disease",
        abstract=(
            "We evaluated bupropion in 20 patients with idiopathic Parkinson's disease. "
            "Parkinsonism lessened in half the patients, although side effects were frequent."
        ),
        similarity=0.9,
        pubtype=["Journal Article"],
    )
    response = json.dumps(
        {
            "verdicts": {"6431314": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "A single uncontrolled study reported mild efficacy (PMID: 6431314).",
            "strength": "moderate",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": [
                "Parkinsonism lessened in half the patients (PMID: 6431314)."
            ],
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=response),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(return_value={"6431314": "supporting"}),
        ),
    ):
        result = await svc.synthesize("CHEMBL894", "Parkinson disease", [abstract])

    assert result.model_dump() == {
        "summary": "A single uncontrolled study reported mild efficacy (PMID: 6431314).",
        "study_count": 1,
        "strength": "moderate",
        "direction": "supports",
        "evidence_basis": "drug_specific",
        "is_observational": None,
        "is_animal_only": False,
        "key_findings": ["Parkinsonism lessened in half the patients (PMID: 6431314)."],
        "supporting_pmids": ["6431314"],
        "contradicting_pmids": [],
        "relevant_pmids": ["6431314"],
        "contaminated_pmids": [],
        "neutral_pmids": [],
        "safety_summary": "",
        "regulatory_safety_summary": "",
        "regulatory_safety_full_labels": "",
        "pharmacovigilance_summary": "",
        "literature_safety_summary": "",
        "label_safety_available": None,
        "safety_pmids": [],
        "safety_severity": None,
        "indication_harm": None,
        "indication_harm_summary": "",
        "indication_harm_pmids": [],
    }


async def test_synthesize_degrades_to_safe_floor_on_invalid_json(svc):
    """Unparseable LLM JSON degrades to a safe untested floor (no raise): basis none, strength
    none, all abstracts contaminated. The tolerant parse + floor keeps the pipeline alive.
    """
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


# --- _judge_pmid_drug_identity sub-call (the per-PMID identity authority) ---


async def test_pmid_drug_identity_isolated_and_cached_per_pmid(tmp_path):
    """A same-class drug is rejected without the model, while adding a batch neighbour reuses the
    existing exact-drug decision. A fresh cache cannot change the deterministic rejection.
    """
    from indication_scout.services.retrieval import _judge_pmid_drug_identity

    sildenafil = AbstractResult(
        pmid="1",
        title="Sildenafil trial",
        abstract="Sildenafil was evaluated against placebo.",
        similarity=0.9,
    )
    pf_compound = AbstractResult(
        pmid="27113485",
        title="PF-00489791 trial",
        abstract="The PDE5 inhibitor PF-00489791 was evaluated against placebo.",
        similarity=0.9,
    )
    neighbour = AbstractResult(
        pmid="3",
        title="Another sildenafil trial",
        abstract="Sildenafil was evaluated against placebo.",
        similarity=0.9,
    )
    calls: list[str] = []

    async def identity_llm(prompt: str) -> str:
        calls.append(prompt)
        if "PMID: 27113485" in prompt:
            return json.dumps({"verdict": "not_studied"})
        return json.dumps({"verdict": "studied"})

    with patch("indication_scout.services.retrieval.query_small_llm", new=identity_llm):
        first = await _judge_pmid_drug_identity(
            "CHEMBL192",
            ["sildenafil", "viagra"],
            [sildenafil, pf_compound],
            tmp_path,
        )
        second = await _judge_pmid_drug_identity(
            "CHEMBL192",
            ["sildenafil", "viagra"],
            [pf_compound, neighbour, sildenafil],
            tmp_path,
        )
        fresh_cache = await _judge_pmid_drug_identity(
            "CHEMBL192",
            ["sildenafil", "viagra"],
            [pf_compound],
            tmp_path / "fresh_cache",
        )

    assert first == {"1": "studied", "27113485": "not_studied"}
    assert second == {"27113485": "not_studied", "3": "studied", "1": "studied"}
    assert fresh_cache == {"27113485": "not_studied"}
    assert len(calls) == 2
    assert all("PMID: 27113485" not in prompt for prompt in calls)


async def test_synthesize_class_level_paper_never_counts_as_drug_specific_evidence(svc):
    """A pooled class paper is kept out of the synthesis prompt and every evidence list, so one
    drug-specific study cannot carry class-level papers into the supporting list at full weight.
    """
    abstracts = [
        AbstractResult(
            pmid="11111111",
            title="Sildenafil in stroke",
            abstract="Sildenafil improved recovery against placebo.",
            similarity=0.95,
        ),
        AbstractResult(
            pmid="42051769",
            title="PDE5 inhibitors meta-analysis",
            abstract="Sildenafil or tadalafil increased cerebral blood flow.",
            similarity=0.94,
        ),
    ]
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Sildenafil improved recovery (PMID: 11111111).",
            "strength": "weak",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": True,
            "key_findings": ["Sildenafil improved recovery (PMID: 11111111)."],
        }
    )
    captured: dict[str, str] = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return main

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(
                return_value={"11111111": "studied", "42051769": "class_level"}
            ),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(return_value={"11111111": "supporting"}),
        ),
    ):
        result = await svc.synthesize("CHEMBL192", "ischemic stroke", abstracts)

    assert "PMID: 11111111" in captured["prompt"]
    assert "42051769" not in captured["prompt"]
    assert result.strength == "weak"
    assert result.direction == "supports"
    assert result.evidence_basis == "drug_specific"
    assert result.study_count == 1
    assert result.supporting_pmids == ["11111111"]
    assert result.contradicting_pmids == []
    assert result.relevant_pmids == ["11111111"]
    assert result.contaminated_pmids == ["42051769"]
    assert result.neutral_pmids == []


async def test_synthesize_reports_class_level_basis_when_no_drug_specific_paper(svc):
    """Only class-level papers survive the gates, so the pair is graded ungraded-but-class-backed
    without ever reaching the synthesis model."""
    abstracts = [
        AbstractResult(
            pmid="42051769",
            title="PDE5 inhibitors meta-analysis",
            abstract="Sildenafil or tadalafil increased cerebral blood flow.",
            similarity=0.94,
        )
    ]
    query_llm = AsyncMock()
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={"42051769": "class_level"}),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=query_llm),
    ):
        result = await svc.synthesize("CHEMBL192", "ischemic stroke", abstracts)

    assert query_llm.await_count == 0
    assert result.summary == ""
    assert result.study_count == 0
    assert result.strength == "none"
    assert result.direction == "none"
    assert result.evidence_basis == "class_level"
    assert result.is_observational is None
    assert result.is_animal_only is None
    assert result.key_findings == []
    assert result.supporting_pmids == []
    assert result.contradicting_pmids == []
    assert result.relevant_pmids == []
    assert result.contaminated_pmids == ["42051769"]
    assert result.neutral_pmids == []


async def test_synthesize_target_gate_excludes_disease_observed_but_not_treated(
    tmp_path,
):
    """The drug was given for a different target and the candidate disease was only counted as an
    outcome, so the paper never reaches the synthesis prompt or the evidence lists."""
    svc = RetrievalService(tmp_path)
    abstracts = [
        AbstractResult(
            pmid="11111111",
            title="Sildenafil in stroke",
            abstract="Sildenafil improved recovery against placebo.",
            similarity=0.95,
        ),
        AbstractResult(
            pmid="29092891",
            title="Sildenafil and device thrombosis on Heart Mate II support",
            abstract="Sildenafil was associated with reduced device thrombosis and ischemic stroke.",
            similarity=0.94,
        ),
    ]
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Sildenafil improved recovery (PMID: 11111111).",
            "strength": "weak",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": True,
            "key_findings": ["Sildenafil improved recovery (PMID: 11111111)."],
        }
    )
    captured: dict[str, str] = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return main

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={"11111111": "studied", "29092891": "studied"}),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease",
            new=AsyncMock(return_value={"11111111": True, "29092891": False}),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(return_value={"11111111": "supporting"}),
        ),
    ):
        result = await svc.synthesize("CHEMBL192", "ischemic stroke", abstracts)

    assert "PMID: 11111111" in captured["prompt"]
    assert "29092891" not in captured["prompt"]
    assert result.study_count == 1
    assert result.strength == "weak"
    assert result.direction == "supports"
    assert result.evidence_basis == "drug_specific"
    assert result.supporting_pmids == ["11111111"]
    assert result.contradicting_pmids == []
    assert result.relevant_pmids == ["11111111"]
    assert result.contaminated_pmids == ["29092891"]
    assert result.neutral_pmids == []


async def test_pmid_treats_disease_isolated_and_cached_per_pmid(tmp_path):
    """Each target decision sees one abstract and survives changes to the surrounding batch."""
    from indication_scout.services.retrieval import _judge_pmid_treats_disease

    adult_stroke = AbstractResult(
        pmid="12411660",
        title="Sildenafil promotes functional recovery after stroke in rats",
        abstract="Rats received sildenafil after embolic middle cerebral artery occlusion.",
        similarity=0.9,
    )
    neonatal_hi = AbstractResult(
        pmid="28343223",
        title="Sildenafil in the developing ischemic mouse brain",
        abstract="Nine-day-old mice received sildenafil after neonatal hypoxic-ischemic injury.",
        similarity=0.9,
    )
    neighbour = AbstractResult(
        pmid="19717023",
        title="Sildenafil treatment of subacute ischemic stroke",
        abstract="Patients received sildenafil during recovery from ischemic stroke.",
        similarity=0.9,
    )
    calls: list[str] = []

    async def target_llm(prompt: str) -> str:
        calls.append(prompt)
        verdict = "not_treats" if "PMID: 28343223" in prompt else "treats"
        return json.dumps({"verdict": verdict})

    with patch("indication_scout.services.retrieval.query_small_llm", new=target_llm):
        first = await _judge_pmid_treats_disease(
            "CHEMBL192",
            "sildenafil",
            "ischemic stroke",
            [adult_stroke, neonatal_hi],
            tmp_path,
        )
        second = await _judge_pmid_treats_disease(
            "CHEMBL192",
            "sildenafil",
            "ischemic stroke",
            [neighbour, neonatal_hi, adult_stroke],
            tmp_path,
        )
        fresh = await _judge_pmid_treats_disease(
            "CHEMBL192",
            "sildenafil",
            "ischemic stroke",
            [neonatal_hi],
            tmp_path / "fresh",
        )

    assert first == {"12411660": True, "28343223": False}
    assert second == {"19717023": True, "28343223": False, "12411660": True}
    assert fresh == {"28343223": False}
    assert len(calls) == 4
    assert all(not ("12411660" in prompt and "28343223" in prompt) for prompt in calls)


async def test_synthesize_drug_identity_gate_excludes_wrong_drug_before_batch(svc):
    """The batch model cannot count a PF-00489791 paper as sildenafil evidence because the
    authoritative identity gate removes it from the synthesis prompt and final evidence lists.
    """
    abstracts = [
        AbstractResult(
            pmid="11111111",
            title="Sildenafil trial",
            abstract="Sildenafil improved the primary endpoint against placebo.",
            similarity=0.95,
        ),
        AbstractResult(
            pmid="27113485",
            title="PF-00489791 trial",
            abstract="PF-00489791 reduced albuminuria against placebo.",
            similarity=0.94,
        ),
    ]
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Sildenafil improved the primary endpoint (PMID: 11111111).",
            "strength": "moderate",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": [
                "Sildenafil improved the primary endpoint (PMID: 11111111)."
            ],
        }
    )
    captured: dict[str, str] = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return main

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil", "viagra"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(
                return_value={"11111111": "studied", "27113485": "not_studied"}
            ),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_directions",
            new=AsyncMock(return_value={"11111111": "supporting"}),
        ),
    ):
        result = await svc.synthesize("CHEMBL192", "diabetic nephropathy", abstracts)

    assert "PMID: 11111111" in captured["prompt"]
    assert "27113485" not in captured["prompt"]
    assert "PF-00489791" not in captured["prompt"]
    assert result.model_dump() == {
        "summary": "Sildenafil improved the primary endpoint (PMID: 11111111).",
        "study_count": 1,
        "strength": "moderate",
        "direction": "supports",
        "evidence_basis": "drug_specific",
        "is_observational": False,
        "is_animal_only": False,
        "key_findings": ["Sildenafil improved the primary endpoint (PMID: 11111111)."],
        "supporting_pmids": ["11111111"],
        "contradicting_pmids": [],
        "relevant_pmids": ["11111111"],
        "contaminated_pmids": ["27113485"],
        "neutral_pmids": [],
        "safety_summary": "",
        "regulatory_safety_summary": "",
        "regulatory_safety_full_labels": "",
        "pharmacovigilance_summary": "",
        "literature_safety_summary": "",
        "label_safety_available": None,
        "safety_pmids": [],
        "safety_severity": None,
        "indication_harm": None,
        "indication_harm_summary": "",
        "indication_harm_pmids": [],
    }


# --- _judge_pmid_directions sub-call (the per-PMID direction authority) ---


async def test_judge_pmid_directions_parses_and_validates():
    """The sub-call returns a {pmid: direction} map, keeping only valid directions for PMIDs that
    were actually sent (an out-of-set or unrecognized verdict is dropped, not trusted).
    """
    from indication_scout.services.retrieval import _judge_pmid_directions

    abstracts = [
        AbstractResult(pmid="1", title="t1", abstract="a1", similarity=0.9),
        AbstractResult(pmid="2", title="t2", abstract="a2", similarity=0.9),
    ]
    resp = json.dumps(
        {"1": "supporting", "2": "neutral", "999": "supporting", "3": "bogus"}
    )
    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(return_value=resp),
    ):
        out = await _judge_pmid_directions("metformin", "colorectal cancer", abstracts)
    # neutral is a valid verdict and kept; out-of-set (999) and unrecognized (bogus) dropped.
    assert out == {"1": "supporting", "2": "neutral"}


async def test_synthesize_neutral_pmid_excluded_from_both_lists(svc):
    """A relevant-but-non-efficacy abstract (PK/safety) labeled 'neutral' by the direction sub-call
    stays relevant (counts in study_count) but is in NEITHER supporting nor contradicting, so it
    can't flip a clean 'supports' to 'mixed' (thalidomide × prostate PK study)."""
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Efficacy plus a PK study (PMID: 11111111; PMID: 22222222).",
            "strength": "moderate",
            "is_observational": False,
            "key_findings": [],
        }
    )
    sub = json.dumps({"11111111": "supporting", "22222222": "neutral"})
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=main),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=sub),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "prostate cancer", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == ["11111111"]
    assert "22222222" not in result.contradicting_pmids
    assert "22222222" in result.relevant_pmids  # neutral still counts as relevant
    assert result.neutral_pmids == ["22222222"]  # surfaced as context, not dropped
    assert result.study_count == 2
    assert result.direction == "supports"  # the PK paper did not make it "mixed"


async def test_synthesize_focuses_final_judgment_when_papers_disagree(svc):
    """A focused judgment replaces the batch model's mixed direction and matching prose."""
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "contradicting"},
            "evidence_basis": "drug_specific",
            "summary": "The evidence is mixed.",
            "strength": "moderate",
            "direction": "mixed",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": ["The papers disagree."],
        }
    )
    focused = json.dumps(
        {
            "direction": "contradicts",
            "summary": "The controlled evidence contradicts efficacy (PMID: 22222222).",
            "key_findings": ["The controlled trial found no benefit (PMID: 22222222)."],
        }
    )
    sub = json.dumps({"11111111": "supporting", "22222222": "contradicting"})
    query_llm = AsyncMock(side_effect=[main, focused])
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={"11111111": "studied", "22222222": "studied"}),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=query_llm),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=sub),
        ),
    ):
        result = await svc.synthesize("CHEMBL894", "ptsd", _SAMPLE_ABSTRACTS)

    assert (
        result.summary
        == "The controlled evidence contradicts efficacy (PMID: 22222222)."
    )
    assert result.study_count == 2
    assert result.strength == "moderate"
    assert result.direction == "contradicts"
    assert result.evidence_basis == "drug_specific"
    assert result.is_observational is False
    assert result.is_animal_only is False
    assert result.key_findings == [
        "The controlled trial found no benefit (PMID: 22222222)."
    ]
    assert result.supporting_pmids == ["11111111"]
    assert result.contradicting_pmids == ["22222222"]
    assert result.relevant_pmids == ["11111111", "22222222"]
    assert result.contaminated_pmids == []
    assert result.neutral_pmids == []
    assert query_llm.await_count == 2


async def test_all_neutral_abstracts_force_direction_none(svc):
    """Every relevant abstract non-efficacy (PK / safety / mechanism): there is no result in either
    direction, so the deterministic rollup must overwrite the LLM's claimed direction with "none".
    Without this the LLM's word survived unchecked and the non-zero study_count carried the pair past
    the supervisor's zero-evidence gate (sildenafil x astrocytoma read "weak, supports" on four
    mechanism abstracts and an empty supporting list)."""
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Mechanism only (PMID: 11111111; PMID: 22222222).",
            "strength": "weak",
            "direction": "supports",
            "is_observational": False,
            "key_findings": [],
        }
    )
    sub = json.dumps({"11111111": "neutral", "22222222": "neutral"})
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=main),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=sub),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "prostate cancer", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == []
    assert result.contradicting_pmids == []
    assert result.neutral_pmids == ["11111111", "22222222"]
    assert result.study_count == 2  # still relevant, still cited as context
    assert result.direction == "none"  # overwrites the LLM's "supports"


async def test_judge_pmid_directions_empty_on_unparseable():
    from indication_scout.services.retrieval import _judge_pmid_directions

    abstracts = [AbstractResult(pmid="1", title="t", abstract="a", similarity=0.9)]
    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(return_value="not json"),
    ):
        out = await _judge_pmid_directions("metformin", "x", abstracts)
    assert out == {}


async def test_judge_pmid_directions_no_relevant_skips_call():
    """No relevant abstracts → no sub-call made (empty map)."""
    from indication_scout.services.retrieval import _judge_pmid_directions

    mock = AsyncMock(return_value="{}")
    with patch("indication_scout.services.retrieval.query_small_llm", new=mock):
        out = await _judge_pmid_directions("metformin", "x", [])
    assert out == {}
    mock.assert_not_awaited()


async def test_synthesize_pmid_direction_overrides_verdict(svc):
    """The sub-call is authoritative: a PMID synthesize labeled 'supporting' is moved to
    contradicting when the direction sub-call says so (metformin × steatosis: a comparator's
    benefit must not read as supporting for metformin)."""
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Evidence (PMID: 11111111; PMID: 22222222).",
            "strength": "moderate",
            "is_observational": False,
            "key_findings": [],
        }
    )
    sub = json.dumps({"11111111": "contradicting", "22222222": "contradicting"})
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=main),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=sub),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "hepatic steatosis", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == []
    assert set(result.contradicting_pmids) == {"11111111", "22222222"}
    assert result.direction == "contradicts"


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


# --- safety_search (delegates to pubmed_ae.search_adverse_events: drug-level + disease-scoped) ---


def _safety_abs(pmid, title):
    from indication_scout.models.model_pubmed_abstract import PubmedAbstract

    return PubmedAbstract(pmid=pmid, title=title, abstract="body")


async def test_safety_search_fetches_and_dedupes_both_pools(svc):
    """safety_search fetches the DRUG-LEVEL pool and (when disease given) the DISEASE-SCOPED pool,
    deduped drug-level-first, as AbstractResults."""
    calls = []

    async def fake_search(
        pref,
        cache_dir,
        date_before=None,
        disease=None,
        disease_aliases=None,
    ):
        calls.append((disease, disease_aliases))
        if disease is None:
            return [_safety_abs("111", "drug-wide"), _safety_abs("222", "shared")]
        return [_safety_abs("222", "shared"), _safety_abs("333", "disease-only")]

    from indication_scout.utils.cache import cache_set

    cache_set(
        "disease_aliases",
        {"chembl_id": "CHEMBL122", "disease": "colorectal cancer"},
        ["bowel cancer"],
        svc.cache_dir,
    )

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.search_adverse_events",
            new=fake_search,
        ),
    ):
        result = await svc.safety_search("CHEMBL122", disease="colorectal cancer")

    # One drug-level call (disease=None) + one disease-scoped call.
    assert calls == [(None, None), ("colorectal cancer", ["bowel cancer"])]
    assert [r.pmid for r in result.drug_level] == ["111", "222"]
    assert [r.pmid for r in result.disease_scoped] == ["222", "333"]
    assert [r.pmid for r in result.combined] == ["111", "222", "333"]
    assert result.drug_level[0].title == "drug-wide"


async def test_safety_search_drug_level_only_when_no_disease(svc):
    """With no disease, safety_search runs only the drug-level pool (one call)."""
    calls = []

    async def fake_search(pref, cache_dir, date_before=None, disease=None):
        calls.append(disease)
        return [_safety_abs("111", "x")]

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.search_adverse_events",
            new=fake_search,
        ),
    ):
        result = await svc.safety_search("CHEMBL122")

    assert calls == [None]
    assert [r.pmid for r in result.drug_level] == ["111"]
    assert result.disease_scoped == []
    assert [r.pmid for r in result.combined] == ["111"]


# --- summarize_safety (source-separated production facts; literature-only holdout) ---

_SAFETY_ABSTRACTS = [
    AbstractResult(
        pmid="11696466",
        title="Cardiovascular thrombotic events in trials of rofecoxib.",
        abstract="Rofecoxib increased cardiovascular thrombotic events versus placebo.",
        similarity=0.0,
    ),
]

_SAMPLE_SAFETY_LLM_RESPONSE = json.dumps(
    {
        "verdicts": [
            {
                "pmid": "11696466",
                "status": "confirmed_harm",
                "adverse_outcome": "increased cardiovascular thrombotic events",
                "evidence_quote": (
                    "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                ),
            }
        ]
    }
)


def _safety_profile() -> DrugProfile:
    from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning

    return DrugProfile(
        chembl_id="CHEMBL122",
        drug_warnings=[
            DrugWarning(
                warning_type="Withdrawn",
                description="Increased risk of cardiovascular events",
                toxicity_class="cardiotoxicity",
                year=2004,
            ),
        ],
        adverse_events=[
            AdverseEvent(
                name="myocardial infarction", count=100, log_likelihood_ratio=8913.0
            ),
        ],
    )


async def test_summarize_safety_prod_uses_ot_signal_and_severity(svc):
    """Production separates exact label text from Open Targets warning and FAERS metadata."""
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    with (
        patch(
            "indication_scout.services.retrieval.RetrievalService._get_label_safety_records",
            new=AsyncMock(
                return_value=[
                    FDALabelSafetyRecord(
                        set_id="set-1",
                        effective_time="20260101",
                        boxed_warnings=["WARNING: Cardiovascular thrombotic risk."],
                    )
                ]
            ),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(
                side_effect=AssertionError("production must be deterministic")
            ),
        ),
    ):
        result = await svc.summarize_safety(
            "CHEMBL122", "arthritis", _safety_profile(), _SAFETY_ABSTRACTS
        )

    assert "WARNING: Cardiovascular thrombotic risk." in result.regulatory_summary
    assert "warning type: Withdrawn" in result.regulatory_summary
    assert "toxicity categories: cardiotoxicity" in result.regulatory_summary
    assert "myocardial infarction" in result.pharmacovigilance_summary
    assert "not proof of causation" in result.pharmacovigilance_summary
    assert result.literature_summary == ""
    assert result.safety_pmids == []
    assert result.safety_severity == "withdrawn"
    assert result.label_data_available is True


async def test_format_regulatory_safety_single_text_skips_llm(svc):
    """When every label agrees verbatim, no LLM call is needed — return the text directly."""
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    label_records = [
        FDALabelSafetyRecord(
            set_id="set-old",
            effective_time="20240101",
            brand_names=["WELLBUTRIN SR"],
            boxed_warnings=["WARNING: the same warning."],
        ),
        FDALabelSafetyRecord(
            set_id="set-new",
            effective_time="20260101",
            brand_names=["Bupropion Hydrochloride XL"],
            boxed_warnings=["WARNING: the same warning."],
        ),
    ]

    with (
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
        ),
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(
                side_effect=AssertionError("get_all_drug_names must not be called")
            ),
        ),
    ):
        summary, full_labels = await svc._format_regulatory_safety(
            "CHEMBL894", label_records, warnings=[]
        )

    assert summary == (
        "FDA label boxed-warning text (all 2 FDA-approved product labels agree "
        "verbatim):\nWARNING: the same warning."
    )
    assert "WELLBUTRIN SR" in full_labels
    assert "Bupropion Hydrochloride XL" in full_labels
    assert full_labels.count("WARNING: the same warning.") == 2


async def test_format_regulatory_safety_digests_distinct_texts_via_llm(svc):
    """Distinct boxed-warning texts across products go to the LLM for a summarized digest;
    the full verbatim text of every product is still returned separately for the appendix.
    """
    from indication_scout.models.model_fda import FDALabelSafetyRecord

    label_records = [
        FDALabelSafetyRecord(
            set_id="set-old",
            effective_time="20240101",
            brand_names=["WELLBUTRIN SR"],
            boxed_warnings=["WARNING: older wording of the same warning."],
        ),
        FDALabelSafetyRecord(
            set_id="set-new",
            effective_time="20260101",
            brand_names=["Bupropion Hydrochloride XL"],
            boxed_warnings=["WARNING: newest wording, with a differing age cutoff."],
        ),
    ]
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Both labels warn of suicidality risk; the newest label differs on age cutoff."

    with (
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion", "wellbutrin"]),
        ),
    ):
        summary, full_labels = await svc._format_regulatory_safety(
            "CHEMBL894", label_records, warnings=[]
        )

    assert "bupropion" in captured["prompt"]
    assert "WARNING: older wording of the same warning." in captured["prompt"]
    assert "WARNING: newest wording, with a differing age cutoff." in captured["prompt"]
    assert "2 product labels, LLM-summarized" in summary
    assert "differs on age cutoff" in summary
    assert "WELLBUTRIN SR" in full_labels
    assert "Bupropion Hydrochloride XL" in full_labels
    assert "WARNING: older wording of the same warning." in full_labels
    assert "WARNING: newest wording, with a differing age cutoff." in full_labels


async def test_summarize_safety_holdout_omits_ot_signal(svc):
    """Holdout (date_before set): OT warnings/AEs are OMITTED from the prompt (undateable → would
    leak); literature findings require a source-verifiable exact quote."""
    from datetime import date

    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_SAFETY_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
    ):
        result = await svc.summarize_safety(
            "CHEMBL122",
            "arthritis",
            _safety_profile(),
            _SAFETY_ABSTRACTS,
            date_before=date(2003, 1, 1),
        )

    # OT warning text is NOT in the prompt (suppressed in holdout).
    assert "Withdrawn" not in captured["prompt"]
    assert "myocardial infarction" not in captured["prompt"]
    assert result.regulatory_summary == ""
    assert result.pharmacovigilance_summary == ""
    assert result.literature_summary == (
        'Date-eligible literature reported: "Rofecoxib increased cardiovascular thrombotic '
        'events versus placebo." '
        "(PMID: 11696466)."
    )
    assert result.safety_pmids == ["11696466"]
    assert result.safety_severity is None
    assert result.label_data_available is None


async def test_summarize_safety_no_signal_returns_unavailable(svc):
    """No source signal returns empty summaries and unavailable severity."""
    with (
        patch.object(svc, "_get_label_safety_records", new=AsyncMock(return_value=[])),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
        ),
    ):
        result = await svc.summarize_safety(
            "CHEMBL999", "arthritis", DrugProfile(chembl_id="CHEMBL999"), []
        )

    assert result.regulatory_summary == ""
    assert result.pharmacovigilance_summary == ""
    assert result.literature_summary == ""
    assert result.safety_summary == ""
    assert result.safety_pmids == []
    assert result.safety_severity is None
    assert result.label_data_available is True


async def test_summarize_safety_handles_null_adverse_event_fields(svc):
    """Regression: an OT AdverseEvent with null count / log_likelihood_ratio must not crash the
    prompt's format string (the ':.1f' would raise TypeError on None)."""
    from indication_scout.models.model_open_targets import AdverseEvent, DrugWarning

    profile = DrugProfile(
        chembl_id="CHEMBL122",
        drug_warnings=[DrugWarning(warning_type="Black Box Warning")],
        adverse_events=[
            AdverseEvent(name="hepatotoxicity", count=None, log_likelihood_ratio=None)
        ],
    )
    with (
        patch.object(svc, "_get_label_safety_records", new=AsyncMock(return_value=[])),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(
                side_effect=AssertionError("production must be deterministic")
            ),
        ),
    ):
        result = await svc.summarize_safety(
            "CHEMBL122", "arthritis", profile, _SAFETY_ABSTRACTS
        )

    assert "warning type: Black Box Warning" in result.regulatory_summary
    assert result.pharmacovigilance_summary == ""
    assert "reports: 0" not in result.safety_summary
    assert "logLR: 0.0" not in result.safety_summary
    assert result.safety_severity == "black_box"


# --- classify_indication_harm (the validated concrete disease-specific question) ---


async def test_classify_indication_harm_parses_true(svc):
    """A confirmed per-PMID harm with a source quote is aggregated."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": "patients",
                    "adverse_outcome": "increased cardiovascular thrombotic events",
                    "evidence_quote": (
                        "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                    ),
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is True
    assert summary == (
        "Disease-scoped literature for rofecoxib in colorectal cancer reported: "
        '"Rofecoxib increased cardiovascular thrombotic events versus placebo." '
        "(PMID: 11696466)."
    )
    assert pmids == ["11696466"]


async def test_classify_indication_harm_false_clears_summary_and_pmids(svc):
    """A fully reviewed safety-assessment-only paper is not a harm finding."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "safety_assessed_only",
                    "study_subjects": "patients",
                    "adverse_outcome": None,
                    "evidence_quote": None,
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await svc.classify_indication_harm(
            "CHEMBL122", "migraine", _SAFETY_ABSTRACTS
        )

    assert harm is False
    assert summary == ""
    assert pmids == []


async def test_classify_indication_harm_empty_abstracts_no_llm(svc):
    """No disease-scoped abstracts leaves the harm result unavailable."""
    with patch(
        "indication_scout.services.retrieval.query_llm",
        new=AsyncMock(side_effect=AssertionError("query_llm must not be called")),
    ):
        harm, summary, pmids = await svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", []
        )

    assert harm is None
    assert summary == ""
    assert pmids == []


async def test_classify_indication_harm_rejects_unverified_quote(svc):
    """A claimed harm whose quote is absent from the source remains unavailable."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": "patients",
                    "adverse_outcome": "renal failure",
                    "evidence_quote": "Metformin caused renal failure.",
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=resp),
        ),
        patch("indication_scout.services.retrieval.cache_set") as mock_cache_set,
    ):
        harm, summary, pmids = await svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is None
    assert summary == ""
    assert pmids == []
    mock_cache_set.assert_not_called()


@pytest.mark.parametrize("subjects", ["animals", "cells_or_tissue", "unclear", None])
async def test_classify_indication_harm_rejects_non_patient_study(svc, subjects):
    """A verified harm from non-human work is not patient risk for the indication."""
    resp = json.dumps(
        {
            "verdicts": [
                {
                    "pmid": "11696466",
                    "status": "confirmed_harm",
                    "study_subjects": subjects,
                    "adverse_outcome": "increased cardiovascular thrombotic events",
                    "evidence_quote": (
                        "Rofecoxib increased cardiovascular thrombotic events versus placebo."
                    ),
                }
            ]
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["rofecoxib"]),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=resp),
        ),
    ):
        harm, summary, pmids = await svc.classify_indication_harm(
            "CHEMBL122", "colorectal cancer", _SAFETY_ABSTRACTS
        )

    assert harm is False
    assert summary == ""
    assert pmids == []


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


async def test_get_drug_competitors_filters_broad_canonical_after_merge(tmp_path):
    """A merge cannot recreate a generic candidate removed by the raw-data guard."""
    raw = {
        "diseases": {
            "myalgia": {"competitor_a"},
            "arthralgia": {"competitor_b"},
            "psoriasis": {"competitor_c"},
        },
        "drug_indications": [],
    }
    merge_result = {
        "merge": {"pain": ["myalgia", "arthralgia"]},
        "remove": [],
    }
    mock_client = _make_open_targets_mock(raw)

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_client,
        ),
        patch(
            "indication_scout.services.retrieval.merge_duplicate_diseases",
            new=AsyncMock(return_value=merge_result),
        ),
    ):
        result = await RetrievalService(tmp_path).get_drug_competitors("CHEMBL1")

    assert result == {"psoriasis": {"competitor_c"}}


async def test_get_drug_competitors_merge_retains_empty_competitor_set(tmp_path):
    from indication_scout.utils.cache import cache_get

    raw = {
        "diseases": {
            "sleep disorder alpha": set(),
            "sleep disorder beta": set(),
        },
        "drug_indications": [],
    }
    merge_result = {
        "merge": {
            "canonical sleep disorder": [
                "sleep disorder alpha",
                "sleep disorder beta",
            ]
        },
        "remove": [],
    }
    mock_client = _make_open_targets_mock(raw)

    with (
        patch(
            "indication_scout.services.retrieval.OpenTargetsClient",
            return_value=mock_client,
        ),
        patch(
            "indication_scout.services.retrieval.merge_duplicate_diseases",
            new=AsyncMock(return_value=merge_result),
        ),
    ):
        result = await RetrievalService(tmp_path).get_drug_competitors("CHEMBL1")

    assert result == {"canonical sleep disorder": set()}
    assert cache_get(
        "disease_aliases",
        {"chembl_id": "CHEMBL1", "disease": "canonical sleep disorder"},
        tmp_path,
    ) == ["sleep disorder alpha", "sleep disorder beta"]


async def test_get_drug_competitors_returns_cached(tmp_path):
    """A cache hit is filtered without calling the client or LLM."""
    from indication_scout.config import get_settings
    from indication_scout.utils.cache import cache_set

    cached = {
        "depression": ["competitor_a"],
        "pain": ["competitor_b"],
        "arthritis": ["competitor_c"],
        "inflammation": ["competitor_d"],
    }
    cache_set(
        "competitors_merged",
        {
            "chembl_id": "CHEMBL1",
            "date_before": None,
            "top_k": get_settings().literature_top_k,
            "logic_version": "cache_disease_aliases_v1",
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


async def test_get_drug_competitors_returns_cached_empty_result(tmp_path):
    from indication_scout.config import get_settings
    from indication_scout.utils.cache import cache_set

    cache_set(
        "competitors_merged",
        {
            "chembl_id": "CHEMBL1",
            "date_before": None,
            "top_k": get_settings().literature_top_k,
            "logic_version": "cache_disease_aliases_v1",
        },
        {},
        tmp_path,
    )

    mock_client = AsyncMock()
    with patch(
        "indication_scout.services.retrieval.OpenTargetsClient",
        return_value=mock_client,
    ):
        result = await RetrievalService(tmp_path).get_drug_competitors("CHEMBL1")

    assert result == {}
    mock_client.__aenter__.assert_not_called()
