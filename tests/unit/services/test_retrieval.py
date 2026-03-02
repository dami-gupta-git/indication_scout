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
from indication_scout.services.retrieval import (
    build_drug_profile,
    embed_abstracts,
    expand_search_terms,
    extract_organ_term,
    fetch_and_cache,
    fetch_new_abstracts,
    get_stored_pmids,
    insert_abstracts,
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
        name="METFORMIN",
        synonyms=["Glucophage", "Glucophage"],  # intentional duplicate
        trade_names=["Fortamet", "Glucophage"],  # overlap with synonyms
        drug_type="Small molecule",
        is_approved=True,
        max_clinical_phase=4.0,
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
        name="metformin",
        synonyms=["Glucophage", "Fortamet"],
        target_gene_symbols=["PRKAA1", "PRKAA2", "STK11"],
        mechanisms_of_action=[
            "AMP-activated protein kinase activator",
            "mTOR inhibitor",
        ],
        atc_codes=["A10BA02"],
        atc_descriptions=["BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS", "Biguanides"],
        drug_type="Small molecule",
    )


# --- DrugProfile.from_rich_drug_data ---


def test_drug_profile_from_rich_drug_data(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.name == "METFORMIN"
    assert profile.synonyms == ["Glucophage", "Fortamet"]
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]
    assert profile.atc_codes == ["A10BA02"]
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]
    assert profile.drug_type == "Small molecule"


def test_drug_profile_from_rich_drug_data_synonyms_deduped(
    rich_metformin, atc_metformin
):
    """synonyms = drug.synonyms + drug.trade_names, deduplicated, order-preserving."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    # synonyms: ["Glucophage", "Glucophage"] → deduped → ["Glucophage"]
    # trade_names: ["Fortamet", "Glucophage"] → "Glucophage" already seen
    assert profile.synonyms == ["Glucophage", "Fortamet"]


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


async def test_extract_organ_term_returns_stripped_string():
    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(return_value="  colon  "),
    ):
        result = await extract_organ_term("colorectal cancer")
    assert result == "colon"


async def test_extract_organ_term_returns_cached_result(tmp_path):
    from indication_scout.utils.cache import cache_set

    cache_set("organ_term", {"disease_name": "colorectal cancer"}, "colon", tmp_path)

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(),
        ) as mock_llm,
    ):
        result = await extract_organ_term("colorectal cancer")

    assert result == "colon"
    mock_llm.assert_not_called()


# --- expand_search_terms ---


async def test_expand_search_terms_returns_list(tmp_path, metformin_profile):
    llm_response = '["metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
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
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "metformin" in captured["prompt"]
    assert "colorectal cancer" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_targets(tmp_path, metformin_profile):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

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
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

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
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "colon" in captured["prompt"]


async def test_expand_search_terms_deduplicates_output(tmp_path, metformin_profile):
    """Case-duplicate entries in LLM output are deduped; first occurrence casing is preserved."""
    llm_response = '["Metformin AND colorectal cancer", "metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
        )

    assert result == ["Metformin AND colorectal cancer", "biguanides AND colon"]


async def test_expand_search_terms_returns_cached_result(tmp_path, metformin_profile):
    from indication_scout.utils.cache import cache_set

    cached_queries = ["metformin AND colorectal cancer", "biguanides AND colon"]
    cache_set(
        "expand_search_terms",
        {"drug_name": "metformin", "disease_name": "colorectal cancer"},
        cached_queries,
        tmp_path,
    )

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(),
        ) as mock_llm,
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
        )

    assert result == cached_queries
    mock_llm.assert_not_called()


# --- build_drug_profile ---


async def test_build_drug_profile_returns_profile(rich_metformin, atc_metformin):
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
        profile = await build_drug_profile("metformin")

    assert profile.name == "METFORMIN"
    assert profile.synonyms == ["Glucophage", "Fortamet"]
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]
    assert profile.atc_codes == ["A10BA02"]
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]
    assert profile.drug_type == "Small molecule"


async def test_build_drug_profile_fetches_atc_per_code(rich_metformin, atc_metformin):
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
        await build_drug_profile("metformin")

    # rich_metformin has one ATC code: "A10BA02"
    assert mock_chembl.get_atc_description.call_count == 1
    mock_chembl.get_atc_description.assert_called_once_with("A10BA02")


async def test_build_drug_profile_no_atc_codes(rich_metformin):
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
        profile = await build_drug_profile("metformin")

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


def test_get_stored_pmids_returns_present_pmids():
    """Only PMIDs that the DB reports as present are returned.

    The DB mock returns ["111", "222"] as existing rows. The third PMID "333"
    is not in the mock result, so it must not appear in the output.
    """
    mock_db = _make_db_session(["111", "222"])

    result = get_stored_pmids(["111", "222", "333"], mock_db)

    assert result == {"111", "222"}


def test_get_stored_pmids_empty_input_returns_empty_set():
    """Empty input short-circuits before hitting the DB.

    No DB query should be made when there are no PMIDs to check — avoids
    sending a vacuous ANY(ARRAY[]) query to Postgres.
    """
    mock_db = _make_db_session([])

    result = get_stored_pmids([], mock_db)

    assert result == set()
    mock_db.execute.assert_not_called()


def test_get_stored_pmids_all_present():
    """All input PMIDs present in DB → full set returned."""
    mock_db = _make_db_session(["111", "222"])

    result = get_stored_pmids(["111", "222"], mock_db)

    assert result == {"111", "222"}


def test_get_stored_pmids_none_present():
    """No input PMIDs present in DB → empty set returned."""
    mock_db = _make_db_session([])

    result = get_stored_pmids(["111", "222"], mock_db)

    assert result == set()


def test_get_stored_pmids_passes_pmids_to_query():
    """The pmids list is passed as a bind parameter to the SQL query."""
    mock_db = _make_db_session([])
    pmids = ["111", "222", "333"]

    get_stored_pmids(pmids, mock_db)

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
async def test_fetch_new_abstracts(all_pmids, stored_pmids, expected_fetched):
    """fetch_new_abstracts calls fetch_abstracts with only the new PMIDs.

    Three cases: all new, all stored, mixed. When stored_pmids covers everything
    the network call is skipped entirely (fetch_abstracts not called).
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.fetch_abstracts = AsyncMock(
        return_value=[_make_pubmed_abstract(p) for p in expected_fetched]
    )

    with patch(
        "indication_scout.services.retrieval.PubMedClient", return_value=mock_client
    ):
        result = await fetch_new_abstracts(all_pmids, stored_pmids)

    if not expected_fetched:
        mock_client.fetch_abstracts.assert_not_called()
        assert result == []
    else:
        mock_client.fetch_abstracts.assert_called_once_with(expected_fetched)
        assert [r.pmid for r in result] == expected_fetched


# --- embed_abstracts ---


def _make_abstract(pmid: str, title: str, abstract: str | None) -> PubmedAbstract:
    return PubmedAbstract(pmid=pmid, title=title, abstract=abstract)


def test_embed_abstracts_texts_contain_title_and_abstract():
    """embed() is called with '<title>. <abstract>' for each abstract."""
    abstracts = [_make_abstract("1", "My Title", "My abstract text.")]
    mock_vectors = [[0.1] * 768]

    with patch("indication_scout.services.retrieval.embed", return_value=mock_vectors) as mock_embed:
        result = embed_abstracts(abstracts)

    mock_embed.assert_called_once_with(["My Title. My abstract text."])
    assert len(result) == 1
    assert result[0][0].pmid == "1"
    assert result[0][1] == mock_vectors[0]


def test_embed_abstracts_none_abstract_produces_title_dot_space():
    """An abstract of None produces '<title>. ' without crashing."""
    abstracts = [_make_abstract("2", "Only Title", None)]
    mock_vectors = [[0.2] * 768]

    with patch("indication_scout.services.retrieval.embed", return_value=mock_vectors) as mock_embed:
        result = embed_abstracts(abstracts)

    mock_embed.assert_called_once_with(["Only Title. "])
    assert result[0][0].pmid == "2"


def test_embed_abstracts_empty_input_skips_embed():
    """Empty input returns [] without calling embed()."""
    with patch("indication_scout.services.retrieval.embed") as mock_embed:
        result = embed_abstracts([])

    mock_embed.assert_not_called()
    assert result == []


def test_embed_abstracts_vectors_align_to_abstracts_by_index():
    """Each abstract is paired with the vector at the same index."""
    abstracts = [
        _make_abstract("10", "Title A", "Abstract A"),
        _make_abstract("20", "Title B", "Abstract B"),
        _make_abstract("30", "Title C", "Abstract C"),
    ]
    mock_vectors = [[float(i)] * 768 for i in range(3)]

    with patch("indication_scout.services.retrieval.embed", return_value=mock_vectors):
        result = embed_abstracts(abstracts)

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


def test_insert_abstracts_calls_execute_and_commit():
    """session.execute() and session.commit() are called when pairs is non-empty."""
    mock_db = MagicMock()
    pairs = [_make_pair("111"), _make_pair("222")]

    with patch("indication_scout.services.retrieval.insert") as mock_insert:
        mock_stmt = MagicMock()
        mock_insert.return_value.values.return_value.on_conflict_do_nothing.return_value = mock_stmt
        insert_abstracts(pairs, mock_db)

    mock_db.execute.assert_called_once_with(mock_stmt)
    mock_db.commit.assert_called_once()


def test_insert_abstracts_empty_pairs_skips_db():
    """Empty pairs list does not touch the DB."""
    mock_db = MagicMock()

    insert_abstracts([], mock_db)

    mock_db.execute.assert_not_called()
    mock_db.commit.assert_not_called()


def test_insert_abstracts_rows_contain_all_fields():
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
        insert_abstracts([(abstract, vector)], mock_db)

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


async def test_fetch_and_cache_returns_deduped_pmids():
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
        patch("indication_scout.services.retrieval.PubMedClient", return_value=mock_client),
        patch("indication_scout.services.retrieval.embed", return_value=[]),
        patch("indication_scout.services.retrieval.insert"),
    ):
        result = await fetch_and_cache(["query1", "query2"], mock_db)

    assert result == ["111", "222", "333"]
    assert len(result) == len(set(result))


async def test_fetch_and_cache_calls_search_per_query():
    """search() is called once for each query string."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(return_value=[])
    mock_client.fetch_abstracts = AsyncMock(return_value=[])

    with (
        patch("indication_scout.services.retrieval.PubMedClient", return_value=mock_client),
        patch("indication_scout.services.retrieval.insert"),
    ):
        await fetch_and_cache(["q1", "q2", "q3"], mock_db)

    assert mock_client.search.call_count == 3
    mock_client.search.assert_any_call("q1", max_results=500)
    mock_client.search.assert_any_call("q2", max_results=500)
    mock_client.search.assert_any_call("q3", max_results=500)


async def test_fetch_and_cache_empty_queries_returns_empty():
    """Empty query list returns [] without hitting PubMed."""
    mock_db = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.search = AsyncMock(return_value=[])

    with patch("indication_scout.services.retrieval.PubMedClient", return_value=mock_client):
        result = await fetch_and_cache([], mock_db)

    assert result == []
    mock_client.search.assert_not_called()
