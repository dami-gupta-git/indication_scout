"""Unit tests for services/retrieval — no network, no LLM calls."""

import json
from collections.abc import Callable
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.constants import (
    LITERATURE_RECENT_BUDGET,
    LITERATURE_SCREEN_BATCH,
    LITERATURE_SCREEN_MAX_DEPTH,
    LITERATURE_SCREEN_TARGET_PASSERS,
    LITERATURE_SHALLOW_DEPTH,
    SYNTHESIS_MAX_TOKENS,
)
from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_clinical_trials import SearchTrialsResult, Trial
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary, PmidJudgment
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    RichDrugData,
    TargetData,
)
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.services import retrieval
from indication_scout.services.retrieval import (
    AbstractResult,
    RetrievalService,
    _recent_pass,
    _union_tagged,
    merge_same_trial,
    nct_ids,
    pub_year,
    select_synthesis_set,
)

_FIND_TRIAL_LINKED_PMIDS = RetrievalService.find_trial_linked_pmids

# --- Fixtures ---


def _judgments(*rows: tuple[str, str, bool, bool]) -> dict[str, PmidJudgment]:
    """Build a per-paper grade result from (pmid, verdict, is_human, is_controlled) rows."""
    return {
        pmid: PmidJudgment(
            verdict=verdict, is_human=is_human, is_controlled=is_controlled, relevance=8
        )
        for pmid, verdict, is_human, is_controlled in rows
    }


def _grades(*rows: tuple[str, str, bool, bool]) -> AsyncMock:
    """Patch value for _judge_pmid_grades returning fixed grades."""
    return AsyncMock(return_value=_judgments(*rows))


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


@pytest.mark.parametrize(
    "resolved, expected_query",
    [
        # label route: the MeSH preferred heading is the disease term
        (
            ("D019969", "Amphetamine-Related Disorders", "label"),
            'metformin AND "Amphetamine-Related Disorders"',
        ),
        # ot_synonym route: the heading came from an Open Targets synonym and can be broader, so the raw name is kept
        (
            ("D019969", "Amphetamine-Related Disorders", "ot_synonym"),
            'metformin AND "methamphetamine dependence"',
        ),
    ],
)
async def test_expand_search_terms_disease_term_by_route(
    tmp_path, metformin_profile, resolved, expected_query
):
    """PubMed queries use the MeSH heading on the label route and the raw disease name on the ot_synonym route."""
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch(
            "indication_scout.services.retrieval.RetrievalService.extract_organ_term",
            new=AsyncMock(return_value="brain"),
        ),
        patch(
            "indication_scout.services.disease_helper.resolve_mesh_id",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value='["metformin AND <DISEASE>"]'),
        ),
    ):
        result = await RetrievalService(tmp_path).expand_search_terms(
            "CHEMBL1431", "methamphetamine dependence", metformin_profile
        )

    assert result == [expected_query]


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


async def test_find_trial_linked_pmids_batches_nct_secondary_source_query(tmp_path):
    from indication_scout.config import get_settings

    cutoff = date(2025, 1, 1)
    trial_result = SearchTrialsResult(
        total_count=3,
        trials=[
            Trial(nct_id="NCT04167306"),
            Trial(nct_id="NCT03169244"),
            Trial(nct_id="NCT04167306"),
        ],
    )
    trials_client = AsyncMock()
    trials_client.__aenter__ = AsyncMock(return_value=trials_client)
    trials_client.__aexit__ = AsyncMock(return_value=None)
    trials_client.search_trials = AsyncMock(return_value=trial_result)
    pubmed_client = AsyncMock()
    pubmed_client.__aenter__ = AsyncMock(return_value=pubmed_client)
    pubmed_client.__aexit__ = AsyncMock(return_value=None)
    pubmed_client.search = AsyncMock(return_value=["40487775", "40487775"])

    with (
        patch(
            "indication_scout.services.retrieval.resolve_mesh_id",
            new=AsyncMock(return_value=("D000437", "Alcoholism", "label")),
        ),
        patch(
            "indication_scout.services.retrieval.ClinicalTrialsClient",
            return_value=trials_client,
        ),
        patch(
            "indication_scout.services.retrieval.PubMedClient",
            return_value=pubmed_client,
        ),
    ):
        result = await _FIND_TRIAL_LINKED_PMIDS(
            RetrievalService(tmp_path), "bupropion", "alcohol dependence", cutoff
        )

    assert result == ["40487775"]
    trials_client.search_trials.assert_awaited_once_with(
        "bupropion", "Alcoholism", date_before=cutoff
    )
    pubmed_client.search.assert_awaited_once_with(
        "NCT04167306[si] OR NCT03169244[si]",
        max_results=get_settings().pubmed_max_results,
        date_before=cutoff,
    )


async def test_cache_trial_reference_abstracts_fetches_missing_paper(svc):
    abstract = PubmedAbstract(
        pmid="40487775",
        title="Bupropion and naltrexone for alcohol use disorder",
        abstract="Randomized clinical trial results.",
    )
    mock_db = MagicMock()
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.fetch_abstracts = AsyncMock(return_value=[abstract])

    with (
        patch.object(svc, "get_stored_pmids", return_value=set()),
        patch.object(
            svc, "embed_abstracts", new=AsyncMock(return_value=[(abstract, [0.1])])
        ),
        patch.object(svc, "insert_abstracts") as insert_abstracts,
        patch("indication_scout.services.retrieval.PubMedClient", return_value=client),
    ):
        result = await svc.cache_trial_reference_abstracts(["40487775"], mock_db, None)

    assert result == ["40487775"]
    client.fetch_abstracts.assert_awaited_once_with(["40487775"])
    insert_abstracts.assert_called_once_with([(abstract, [0.1])], mock_db)


async def test_cache_trial_reference_abstracts_applies_holdout_post_guard(svc):
    cutoff = date(2025, 1, 1)
    mock_db = MagicMock()
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(
            svc,
            "_filter_pmids_by_date",
            new=AsyncMock(return_value=[]),
        ) as filter_pmids,
        patch("indication_scout.services.retrieval.PubMedClient", return_value=client),
    ):
        result = await svc.cache_trial_reference_abstracts(
            ["40487775"], mock_db, cutoff
        )

    assert result == []
    filter_pmids.assert_awaited_once_with(["40487775"], cutoff, mock_db, client)
    client.fetch_abstracts.assert_not_awaited()


# --- semantic_search ---


def _make_db_with_rows(rows: list[tuple]) -> MagicMock:
    """Return a mock Session whose execute().fetchall() yields the given rows. A (pmid, title, abstract, similarity) row
    is given a None pub_date, so it is undated and the recent pass skips it."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        row + (None,) if len(row) == 4 else row for row in rows
    ]
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


@pytest.fixture(autouse=True)
def mock_trial_lane_discovery():
    """Keep ordinary retrieval unit tests independent of registry and PubMed calls."""
    with patch.object(
        RetrievalService,
        "find_trial_linked_pmids",
        new=AsyncMock(return_value=[]),
    ):
        yield


@pytest.fixture
def mock_gates_none():
    """Both screening gates reject every paper, so semantic_search reads to the full depth with no model call."""
    with (
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease",
            new=AsyncMock(return_value={}),
        ),
    ):
        yield


def _passing_gates(passes: Callable[[str], bool]) -> tuple[AsyncMock, AsyncMock]:
    """Gate mocks where a paper passes both gates iff `passes(pmid)`; the treats-disease mock records its inputs."""

    async def identity(chembl_id, drug_names, results, cache_dir):
        return {
            r.pmid: ("studied" if passes(r.pmid) else "other_drug") for r in results
        }

    async def treats(chembl_id, pref_name, disease, results, cache_dir):
        return {r.pmid: True for r in results}

    return AsyncMock(side_effect=identity), AsyncMock(side_effect=treats)


async def test_semantic_search_returns_ranked_dicts(
    svc, mock_pubtypes_empty, mock_gates_none
):
    """Returns list of dicts with pmid, title, abstract, similarity for each DB row."""
    db_rows = [
        ("111", "Metformin in colorectal cancer", "Abstract A", 0.92),
        ("222", "Title B", "A Glucophage cohort, abstract B", 0.85),
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
    assert result[0].title == "Metformin in colorectal cancer"
    assert result[0].abstract == "Abstract A"
    assert result[0].similarity == 0.92
    assert result[1].pmid == "222"
    assert result[1].title == "Title B"
    assert result[1].abstract == "A Glucophage cohort, abstract B"
    assert result[1].similarity == 0.85


@pytest.mark.parametrize(
    "title, abstract, kept",
    [
        ("Smoking cessation trial", "Patients received Wellbutrin SR daily.", True),
        ("Bupropion for ADHD", "Open-label pilot.", True),
        ("Buprenorphine maintenance", "Opioid use disorder cohort.", False),
        ("Antidepressant class review", "SSRIs and SNRIs compared.", False),
    ],
)
async def test_semantic_search_filters_pool_by_drug_name(
    svc, mock_pubtypes_empty, mock_gates_none, title, abstract, kept
):
    """Only papers naming the drug (any accepted name, whole token, title or abstract) are ranked."""
    mock_db = _make_db_with_rows([("111", title, abstract, 0.9)])

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion", "wellbutrin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[[0.1] * 768],
        ),
    ):
        result = await svc.semantic_search("adhd", "CHEMBL894", ["111"], mock_db)

    assert [paper.pmid for paper in result] == (["111"] if kept else [])


async def test_semantic_search_filter_frees_slots_from_other_drug_papers(
    svc, mock_pubtypes_empty, mock_gates_none
):
    """Higher-similarity papers about other drugs no longer displace the drug's own papers.

    60 sertraline papers outrank 20 bupropion papers. Before the filter the shortlist was the top 15 of the
    pool, all sertraline; now it is exactly the 20 bupropion papers, in similarity order.
    """
    other = [
        (f"o{i}", f"Sertraline study {i}", "Sertraline arm.", 0.99 - i / 1000)
        for i in range(60)
    ]
    own = [
        (f"b{i}", f"Bupropion study {i}", "Bupropion arm.", 0.80 - i / 1000)
        for i in range(20)
    ]
    mock_db = _make_db_with_rows(other + own)

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[[0.1] * 768],
        ),
    ):
        result = await svc.semantic_search(
            "depression", "CHEMBL894", [row[0] for row in other + own], mock_db
        )

    assert [paper.pmid for paper in result] == [f"b{i}" for i in range(20)]
    assert result[0].similarity == 0.80
    assert result[-1].similarity == 0.80 - 19 / 1000


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


async def test_semantic_search_returns_screen_depth_papers(
    svc, mock_pubtypes_empty, mock_gates_none
):
    """With no paper passing the gates, the window is read to LITERATURE_SCREEN_MAX_DEPTH and no further."""
    top_k = LITERATURE_SCREEN_MAX_DEPTH
    # Build more rows than the depth so the slice has work to do.
    db_rows = [
        (f"{i}", f"Metformin study {i}", f"Abstract {i}", 0.9 - 0.001 * i)
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


def _metformin_rows(count: int) -> list[tuple]:
    return [
        (f"{i}", f"Metformin study {i}", f"Abstract {i}", 0.9 - 0.001 * i)
        for i in range(count)
    ]


async def _search_with_gates(
    svc, rows: list[tuple], passes: Callable[[str], bool], **kwargs
) -> tuple[list, AsyncMock]:
    identity, treats = _passing_gates(passes)
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[[0.1] * 768],
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=identity,
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease", new=treats
        ),
    ):
        result = await svc.semantic_search(
            "diabetes",
            "CHEMBL1431",
            [row[0] for row in rows],
            _make_db_with_rows(rows),
            **kwargs,
        )
    return result, treats


async def test_semantic_search_stops_at_batch_where_target_passers_reached(
    svc, mock_pubtypes_empty
):
    """Every paper passes: reading stops at the first batch boundary at or past the target, and papers past it are
    never sent to the gates."""
    rows = _metformin_rows(LITERATURE_SCREEN_MAX_DEPTH)
    result, treats = await _search_with_gates(svc, rows, lambda pmid: True)

    batches = -(-LITERATURE_SCREEN_TARGET_PASSERS // LITERATURE_SCREEN_BATCH)
    expected = batches * LITERATURE_SCREEN_BATCH
    assert [paper.pmid for paper in result] == [str(i) for i in range(expected)]
    judged = {r.pmid for call in treats.await_args_list for r in call.args[3]}
    assert judged == {str(i) for i in range(expected)}


async def test_semantic_search_counts_only_papers_passing_both_gates(
    svc, mock_pubtypes_empty
):
    """Half the papers fail the drug-identity gate, so twice as many batches are read to reach the target."""
    rows = _metformin_rows(LITERATURE_SCREEN_MAX_DEPTH)
    result, _ = await _search_with_gates(svc, rows, lambda pmid: int(pmid) % 2 == 0)

    batches = -(-(2 * LITERATURE_SCREEN_TARGET_PASSERS) // LITERATURE_SCREEN_BATCH)
    assert len(result) == batches * LITERATURE_SCREEN_BATCH


async def test_semantic_search_non_adaptive_skips_gates_and_reads_full_depth(
    svc, mock_pubtypes_empty
):
    """adaptive=False (the depth sweep) returns `depth` papers without calling either gate."""
    rows = _metformin_rows(40)
    result, treats = await _search_with_gates(
        svc, rows, lambda pmid: True, depth=25, adaptive=False, recent_pass=False
    )

    assert len(result) == 25
    treats.assert_not_awaited()


@pytest.mark.parametrize(
    "deep, recent, expected_pmids, gates_called",
    [
        # Deep screening off: the fixed shallow window, no gate calls, then the recent pass.
        (
            False,
            True,
            [str(i) for i in range(LITERATURE_SHALLOW_DEPTH)]
            + [
                str(i)
                for i in range(
                    LITERATURE_SHALLOW_DEPTH,
                    LITERATURE_SHALLOW_DEPTH + LITERATURE_RECENT_BUDGET,
                )
            ],
            False,
        ),
        # Both off: the fixed shallow window only.
        (False, False, [str(i) for i in range(LITERATURE_SHALLOW_DEPTH)], False),
        # Recent pass off: the adaptive window only (every paper passes, so it stops at the target's batch).
        (
            True,
            False,
            [
                str(i)
                for i in range(
                    -(-LITERATURE_SCREEN_TARGET_PASSERS // LITERATURE_SCREEN_BATCH)
                    * LITERATURE_SCREEN_BATCH
                )
            ],
            True,
        ),
    ],
)
async def test_semantic_search_follows_screening_settings(
    svc, mock_pubtypes_empty, deep, recent, expected_pmids, gates_called
):
    """The literature_deep_screening and literature_recent_pass settings pick the window when the caller passes
    neither `adaptive` nor `recent_pass`. Every row has a NULL pub_date except the recent ones.
    """
    rows = [
        (
            row[0],
            row[1],
            row[2],
            row[3],
            "2025-Mar" if int(row[0]) >= LITERATURE_SHALLOW_DEPTH else None,
        )
        for row in _metformin_rows(LITERATURE_SCREEN_MAX_DEPTH)
    ]
    settings = retrieval._settings.model_copy(
        update={"literature_deep_screening": deep, "literature_recent_pass": recent}
    )
    with patch.object(retrieval, "_settings", settings):
        result, treats = await _search_with_gates(svc, rows, lambda pmid: True)

    assert [paper.pmid for paper in result] == expected_pmids
    assert treats.await_count > 0 if gates_called else treats.await_count == 0


@pytest.mark.parametrize("passes_disease_gate", [True, False])
async def test_semantic_search_trial_lane_requires_both_relevance_gates(
    svc, mock_pubtypes_empty, passes_disease_gate
):
    """A trial-linked paper outside the depth takes the reserved last slot only if it passes both gates."""
    top_k = LITERATURE_SCREEN_MAX_DEPTH
    expected_first = "1"
    expected_last = str(top_k + 1) if passes_disease_gate else str(top_k)
    db_rows = [
        (
            str(index),
            f"Bupropion trial {index}",
            f"Abstract {index}",
            1.0 - index / 1000,
        )
        for index in range(1, top_k + 3)
    ]
    mock_db = _make_db_with_rows(db_rows)

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["bupropion"]),
        ),
        patch(
            "indication_scout.services.retrieval.embed_async",
            return_value=[[0.1] * 768],
        ),
        patch.object(
            svc,
            "find_trial_linked_pmids",
            new=AsyncMock(return_value=[str(top_k + 1), str(top_k + 2)]),
        ),
        patch.object(
            svc,
            "cache_trial_reference_abstracts",
            new=AsyncMock(return_value=[str(top_k + 1), str(top_k + 2)]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(
                return_value={
                    str(top_k + 1): "studied",
                    str(top_k + 2): "studied",
                }
            ),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease",
            new=AsyncMock(
                return_value={
                    str(top_k + 1): passes_disease_gate,
                    str(top_k + 2): passes_disease_gate,
                }
            ),
        ),
    ):
        result = await svc.semantic_search(
            "alcohol dependence",
            "CHEMBL894",
            [str(index) for index in range(1, top_k + 3)],
            mock_db,
        )

    assert len(result) == top_k
    assert result[0].pmid == expected_first
    assert result[-1].pmid == expected_last
    assert len({paper.pmid for paper in result}) == top_k
    assert str(top_k + 2) not in {paper.pmid for paper in result}


@pytest.mark.parametrize(
    "pub_date, expected",
    [
        ("2021", 2021),
        ("2021-Nov", 2021),
        ("2021-11-03", 2021),
        (None, None),
        ("Spring 2002", None),
    ],
)
def test_pub_year(pub_date, expected):
    assert pub_year(pub_date) == expected


def test_recent_pass_respects_budget_window_and_dates():
    """Papers are taken in ranked order from the reference year minus LITERATURE_RECENT_YEARS on, skipping window
    papers, undated and older papers, up to LITERATURE_RECENT_BUDGET."""
    ranked = [
        AbstractResult(pmid=str(i), title=f"T{i}", abstract="A", similarity=0.5)
        for i in range(40)
    ]
    years = {str(i): 2024 for i in range(30)}
    years["1"] = 2022
    years.pop("2")

    result = _recent_pass(ranked, years, 2026, exclude={"0"})

    expected = [str(i) for i in range(3, 30)][:LITERATURE_RECENT_BUDGET]
    assert [r.pmid for r in result] == expected


def test_union_tagged_keeps_one_copy_with_both_tags():
    window = [
        AbstractResult(
            pmid="1",
            title="T1",
            abstract="A1",
            similarity=0.9,
            selected_by=["relevance"],
        )
    ]
    extra = [
        AbstractResult(pmid="1", title="T1", abstract="A1", similarity=0.9),
        AbstractResult(pmid="2", title="T2", abstract="A2", similarity=0.4),
    ]

    result = _union_tagged(window, extra, "recent")

    assert len(result) == 2
    assert result[0].pmid == "1"
    assert result[0].title == "T1"
    assert result[0].abstract == "A1"
    assert result[0].similarity == 0.9
    assert result[0].selected_by == ["relevance", "recent"]
    assert result[1].pmid == "2"
    assert result[1].selected_by == ["recent"]


@pytest.mark.parametrize(
    "date_before, expected_recent",
    [
        (None, [str(i) for i in range(40, 50)][:LITERATURE_RECENT_BUDGET]),
        (date(2020, 6, 1), [str(i) for i in range(40, 55)][:LITERATURE_RECENT_BUDGET]),
    ],
)
async def test_semantic_search_appends_recent_papers_after_window(
    svc, mock_pubtypes_empty, date_before, expected_recent
):
    """Every paper passes, so the window stops at two batches; the recent pass then adds the later papers dated within
    LITERATURE_RECENT_YEARS of the run year, or of date_before's year when set. Undated papers are skipped.
    """
    rows = [
        (
            str(i),
            f"Metformin study {i}",
            f"Abstract {i}",
            0.9 - 0.001 * i,
            "2025-Mar" if i < 50 else ("2018" if i < 55 else None),
        )
        for i in range(60)
    ]
    with patch.object(
        svc, "_filter_pmids_by_date", new=AsyncMock(side_effect=lambda p, *a: p)
    ):
        result, _ = await _search_with_gates(
            svc, rows, lambda pmid: True, date_before=date_before
        )

    window_size = (
        -(-LITERATURE_SCREEN_TARGET_PASSERS // LITERATURE_SCREEN_BATCH)
        * LITERATURE_SCREEN_BATCH
    )
    assert [r.pmid for r in result] == [str(i) for i in range(window_size)] + (
        expected_recent
    )
    assert result[0].selected_by == ["relevance"]
    extra = result[window_size]
    assert extra.pmid == "40"
    assert extra.title == "Metformin study 40"
    assert extra.abstract == "Abstract 40"
    assert extra.similarity == 0.9 - 0.001 * 40
    assert extra.pubtype == []
    assert extra.selected_by == ["recent"]


async def test_semantic_search_similarity_is_float(
    svc, mock_pubtypes_empty, mock_gates_none
):
    """similarity values in returned dicts are plain Python floats."""
    from decimal import Decimal

    db_rows = [("111", "Metformin", "Abstract", Decimal("0.8765"))]
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


async def test_semantic_search_releases_db_before_fetching_pubtypes(
    svc, mock_gates_none
):
    """The pgvector read transaction ends before the PubMed request starts."""
    db_rows = [("111", "Metformin", "Abstract", 0.9)]
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
            title="Metformin",
            abstract="Abstract",
            similarity=0.9,
            pubtype=[],
            selected_by=["relevance"],
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

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(
                    ("11111111", "supporting", True, True),
                    ("22222222", "supporting", False, False),
                )
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

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
        captured["prompt"] = prompt
        return _SAMPLE_LLM_RESPONSE

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["SENTINEL_PREF", "SENTINEL_SYN"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(
                    ("11111111", "supporting", True, True),
                    ("22222222", "supporting", False, False),
                )
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "supporting", False, False),
            ),
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "supporting", False, False),
            ),
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
    """An uncontrolled Parkinson study cannot render as RCT-backed in the candidate card. The one human efficacy paper
    keeps the pair off animal-only, so the model's strength stands."""
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(("6431314", "supporting", True, False))
            ),
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
        "contradicting_trial_pmids": [],
        "secondary_report_pmids": {},
        "screened_not_selected_pmids": [],
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


async def test_synthesize_neutral_paper_cannot_certify_controlled_design(svc):
    """Only a paper carrying an efficacy verdict may set the design word or count as human evidence. sildenafil x
    ischemic stroke was graded on two animal studies while an uncontrolled 12-patient safety study — held as context, in
    neither directional list — supplied the card's "RCT-backed / controlled" claim (its background sentence described
    placebo-controlled RAT work) and later kept the pair off "animal-only" at "moderate". With no human efficacy paper
    the pair is animal-only and capped at weak.
    """
    abstracts = [
        AbstractResult(
            pmid="12411660",
            title="Sildenafil promotes neurogenesis after stroke in rats",
            abstract="Sildenafil improved functional recovery in rats with embolic middle cerebral artery occlusion.",
            similarity=0.95,
        ),
        AbstractResult(
            pmid="19717023",
            title="Safety of sildenafil after acute ischemic stroke",
            abstract=(
                "In several animal studies of rats with ischemic stroke, treatment with sildenafil improved functional "
                "outcomes compared with placebo. We conducted a safety study of sildenafil in 12 patients."
            ),
            similarity=0.93,
        ),
    ]
    response = json.dumps(
        {
            "verdicts": {"12411660": "supporting", "19717023": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Sildenafil aided recovery in rodents (PMID: 12411660).",
            "strength": "moderate",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": ["Rodent recovery improved (PMID: 12411660)."],
        }
    )
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={"12411660": "studied", "19717023": "studied"}),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_treats_disease",
            new=AsyncMock(return_value={"12411660": True, "19717023": True}),
        ),
        patch(
            "indication_scout.services.retrieval.query_llm",
            new=AsyncMock(return_value=response),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(
                    ("12411660", "supporting", False, False),
                    ("19717023", "neutral", True, True),
                )
            ),
        ),
    ):
        result = await svc.synthesize("CHEMBL192", "ischemic stroke", abstracts)

    assert result.is_observational is None
    assert result.is_animal_only is True
    assert result.supporting_pmids == ["12411660"]
    assert result.neutral_pmids == ["19717023"]
    assert result.relevant_pmids == ["12411660", "19717023"]
    assert result.study_count == 1  # the neutral paper is context, not a study
    assert result.strength == "weak"
    assert result.direction == "supports"
    assert result.evidence_basis == "drug_specific"
    assert result.contaminated_pmids == []


async def test_synthesize_degrades_to_safe_floor_on_invalid_json(svc):
    """Unparseable LLM JSON degrades to a safe untested floor (no raise): basis none, strength
    none, all abstracts contaminated. The tolerant parse + floor keeps the pipeline alive. The floor is not cached, so
    a second call asks the LLM again instead of reading the floor back.
    """
    llm_mock = AsyncMock(return_value="not valid json at all")
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=llm_mock),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "supporting", False, False),
            ),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )
        assert llm_mock.await_count == 1
        await svc.synthesize("CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS)
        assert llm_mock.await_count == 2
    assert llm_mock.await_args.kwargs["max_tokens"] == SYNTHESIS_MAX_TOKENS
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
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "supporting", False, False),
            ),
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

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(("11111111", "supporting", False, True))
            ),
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

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(("11111111", "supporting", False, True))
            ),
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
            abstract=(
                "In a randomized placebo-controlled trial, sildenafil improved the primary "
                "endpoint against placebo."
            ),
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

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(
                return_value=_judgments(("11111111", "supporting", True, True))
            ),
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
        "contradicting_trial_pmids": [],
        "secondary_report_pmids": {},
        "screened_not_selected_pmids": [],
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


# --- _judge_pmid_grades (the per-paper direction and design authority) ---


def _grade_json(verdict: str, relevance: int | str) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "is_human": True,
            "is_controlled": True,
            "relevance": relevance,
        }
    )


async def test_pmid_grade_isolated_and_cached_per_pmid(tmp_path):
    """Each grade sees one abstract only, is made once per paper, and is served from cache on a later batch."""
    from indication_scout.services.retrieval import _judge_pmid_grades

    trial = AbstractResult(
        pmid="1",
        title="Trial",
        abstract="ALPHA-TRIAL-ABSTRACT",
        similarity=0.9,
    )
    null = AbstractResult(
        pmid="2",
        title="Null",
        abstract="BETA-NULL-ABSTRACT",
        similarity=0.9,
    )
    prompts: list[str] = []

    async def grade_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "PMID: 1\n" in prompt:
            return _grade_json("supporting", 9)
        return _grade_json("contradicting", 9)

    with patch("indication_scout.services.retrieval.query_small_llm", new=grade_llm):
        first = await _judge_pmid_grades(
            "CHEMBL1431", "metformin", "miscarriage", [trial, null], tmp_path
        )
        second = await _judge_pmid_grades(
            "CHEMBL1431", "metformin", "miscarriage", [null, trial], tmp_path
        )

    expected = {
        "1": PmidJudgment(
            verdict="supporting", is_human=True, is_controlled=True, relevance=9
        ),
        "2": PmidJudgment(
            verdict="contradicting", is_human=True, is_controlled=True, relevance=9
        ),
    }
    assert first == expected
    assert second == expected
    assert len(prompts) == 2
    assert all(
        ("ALPHA-TRIAL-ABSTRACT" in prompt) != ("BETA-NULL-ABSTRACT" in prompt)
        for prompt in prompts
    )


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        _grade_json("bogus", 5),
        _grade_json("supporting", 0),
        _grade_json("supporting", 11),
        _grade_json("supporting", "7"),
        json.dumps({"verdict": "supporting", "is_human": True, "is_controlled": True}),
    ],
)
async def test_pmid_grade_unusable_response_drops_paper_and_is_not_cached(
    tmp_path, response
):
    """An unparseable response, an unknown verdict, or a relevance missing or outside 1-10 drops the paper, and nothing is
    cached, so the next call asks again."""
    from indication_scout.services.retrieval import _judge_pmid_grades

    paper = AbstractResult(pmid="1", title="t", abstract="a", similarity=0.9)
    llm = AsyncMock(return_value=response)
    with patch("indication_scout.services.retrieval.query_small_llm", new=llm):
        first = await _judge_pmid_grades(
            "CHEMBL1431", "metformin", "x", [paper], tmp_path
        )
        second = await _judge_pmid_grades(
            "CHEMBL1431", "metformin", "x", [paper], tmp_path
        )

    assert first == {}
    assert second == {}
    assert llm.await_count == 2


async def test_synthesize_drops_ungraded_paper_and_does_not_cache(svc):
    """A paper the grader could not grade never reaches the synthesis prompt or the evidence lists, and the summary is
    not cached, so the next run grades it again instead of reusing evidence with that paper missing.
    """
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "Metformin reduced CRC incidence (PMID: 11111111).",
            "strength": "moderate",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": [],
        }
    )
    captured: list[str] = []

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
        captured.append(prompt)
        return main

    grades = _grades(("11111111", "supporting", True, True))
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch("indication_scout.services.retrieval._judge_pmid_grades", new=grades),
    ):
        first = await svc.synthesize(
            "CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS
        )
        await svc.synthesize("CHEMBL1431", "colorectal cancer", _SAMPLE_ABSTRACTS)

    assert "PMID: 11111111" in captured[0]
    assert "22222222" not in captured[0]
    assert first.relevant_pmids == ["11111111"]
    assert first.supporting_pmids == ["11111111"]
    assert first.contaminated_pmids == ["22222222"]
    assert len(captured) == 2
    assert grades.await_count == 2


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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "neutral", True, False),
            ),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "prostate cancer", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == ["11111111"]
    assert "22222222" not in result.contradicting_pmids
    assert "22222222" in result.relevant_pmids  # neutral still counts as relevant
    assert result.neutral_pmids == ["22222222"]  # surfaced as context, not dropped
    assert result.study_count == 1  # the neutral paper is context, not a study
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "contradicting", True, True),
            ),
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


async def test_animal_support_does_not_offset_human_contradiction(svc):
    """One animal paper supports, one human trial contradicts: the human paper alone decides the
    direction, so the pair reads "contradicts" without a second judgment call. The animal paper
    stays in the supporting list (sildenafil x acute kidney injury read "mixed" on four rat studies
    against two negative human RCTs)."""
    main = json.dumps(
        {
            "verdicts": {"11111111": "supporting", "22222222": "contradicting"},
            "evidence_basis": "drug_specific",
            "summary": "Rat data support, the human trial does not.",
            "strength": "moderate",
            "direction": "mixed",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": [],
        }
    )
    query_llm = AsyncMock(return_value=main)
    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["sildenafil"]),
        ),
        patch(
            "indication_scout.services.retrieval._judge_pmid_drug_identity",
            new=AsyncMock(return_value={"11111111": "studied", "22222222": "studied"}),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=query_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", False, False),
                ("22222222", "contradicting", True, True),
            ),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL192", "acute kidney injury", _SAMPLE_ABSTRACTS
        )

    assert result.direction == "contradicts"
    assert result.is_animal_only is False
    assert result.supporting_pmids == ["11111111"]
    assert result.contradicting_pmids == ["22222222"]
    assert query_llm.await_count == 1


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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "neutral", True, False),
                ("22222222", "neutral", True, False),
            ),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "prostate cancer", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == []
    assert result.contradicting_pmids == []
    assert result.neutral_pmids == ["11111111", "22222222"]
    assert result.study_count == 0  # neutral papers are context, not studies
    assert result.direction == "none"  # overwrites the LLM's "supports"
    assert result.strength == "none"  # overwrites the LLM's "weak"


async def test_synthesize_pmid_grade_overrides_verdict(svc):
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "contradicting", True, True),
                ("22222222", "contradicting", True, True),
            ),
        ),
    ):
        result = await svc.synthesize(
            "CHEMBL1431", "hepatic steatosis", _SAMPLE_ABSTRACTS
        )
    assert result.supporting_pmids == []
    assert set(result.contradicting_pmids) == {"11111111", "22222222"}
    assert result.direction == "contradicts"


async def test_synthesize_lists_only_non_pilot_human_trials_as_contradicting_trials(
    svc,
):
    """contradicting_trial_pmids holds the contradicting papers that may close a signal: a human study that gave the drug
    and is not a pilot. A pilot trial, a cohort of people already taking the drug and an animal study are contradicting
    papers but not contradicting trials (metformin x Parkinson closed on such papers).
    """
    abstracts = [
        AbstractResult(
            pmid=pmid, title=f"Paper {pmid}", abstract="Results.", similarity=0.9
        )
        for pmid in ("11111111", "22222222", "33333333", "44444444")
    ]
    main = json.dumps(
        {
            "verdicts": dict.fromkeys(
                ("11111111", "22222222", "33333333", "44444444"), "contradicting"
            ),
            "evidence_basis": "drug_specific",
            "summary": "No benefit (PMID: 11111111).",
            "strength": "strong",
            "direction": "contradicts",
            "is_observational": False,
            "key_findings": [],
        }
    )
    grades = {
        "11111111": PmidJudgment(
            verdict="contradicting",
            is_human=True,
            is_controlled=False,
            is_interventional=True,
            is_pilot=False,
            relevance=9,
        ),
        "22222222": PmidJudgment(
            verdict="contradicting",
            is_human=True,
            is_controlled=True,
            is_interventional=True,
            is_pilot=True,
            relevance=9,
        ),
        "33333333": PmidJudgment(
            verdict="contradicting",
            is_human=True,
            is_controlled=False,
            is_interventional=False,
            is_pilot=False,
            relevance=9,
        ),
        "44444444": PmidJudgment(
            verdict="contradicting",
            is_human=False,
            is_controlled=True,
            is_interventional=True,
            is_pilot=False,
            relevance=9,
        ),
    }
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
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(return_value=grades),
        ),
    ):
        result = await svc.synthesize("CHEMBL1431", "parkinson disease", abstracts)

    assert result.contradicting_trial_pmids == ["11111111"]
    assert result.contradicting_pmids == [
        "11111111",
        "22222222",
        "33333333",
        "44444444",
    ]
    assert result.supporting_pmids == []
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
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=_grades(
                ("11111111", "supporting", True, True),
                ("22222222", "supporting", False, False),
            ),
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
    from indication_scout.constants import OPEN_TARGETS_COMPETITOR_LOGIC_VERSION
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
            "prefetch_max": get_settings().open_targets_competitor_prefetch_max,
            "logic_version": "cache_disease_aliases_v2_opus_merge",
            "competitor_logic_version": OPEN_TARGETS_COMPETITOR_LOGIC_VERSION,
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
    from indication_scout.constants import OPEN_TARGETS_COMPETITOR_LOGIC_VERSION
    from indication_scout.utils.cache import cache_set

    cache_set(
        "competitors_merged",
        {
            "chembl_id": "CHEMBL1",
            "date_before": None,
            "top_k": get_settings().literature_top_k,
            "prefetch_max": get_settings().open_targets_competitor_prefetch_max,
            "logic_version": "cache_disease_aliases_v2_opus_merge",
            "competitor_logic_version": OPEN_TARGETS_COMPETITOR_LOGIC_VERSION,
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


# --- same-trial merge and synthesis selection ---


def _paper(
    pmid: str,
    abstract: str = "",
    similarity: float = 0.5,
    selected_by: list[str] | None = None,
) -> AbstractResult:
    return AbstractResult(
        pmid=pmid,
        title=f"Paper {pmid}",
        abstract=abstract,
        similarity=similarity,
        selected_by=selected_by or ["relevance"],
    )


def _grade(verdict: str, relevance: int = 5, trial: bool = False) -> PmidJudgment:
    return PmidJudgment(
        verdict=verdict, is_human=trial, is_controlled=trial, relevance=relevance
    )


def test_nct_ids_normalizes_spaced_number_and_ignores_nine_digits():
    paper = _paper("1", "Registered as NCT 02062190 and NCT01234567; not NCT012345678.")
    assert nct_ids(paper) == frozenset({"NCT02062190", "NCT01234567"})


def test_merge_same_trial_groups_same_direction_under_highest_score():
    papers = [
        _paper("1", "Primary report (NCT01234567)."),
        _paper("2", "Follow-up of NCT01234567."),
        _paper("3", "Unregistered study."),
    ]
    grades = {
        "1": _grade("supporting", relevance=6),
        "2": _grade("supporting", relevance=9),
        "3": _grade("supporting", relevance=7),
    }

    kept, secondary = merge_same_trial(papers, grades)

    assert [paper.pmid for paper in kept] == ["2", "3"]
    assert secondary == {"2": ["1"]}


@pytest.mark.parametrize(
    "abstracts, verdicts",
    [
        # Same trial, different directions: a negative secondary analysis reaches synthesis next to the primary.
        (
            ["Primary (NCT01234567).", "Post hoc of NCT01234567."],
            ["supporting", "contradicting"],
        ),
        # Two independent trials with the same direction and near-identical wording.
        (
            ["Met its endpoint (NCT01234567).", "Met its endpoint (NCT07654321)."],
            ["supporting", "supporting"],
        ),
    ],
)
def test_merge_same_trial_keeps_separate_studies(abstracts, verdicts):
    papers = [_paper(str(i), text) for i, text in enumerate(abstracts)]
    grades = {str(i): _grade(verdict) for i, verdict in enumerate(verdicts)}

    kept, secondary = merge_same_trial(papers, grades)

    assert [paper.pmid for paper in kept] == ["0", "1"]
    assert secondary == {}


def test_merge_same_trial_pooled_paper_does_not_chain_two_trials():
    """A pooled analysis citing two trials is not merged with either primary report, so the two trials stay apart."""
    papers = [
        _paper("a", "Trial A (NCT01234567)."),
        _paper("b", "Trial B (NCT07654321)."),
        _paper("pooled", "Pooled NCT01234567 and NCT07654321."),
    ]
    grades = {pmid: _grade("supporting") for pmid in ("a", "b", "pooled")}

    kept, secondary = merge_same_trial(papers, grades)

    assert [paper.pmid for paper in kept] == ["a", "b", "pooled"]
    assert secondary == {}


def test_select_synthesis_set_returns_everything_that_fits():
    papers = [_paper("1"), _paper("2")]
    grades = {"1": _grade("supporting"), "2": _grade("contradicting")}
    assert select_synthesis_set(papers, grades, 2) == papers
    assert select_synthesis_set([], {}, 20) == []


def test_select_synthesis_set_keeps_direction_shares_and_puts_trials_first():
    """30 contradicting (10 of them trials, all scored below the non-trials) and 5 supporting, cap 20: proportional
    slots are 17 and 2, the spare slot goes to the larger direction, and every trial is read before any non-trial.
    """
    contradicting = [_paper(f"c{i}") for i in range(30)]
    supporting = [_paper(f"s{i}") for i in range(5)]
    grades = {
        **{
            f"c{i}": _grade("contradicting", relevance=3, trial=True) for i in range(10)
        },
        **{f"c{i}": _grade("contradicting", relevance=9) for i in range(10, 30)},
        "s0": _grade("supporting", relevance=2, trial=True),
        **{f"s{i}": _grade("supporting", relevance=8) for i in range(1, 5)},
    }

    selected = [
        paper.pmid
        for paper in select_synthesis_set(contradicting + supporting, grades, 20)
    ]

    assert len(selected) == 20
    assert [pmid for pmid in selected if pmid.startswith("c")] == [
        f"c{i}" for i in range(18)
    ]
    assert [pmid for pmid in selected if pmid.startswith("s")] == ["s0", "s1"]


def test_select_synthesis_set_reads_single_paper_direction_and_kept_pass():
    """A direction with one paper keeps it, and the best recent-pass paper is read ahead of higher-scored ones."""
    supporting = [_paper(f"s{i}", similarity=0.9) for i in range(10)]
    recent = _paper("r", similarity=0.1, selected_by=["recent"])
    mixed = _paper("m")
    grades = {
        **{f"s{i}": _grade("supporting", relevance=9) for i in range(10)},
        "r": _grade("supporting", relevance=4),
        "m": _grade("mixed"),
    }

    selected = [
        paper.pmid
        for paper in select_synthesis_set([*supporting, recent, mixed], grades, 5)
    ]

    assert selected == ["s0", "s1", "s2", "r", "m"]


async def test_synthesize_merges_same_trial_and_samples_past_the_cap(svc):
    """Secondary reports and passers left out by the cap reach neither the prompt nor the contaminated list, and the
    prompt's counts describe the whole merged set, not the sample."""
    papers = [
        _paper("10000001", "Primary report of NCT01234567."),
        _paper("10000002", "Secondary report of NCT01234567."),
        _paper("10000003", "Uncontrolled case series."),
        _paper("10000004", "Second case series."),
    ]
    grades = {
        "10000001": _grade("supporting", relevance=9, trial=True),
        "10000002": _grade("supporting", relevance=5, trial=True),
        "10000003": _grade("supporting", relevance=7),
        "10000004": _grade("supporting", relevance=6),
    }
    response = json.dumps(
        {
            "verdicts": {"10000001": "supporting", "10000003": "supporting"},
            "evidence_basis": "drug_specific",
            "summary": "A trial supports it (PMID: 10000001).",
            "strength": "moderate",
            "direction": "supports",
            "is_observational": False,
            "is_animal_only": False,
            "key_findings": [],
        }
    )
    captured: list[str] = []

    async def capture_llm(prompt: str, max_tokens: int | None = None) -> str:
        captured.append(prompt)
        return response

    with (
        patch(
            "indication_scout.services.retrieval.get_all_drug_names",
            new=AsyncMock(return_value=["metformin"]),
        ),
        patch("indication_scout.services.retrieval.query_llm", new=capture_llm),
        patch(
            "indication_scout.services.retrieval._judge_pmid_grades",
            new=AsyncMock(return_value=grades),
        ),
        patch("indication_scout.services.retrieval.LITERATURE_SYNTHESIS_MAX", 2),
    ):
        result = await svc.synthesize("CHEMBL1431", "colorectal cancer", papers)

    prompt = captured[0]
    assert (
        "PMID: 10000001\nYear: unknown\nGraded direction: supporting\nDesign: controlled human"
        in prompt
    )
    assert "PMID: 10000003" in prompt
    assert "PMID: 10000002" not in prompt
    assert "PMID: 10000004" not in prompt
    assert (
        "The 2 abstracts above are a sample of the 3 papers that passed screening."
        in prompt
    )
    assert "Graded direction counts: supporting 3" in prompt
    assert (
        "Study design counts: controlled human 1, non-human (animal or in vitro) 2"
        in prompt
    )
    assert result.secondary_report_pmids == {"10000001": ["10000002"]}
    assert result.screened_not_selected_pmids == ["10000004"]
    assert result.relevant_pmids == ["10000001", "10000003"]
    assert result.supporting_pmids == ["10000001", "10000003"]
    assert result.contaminated_pmids == []
    assert result.study_count == 2
