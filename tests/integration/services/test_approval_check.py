"""Integration tests for approval_check service — hits real openFDA + LLM APIs."""

import pytest

from indication_scout.data_sources.chembl import get_all_drug_names
from indication_scout.data_sources.fda import FDAClient
from indication_scout.services import approval_check
from indication_scout.services.approval_check import (
    extract_approved_from_labels,
    get_fda_approved_disease_mapping,
    list_approved_indications_from_labels,
)


@pytest.mark.approval_aware
@pytest.mark.parametrize(
    "drug_name, candidates, expected",
    [
        # Curated short-circuit (approved): "morbid obesity" is in
        # CURATED_FDA_APPROVED_CANDIDATES["semaglutide"] → forced "approved"
        # without an LLM call. "alzheimer's disease" falls through to the LLM
        # path and should come back "none" (unrelated to semaglutide).
        (
            "semaglutide",
            ["morbid obesity", "alzheimer's disease"],
            {"morbid obesity": "approved", "alzheimer's disease": "none"},
        ),
        # Curated short-circuit (rejected → "none"): "ischemic stroke" is in
        # CURATED_FDA_REJECTED_CANDIDATES["semaglutide"]. "type 2 diabetes
        # mellitus" goes through the LLM against the real labels → "approved".
        (
            "semaglutide",
            ["ischemic stroke", "type 2 diabetes mellitus"],
            {"ischemic stroke": "none", "type 2 diabetes mellitus": "approved"},
        ),
        # The bug this whole change fixes: a SIBLING of an approved indication
        # must NOT be dropped. T1DM (sibling of approved T2DM) → "none" (kept);
        # NAFLD (broader parent of approved MASH) → "contaminated" (kept,
        # suspect trial counts); T2DM/obesity → "approved" (dropped).
        (
            "semaglutide",
            [
                "type 1 diabetes mellitus",
                "non-alcoholic fatty liver disease",
                "obesity",
                "parkinson disease",
            ],
            {
                "type 1 diabetes mellitus": "none",
                "non-alcoholic fatty liver disease": "contaminated",
                "obesity": "approved",
                "parkinson disease": "none",
            },
        ),
        # Curated short-circuit (contaminated): "colorectal cancer" is in
        # CURATED_FDA_CONTAMINATED_CANDIDATES["sotorasib"] (KRAS G12C is only
        # ~3-4% of CRC) → forced "contaminated" without an LLM call.
        (
            "sotorasib",
            ["colorectal cancer"],
            {"colorectal cancer": "contaminated"},
        ),
        # ON-LABEL cause-subtype of a broad approval (the DKD regression). Empagliflozin is
        # approved for broad CKD: "diabetic nephropathy"/"diabetic kidney disease" are the same CKD
        # by cause → "approved" (on-label, dropped) — NOT "none" (the bug). A DISTINCT disease that
        # merely causes CKD ("polycystic kidney disease") is kept ("none"). T2DM is the qualifier
        # population, not a labeled indication for the kidney pairs → "none".
        (
            "empagliflozin",
            [
                "chronic kidney disease",
                "diabetic nephropathy",
                "polycystic kidney disease",
            ],
            {
                "chronic kidney disease": "approved",
                "diabetic nephropathy": "approved",
                "polycystic kidney disease": "none",
            },
        ),
    ],
)
async def test_get_fda_approved_disease_mapping(
    test_cache_dir, drug_name, candidates, expected
):
    """Per-candidate approval-relationship labels across curated and LLM paths."""
    result = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected


@pytest.mark.approval_aware
@pytest.mark.parametrize(
    "drug_name, candidates, expected",
    [
        # Empty drug_name → early return; every candidate keyed "none".
        ("", ["obesity"], {"obesity": "none"}),
        # Empty candidates → early return; empty dict.
        ("semaglutide", [], {}),
    ],
)
async def test_get_fda_approved_disease_mapping_empty_inputs(
    test_cache_dir, drug_name, candidates, expected
):
    """Empty drug_name or empty candidate list short-circuits without API calls."""
    result = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected


@pytest.mark.parametrize(
    "drug_aliases, must_contain_substrings",
    [
        # Semaglutide aliases (Ozempic = T2DM; Wegovy = chronic weight management
        # / obesity + cardiovascular risk reduction; Wegovy 2024 expansion = MASH).
        # We do not pin exact strings because the LLM controls phrasing — instead
        # we assert that each well-known approved indication has at least one
        # substring match in the returned list.
        (
            ["Ozempic", "Wegovy", "semaglutide"],
            [
                ["diabetes"],  # T2DM approval
                ["weight", "obesity"],  # chronic weight management / obesity
                ["cardiovascular", "cv"],  # 2020 MACE risk reduction approval
                ["kidney", "renal"],  # 2025 FLOW / CKD approval
                ["mash", "steatohepatitis", "fatty liver"],  # 2024 Wegovy MASH approval
            ],
        ),
        # Metformin: established T2DM approval. Single, unambiguous indication
        # makes this a low-noise sanity check that the path runs end-to-end.
        (
            ["metformin"],
            [["diabetes"]],
        ),
    ],
)
async def test_list_approved_indications_from_labels_real_drug(
    test_cache_dir, drug_aliases, must_contain_substrings
):
    """End-to-end: fetch real openFDA labels and extract approved indications via the LLM.

    Asserts that each expected approval appears (case-insensitively) somewhere
    in the returned list — at least one of the substrings in each group must
    match at least one returned indication.
    """
    async with FDAClient(cache_dir=test_cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    assert label_texts, f"openFDA returned no label text for {drug_aliases}"

    indications = await list_approved_indications_from_labels(
        label_texts=label_texts,
        cache_dir=test_cache_dir,
    )

    assert isinstance(indications, list)
    assert all(isinstance(x, str) and x for x in indications), indications
    assert len(indications) >= 1, f"no indications extracted: {indications}"

    lowered = [i.lower() for i in indications]
    for substring_group in must_contain_substrings:
        assert any(any(sub in ind for sub in substring_group) for ind in lowered), (
            f"none of {substring_group} found in extracted indications "
            f"{indications} for {drug_aliases}"
        )


async def test_list_approved_indications_from_labels_manual(test_cache_dir):
    """Manual hand-test slot — edit drug_aliases / expectations in place.

    Same shape as test_list_approved_indications_from_labels_real_drug but
    unparametrized so you can tweak the drug, the expected substring groups,
    and the cache path freely without disturbing the regression cases.
    """
    # drug_aliases = ["dasatinib"]
    drug_aliases = await get_all_drug_names("CHEMBL941", test_cache_dir)
    # must_contain_substrings = [
    #     ["diabetes"],
    #     ["weight", "obesity"],
    # ]

    async with FDAClient(cache_dir=test_cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    assert label_texts, f"openFDA returned no label text for {drug_aliases}"

    indications = await list_approved_indications_from_labels(
        label_texts=label_texts,
        cache_dir=test_cache_dir,
    )
    assert indications

    # assert isinstance(indications, list)
    # assert all(isinstance(x, str) and x for x in indications), indications
    # assert len(indications) >= 1, f"no indications extracted: {indications}"
    #
    # lowered = [i.lower() for i in indications]
    # for substring_group in must_contain_substrings:
    #     assert any(
    #         any(sub in ind for sub in substring_group) for ind in lowered
    #     ), (
    #         f"none of {substring_group} found in extracted indications "
    #         f"{indications} for {drug_aliases}"
    #     )


async def test_list_approved_indications_from_labels_empty_input(test_cache_dir):
    """Empty label_texts short-circuits without an LLM call and returns []."""
    result = await list_approved_indications_from_labels(
        label_texts=[],
        cache_dir=test_cache_dir,
    )
    assert result == []


@pytest.mark.parametrize(
    "drug_aliases, candidates, expected",
    [
        # Ozempic / semaglutide: T2DM is approved per label; AD and hypertension
        # are not. Verbatim input casing is preserved in the returned set.
        (
            ["Ozempic", "semaglutide"],
            ["type 2 diabetes mellitus", "alzheimer's disease", "hypertension"],
            {"type 2 diabetes mellitus"},
        ),
        # Metformin: T2DM approved, AD not. Single, low-noise sanity case.
        (
            ["metformin"],
            ["type 2 diabetes mellitus", "alzheimer's disease"],
            {"type 2 diabetes mellitus"},
        ),
    ],
)
async def test_extract_approved_from_labels_real_drug(
    test_cache_dir, drug_aliases, candidates, expected
):
    """End-to-end: fetch real openFDA labels and verify candidate-filtered approvals."""
    async with FDAClient(cache_dir=test_cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    assert label_texts, f"openFDA returned no label text for {drug_aliases}"

    result = await extract_approved_from_labels(
        label_texts=label_texts,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected


async def test_extract_approved_from_labels_empty_input(test_cache_dir):
    """Empty inputs short-circuit without an LLM call and return an empty set."""
    assert (
        await extract_approved_from_labels(
            label_texts=[],
            candidate_diseases=["obesity"],
            cache_dir=test_cache_dir,
        )
        == set()
    )
    assert (
        await extract_approved_from_labels(
            label_texts=["foo"],
            candidate_diseases=[],
            cache_dir=test_cache_dir,
        )
        == set()
    )


@pytest.mark.approval_aware
async def test_get_fda_approved_disease_mapping_uses_cache(tmp_path, monkeypatch):
    """Second call with same (drug, candidates) hits the fda_drug_disease_approval cache.

    Uses an isolated tmp cache_dir (no pre-populated entries). Counts query_llm
    invocations across two identical calls — first call invokes the LLM once,
    second call must skip it entirely and return identical results.
    """
    real_query_llm = approval_check.query_llm
    call_count = {"n": 0}

    async def counting_query_llm(*args, **kwargs):
        call_count["n"] += 1
        return await real_query_llm(*args, **kwargs)

    monkeypatch.setattr(approval_check, "query_llm", counting_query_llm)

    drug_name = "ozempic"
    candidates = ["type 2 diabetes mellitus", "hypertension"]
    expected = {"type 2 diabetes mellitus": True, "hypertension": False}

    first = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=tmp_path,
    )
    assert first == expected
    assert call_count["n"] == 1

    second = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=tmp_path,
    )
    assert second == expected
    assert call_count["n"] == 1, "second call must hit cache and skip the LLM"
