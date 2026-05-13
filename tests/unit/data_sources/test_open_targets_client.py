"""Unit tests for OpenTargetsClient."""

from unittest.mock import AsyncMock, patch

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import (
    ClinicalDisease,
    DrugData,
    DrugSummary,
    DrugTarget,
)

# --- OpenTargetsClient configuration ---


def test_default_config():
    """Test that client uses default settings."""
    client = OpenTargetsClient()

    assert client.timeout == 30.0
    assert client.max_retries == 3
    assert client.cache_dir is not None
    assert client.cache_dir == DEFAULT_CACHE_DIR


# --- _parse_drug_data: mechanism_of_action ---


def test_parse_drug_data_mechanisms_of_action(tmp_path):
    """_parse_drug_data builds one MechanismOfAction per row, grouping all targets for that row."""
    raw = {
        "id": "CHEMBL1234",
        "name": "TESTDRUG",
        "synonyms": [],
        "tradeNames": [],
        "drugType": "Small molecule",
        "maximumClinicalStage": "PHASE_2",
        "mechanismsOfAction": {
            "rows": [
                {
                    "mechanismOfAction": "Kinase inhibitor",
                    "actionType": "INHIBITOR",
                    "targets": [
                        {"id": "ENSG00000001", "approvedSymbol": "TGT1"},
                        {"id": "ENSG00000002", "approvedSymbol": "TGT2"},
                    ],
                },
                {
                    "mechanismOfAction": "Receptor modulator",
                    "actionType": "MODULATOR",
                    "targets": [
                        {"id": "ENSG00000003", "approvedSymbol": "TGT3"},
                    ],
                },
            ]
        },
        "indications": {"rows": []},
        "drugWarnings": [],
        "adverseEvents": {"rows": [], "criticalValue": None},
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_drug_data(raw)

    assert len(result.mechanisms_of_action) == 2

    moa0 = result.mechanisms_of_action[0]
    assert moa0.mechanism_of_action == "Kinase inhibitor"
    assert moa0.action_type == "INHIBITOR"
    assert moa0.target_ids == ["ENSG00000001", "ENSG00000002"]
    assert moa0.target_symbols == ["TGT1", "TGT2"]

    moa1 = result.mechanisms_of_action[1]
    assert moa1.mechanism_of_action == "Receptor modulator"
    assert moa1.action_type == "MODULATOR"
    assert moa1.target_ids == ["ENSG00000003"]
    assert moa1.target_symbols == ["TGT3"]

    assert len(result.targets) == 3
    assert result.targets[0].mechanism_of_action == "Kinase inhibitor"
    assert result.targets[1].mechanism_of_action == "Kinase inhibitor"
    assert result.targets[2].mechanism_of_action == "Receptor modulator"


def test_parse_drug_data_mechanisms_of_action_empty_when_no_moa(tmp_path):
    """_parse_drug_data sets mechanisms_of_action to [] when mechanismsOfAction is absent."""
    raw = {
        "id": "CHEMBL1234",
        "name": "TESTDRUG",
        "synonyms": [],
        "tradeNames": [],
        "drugType": "Small molecule",
        "maximumClinicalStage": None,
        "mechanismsOfAction": None,
        "indications": {"rows": []},
        "drugWarnings": [],
        "adverseEvents": {"rows": [], "criticalValue": None},
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_drug_data(raw)

    assert result.mechanisms_of_action == []
    assert result.targets == []


# --- get_drug_competitors: None stage guard ---


async def test_get_drug_competitors_skips_summary_with_none_stage(tmp_path):
    """Summaries with max_clinical_stage=None must be skipped without raising TypeError."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="depression")],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage=None,
            diseases=[ClinicalDisease(disease_name="anxiety")],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("CHEMBL1", min_stage="PHASE_3")

    # anxiety (stage=None) must be absent; depression (PHASE_3) must be present
    assert "anxiety" not in result["diseases"]
    assert "depression" in result["diseases"]


# --- get_drug_competitors: raw diseases returned ---


async def test_get_drug_competitors_returns_raw_diseases(tmp_path):
    """get_drug_competitors returns CompetitorRawData with unmerged diseases and indications."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="narcolepsy")],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage="PHASE_3",
            diseases=[ClinicalDisease(disease_name="narcolepsy-cataplexy syndrome")],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("CHEMBL1", min_stage="PHASE_3")

    assert "narcolepsy" in result["diseases"]
    assert "narcolepsy-cataplexy syndrome" in result["diseases"]
    assert result["drug_indications"] == []


async def test_get_drug_competitors_groups_by_disease_id(tmp_path):
    """Diseases sharing the same disease_id collapse to a single key."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(
            drug_name="competitor_a",
            max_clinical_stage="PHASE_3",
            diseases=[
                ClinicalDisease(
                    disease_from_source="type 2 diabetes",
                    disease_id="MONDO_0005148",
                    disease_name="type 2 diabetes mellitus",
                ),
            ],
        ),
        DrugSummary(
            drug_name="competitor_b",
            max_clinical_stage="PHASE_3",
            diseases=[
                ClinicalDisease(
                    disease_from_source="diabetes mellitus, type 2",
                    disease_id="MONDO_0005148",
                    disease_name="diabetes mellitus",
                ),
            ],
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("CHEMBL1", min_stage="PHASE_3")

    assert len(result["diseases"]) == 1
    assert "type 2 diabetes mellitus" in result["diseases"]
    assert result["diseases"]["type 2 diabetes mellitus"] == {
        "competitor_a",
        "competitor_b",
    }


# --- _parse_drug_data: indication.id ---


def test_parse_drug_data_indication_id(tmp_path):
    """_parse_drug_data populates Indication.id from the row's id field."""
    raw = {
        "id": "CHEMBL1234",
        "name": "TESTDRUG",
        "synonyms": [],
        "tradeNames": [],
        "drugType": "Small molecule",
        "maximumClinicalStage": "APPROVAL",
        "mechanismsOfAction": None,
        "indications": {
            "rows": [
                {
                    "id": "abc123hash",
                    "maxClinicalStage": "APPROVAL",
                    "disease": {"id": "MONDO_0001", "name": "test disease"},
                },
            ]
        },
        "drugWarnings": [],
        "adverseEvents": {"rows": [], "criticalValue": None},
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_drug_data(raw)

    [ind] = result.indications
    assert ind.id == "abc123hash"
    assert ind.disease_id == "MONDO_0001"
    assert ind.disease_name == "test disease"
    assert ind.max_clinical_stage == "APPROVAL"


# --- _parse_drug_summary: drug_type and disease_from_source ---


def test_parse_drug_summary_drug_type_and_disease_from_source(tmp_path):
    """_parse_drug_summary populates drug_type and ClinicalDisease.disease_from_source."""
    raw = {
        "id": "hash1",
        "drug": {
            "id": "CHEMBL999",
            "name": "TESTDRUG",
            "drugType": "Protein",
        },
        "maxClinicalStage": "PHASE_3",
        "diseases": [
            {
                "diseaseFromSource": "Type II diabetes",
                "disease": {"id": "MONDO_0005148", "name": "type 2 diabetes mellitus"},
            },
        ],
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_drug_summary(raw)

    assert result.id == "hash1"
    assert result.drug_id == "CHEMBL999"
    assert result.drug_type == "Protein"
    [d] = result.diseases
    assert d.disease_from_source == "Type II diabetes"
    assert d.disease_id == "MONDO_0005148"
    assert d.disease_name == "type 2 diabetes mellitus"


# --- _parse_expression: RNAExpression.unit and ProteinExpression.cell_types ---


def test_parse_expression_rna_unit_and_cell_types(tmp_path):
    """_parse_expression populates rna.unit and protein.cell_types from raw response."""
    raw = {
        "tissue": {
            "id": "UBERON_0002107",
            "label": "liver",
            "anatomicalSystems": ["digestive system"],
        },
        "rna": {"value": 12.5, "level": 4, "unit": "TPM"},
        "protein": {
            "level": 2,
            "reliability": True,
            "cellType": [
                {"name": "hepatocytes", "level": 3, "reliability": True},
                {"name": "bile duct cells", "level": 1, "reliability": False},
            ],
        },
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_expression(raw)

    assert result.tissue_id == "UBERON_0002107"
    assert result.rna.value == 12.5
    assert result.rna.quantile == 4
    assert result.rna.unit == "TPM"
    assert len(result.protein.cell_types) == 2
    assert result.protein.cell_types[0].name == "hepatocytes"
    assert result.protein.cell_types[0].level == 3
    assert result.protein.cell_types[0].reliability is True
    assert result.protein.cell_types[1].name == "bile duct cells"
    assert result.protein.cell_types[1].level == 1
    assert result.protein.cell_types[1].reliability is False


# --- _parse_phenotype: BiologicalModel.model_id and literature ---


def test_parse_phenotype_biological_model_fields(tmp_path):
    """_parse_phenotype populates BiologicalModel.model_id and literature lists."""
    raw = {
        "modelPhenotypeId": "MP:0001234",
        "modelPhenotypeLabel": "test phenotype",
        "modelPhenotypeClasses": [{"id": "MP:000", "label": "test category"}],
        "biologicalModels": [
            {
                "allelicComposition": "Tgt<tm1> hom",
                "geneticBackground": "C57BL/6",
                "id": "MGI:1234567",
                "literature": ["12345", "67890"],
            },
        ],
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_phenotype(raw)

    assert result.phenotype_id == "MP:0001234"
    [model] = result.biological_models
    assert model.model_id == "MGI:1234567"
    assert model.literature == ["12345", "67890"]
    assert model.allelic_composition == "Tgt<tm1> hom"
    assert model.genetic_background == "C57BL/6"


# --- _parse_constraint: exp, obs, upper_bin6 ---


def test_parse_constraint_all_fields(tmp_path):
    """_parse_constraint populates all GeneticConstraint fields, including exp/obs/upper_bin6."""
    raw = {
        "constraintType": "lof",
        "exp": 143.5,
        "obs": 60.0,
        "oe": 0.418,
        "oeLower": 0.335,
        "oeUpper": 0.515,
        "score": 0.065,
        "upperBin": 1,
        "upperBin6": 1,
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_constraint(raw)

    assert result.constraint_type == "lof"
    assert result.exp == 143.5
    assert result.obs == 60.0
    assert result.oe == 0.418
    assert result.oe_lower == 0.335
    assert result.oe_upper == 0.515
    assert result.score == 0.065
    assert result.upper_bin == 1
    assert result.upper_bin6 == 1


# --- _parse_association: disease_description field ---


def test_parse_association_populates_disease_description(tmp_path):
    """_parse_association copies disease.description from the raw row."""
    raw = {
        "score": 0.77,
        "datatypeScores": [
            {"id": "genetic_association", "score": 0.76},
            {"id": "animal_model", "score": 0.56},
        ],
        "disease": {
            "id": "EFO_0000400",
            "name": "type 2 diabetes mellitus",
            "description": (
                "A type of diabetes mellitus that is characterized by insulin "
                "resistance or desensitization and increased blood glucose levels."
            ),
            "therapeuticAreas": [{"id": "EFO_0000540", "name": "metabolic disease"}],
        },
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_association(raw)

    assert result.disease_id == "EFO_0000400"
    assert result.disease_name == "type 2 diabetes mellitus"
    assert result.disease_description.startswith("A type of diabetes mellitus")
    assert result.overall_score == 0.77
    assert result.datatype_scores == {"genetic_association": 0.76, "animal_model": 0.56}


def test_parse_association_missing_description_defaults_to_empty(tmp_path):
    """When disease.description is absent or null, disease_description is ''."""
    raw = {
        "score": 0.5,
        "datatypeScores": [],
        "disease": {
            "id": "EFO_0000001",
            "name": "some disease",
            "description": None,
            "therapeuticAreas": [],
        },
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_association(raw)

    assert result.disease_description == ""


# --- _parse_target_data: function_descriptions field ---


def test_parse_target_data_populates_function_descriptions(tmp_path):
    """_parse_target_data copies functionDescriptions list from the raw node."""
    raw = {
        "id": "ENSG00000112164",
        "approvedSymbol": "GLP1R",
        "approvedName": "glucagon like peptide 1 receptor",
        "functionDescriptions": [
            "G-protein coupled receptor for glucagon-like peptide 1.",
            "Plays a role in regulating insulin secretion.",
        ],
        "associatedDiseases": {"rows": []},
        "pathways": [],
        "interactions": {"rows": []},
        "drugAndClinicalCandidates": {"rows": []},
        "expressions": [],
        "mousePhenotypes": [],
        "safetyLiabilities": [],
        "geneticConstraint": [],
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_target_data(raw)

    assert result.symbol == "GLP1R"
    assert len(result.function_descriptions) == 2
    assert result.function_descriptions[0].startswith("G-protein coupled")


def test_parse_target_data_missing_function_descriptions_defaults_to_empty(tmp_path):
    """When functionDescriptions is absent or null, function_descriptions is []."""
    raw = {
        "id": "ENSG00000000001",
        "approvedSymbol": "TEST",
        "approvedName": "test target",
        "functionDescriptions": None,
        "associatedDiseases": {"rows": []},
        "pathways": [],
        "interactions": {"rows": []},
        "drugAndClinicalCandidates": {"rows": []},
        "expressions": [],
        "mousePhenotypes": [],
        "safetyLiabilities": [],
        "geneticConstraint": [],
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_target_data(raw)

    assert result.function_descriptions == []


# --- _parse_evidence ---


def test_parse_evidence_all_fields(tmp_path):
    """_parse_evidence populates every EvidenceRecord field from the raw row."""
    raw = {
        "datatypeId": "genetic_association",
        "score": 0.85,
        "directionOnTarget": "LoF",
        "directionOnTrait": "risk",
        "disease": {"id": "EFO_0003847", "name": "Infantile dystonia-parkinsonism"},
        "variantFunctionalConsequence": {
            "id": "SO_0002054",
            "label": "loss_of_function_variant",
        },
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_evidence(raw)

    assert result.disease_id == "EFO_0003847"
    assert result.datatype_id == "genetic_association"
    assert result.score == 0.85
    assert result.direction_on_target == "LoF"
    assert result.direction_on_trait == "risk"
    assert result.variant_functional_consequence is not None
    assert result.variant_functional_consequence.id == "SO_0002054"
    assert result.variant_functional_consequence.label == "loss_of_function_variant"


def test_parse_evidence_missing_direction_fields_stay_none(tmp_path):
    """directionOnTarget / directionOnTrait / vFC absent → stay None."""
    raw = {
        "datatypeId": "literature",
        "score": 0.4,
        "directionOnTarget": None,
        "directionOnTrait": None,
        "disease": {"id": "EFO_000001", "name": "some disease"},
        "variantFunctionalConsequence": None,
    }

    client = OpenTargetsClient(cache_dir=tmp_path)
    result = client._parse_evidence(raw)

    assert result.direction_on_target is None
    assert result.direction_on_trait is None
    assert result.variant_functional_consequence is None


# --- get_target_evidences ---


async def test_get_target_evidences_empty_efo_ids_short_circuits(tmp_path):
    """Empty efo_ids list returns {} without hitting the network."""
    client = OpenTargetsClient(cache_dir=tmp_path)
    mock_gql = AsyncMock()
    with patch.object(client, "_graphql", mock_gql):
        result = await client.get_target_evidences("ENSG00000000001", [])

    assert result == {}
    mock_gql.assert_not_awaited()


def _evidence_row(
    disease_id: str, dir_t: str, dir_trait: str, vfc_label: str | None = None
) -> dict:
    return {
        "datatypeId": "genetic_association",
        "score": 0.8,
        "directionOnTarget": dir_t,
        "directionOnTrait": dir_trait,
        "disease": {"id": disease_id, "name": disease_id},
        "variantFunctionalConsequence": (
            {"id": "SO_0002054", "label": vfc_label} if vfc_label else None
        ),
    }


def _mock_per_efo_response(efo_to_rows: dict[str, list[dict]]):
    """Build a _graphql mock that routes responses by the efoIds argument."""

    async def fake_graphql(url, query, variables=None):
        efo_ids = (variables or {}).get("efoIds") or []
        # Per-efo fan-out: we always call with exactly one efo.
        assert len(efo_ids) == 1, f"expected single-efo query, got {efo_ids}"
        rows = efo_to_rows.get(efo_ids[0], [])
        return {"data": {"target": {"evidences": {"rows": rows}}}}

    return AsyncMock(side_effect=fake_graphql)


async def test_get_target_evidences_fans_out_per_efo(tmp_path):
    """One call per efo_id, each efo gets its own 200-record budget. Rows
    are returned grouped by disease_id; efo_ids with no evidence map to []."""
    client = OpenTargetsClient(cache_dir=tmp_path)
    efo_to_rows = {
        "EFO_0000400": [
            _evidence_row("EFO_0000400", "GoF", "protect"),
            _evidence_row("EFO_0000400", "GoF", "protect"),
        ],
        "EFO_0003847": [
            _evidence_row("EFO_0003847", "LoF", "risk", "loss_of_function_variant"),
        ],
        "EFO_UNRELATED": [],
    }
    mock_gql = _mock_per_efo_response(efo_to_rows)

    with patch.object(client, "_graphql", mock_gql):
        result = await client.get_target_evidences(
            "ENSG00000112164", ["EFO_0000400", "EFO_0003847", "EFO_UNRELATED"]
        )

    # Three efo_ids → three HTTP calls (fanned out).
    assert mock_gql.await_count == 3
    assert set(result.keys()) == {"EFO_0000400", "EFO_0003847", "EFO_UNRELATED"}
    assert len(result["EFO_0000400"]) == 2
    assert len(result["EFO_0003847"]) == 1
    assert result["EFO_UNRELATED"] == []
    idp = result["EFO_0003847"][0]
    assert idp.direction_on_target == "LoF"
    assert idp.variant_functional_consequence.label == "loss_of_function_variant"


async def test_get_target_evidences_caches_per_efo(tmp_path):
    """Each (target, efo) pair caches independently. Re-requesting the same
    efo hits the cache; a new efo triggers one fresh network call."""
    client = OpenTargetsClient(cache_dir=tmp_path)
    efo_to_rows = {
        "EFO_0000400": [_evidence_row("EFO_0000400", "GoF", "protect")],
        "EFO_0001073": [_evidence_row("EFO_0001073", "GoF", "protect")],
    }
    mock_gql = _mock_per_efo_response(efo_to_rows)

    with patch.object(client, "_graphql", mock_gql):
        # Prime both.
        await client.get_target_evidences(
            "ENSG00000112164", ["EFO_0000400", "EFO_0001073"]
        )
        assert mock_gql.await_count == 2
        # Repeat the same pair: zero extra calls.
        await client.get_target_evidences(
            "ENSG00000112164", ["EFO_0000400", "EFO_0001073"]
        )
        assert mock_gql.await_count == 2
        # Add a new efo: one extra call, the cached two stay cached.
        efo_to_rows["EFO_NEW"] = []
        await client.get_target_evidences(
            "ENSG00000112164", ["EFO_0000400", "EFO_0001073", "EFO_NEW"]
        )
        assert mock_gql.await_count == 3


async def test_get_target_evidences_empty_response_per_efo(tmp_path):
    """When the API returns no rows for an efo, that efo maps to []."""
    client = OpenTargetsClient(cache_dir=tmp_path)
    mock_gql = _mock_per_efo_response({"EFO_001": [], "EFO_002": []})

    with patch.object(client, "_graphql", mock_gql):
        result = await client.get_target_evidences(
            "ENSG00000000001", ["EFO_001", "EFO_002"]
        )

    assert result == {"EFO_001": [], "EFO_002": []}


async def test_get_target_evidences_drops_cross_disease_rows(tmp_path):
    """Defence in depth: if OT returns a row whose disease_id does not match
    the requested efo_id, drop it — the per-efo bucket stays clean."""
    client = OpenTargetsClient(cache_dir=tmp_path)
    efo_to_rows = {
        "EFO_A": [
            _evidence_row("EFO_A", "GoF", "protect"),
            _evidence_row("EFO_SOMETHING_ELSE", "LoF", "risk"),  # should be dropped
        ],
    }
    mock_gql = _mock_per_efo_response(efo_to_rows)

    with patch.object(client, "_graphql", mock_gql):
        result = await client.get_target_evidences("ENSG00000000001", ["EFO_A"])

    assert len(result["EFO_A"]) == 1
    assert result["EFO_A"][0].disease_id == "EFO_A"
