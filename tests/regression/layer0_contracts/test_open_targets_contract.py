"""Open Targets client contract: the GraphQL drug and target payloads parse as recorded.

`get_drug` also enriches from ChEMBL, so the ATC assertion covers that join too.
"""

from __future__ import annotations

import pytest

from indication_scout.data_sources.open_targets import OpenTargetsClient
from tests.regression.layer0_contracts.conftest import ContractClient

pytestmark = pytest.mark.contract


async def test_get_drug_parses_metformin(contract_client: ContractClient) -> None:
    async with contract_client(OpenTargetsClient, "ot_drug") as client:
        drug = await client.get_drug("CHEMBL1431")

    assert drug.chembl_id == "CHEMBL1431"
    assert drug.drug_type == "Small molecule"
    assert drug.maximum_clinical_stage == "APPROVAL"
    assert drug.atc_classifications == ["A10BA02"]
    assert drug.adverse_events_critical_value == pytest.approx(415.1712020775594)

    assert len(drug.mechanisms_of_action) == 2
    moa = drug.mechanisms_of_action[0]
    assert (
        moa.mechanism_of_action
        == "Mitochondrial complex I (NADH dehydrogenase) inhibitor"
    )
    assert moa.action_type == "INHIBITOR"
    assert len(moa.target_ids) == 50

    assert len(drug.warnings) == 4
    warning = drug.warnings[0]
    assert warning.warning_type == "Black Box Warning"
    assert warning.toxicity_class == "respiratory toxicity"
    assert warning.country == "United States"

    assert len(drug.indications) == 248
    assert len(drug.targets) == 51
    target = drug.targets[0]
    assert target.target_id == "ENSG00000198695"
    assert target.target_symbol == "MT-ND6"
    assert target.action_type == "INHIBITOR"

    assert len(drug.adverse_events) == 28
    event = drug.adverse_events[0]
    assert event.name == "lactic acidosis"
    assert event.count == 7697
    assert event.log_likelihood_ratio == pytest.approx(27686.59554770874)


async def test_get_target_data_parses_glp1r(contract_client: ContractClient) -> None:
    async with contract_client(OpenTargetsClient, "ot_target") as client:
        target = await client.get_target_data("ENSG00000112164")

    assert target.target_id == "ENSG00000112164"
    assert target.symbol == "GLP1R"
    assert target.name == "glucagon like peptide 1 receptor"
    assert len(target.function_descriptions) == 1
    assert target.function_descriptions[0].startswith("G protein-coupled receptor")

    # Paginated: a change to the paging loop shows up as a different count.
    assert len(target.associations) == 1187
    association = target.associations[0]
    assert association.disease_id == "MONDO_0005148"
    assert association.disease_name == "type 2 diabetes mellitus"
    assert association.overall_score == pytest.approx(0.7607441891473111)
    assert association.datatype_scores["genetic_association"] == pytest.approx(
        0.7402293861355427
    )
    assert association.datasource_scores["europepmc"] == pytest.approx(
        0.9905932528443046
    )

    assert len(target.pathways) == 3
    assert target.pathways[0].pathway_id == "R-HSA-420092"
    assert target.pathways[0].pathway_name == "Glucagon-type ligand receptors"
    assert target.pathways[0].top_level_pathway == "Signal Transduction"

    assert len(target.interactions) == 200
    interaction = target.interactions[0]
    assert interaction.interacting_target_symbol == "GCG"
    assert interaction.interaction_score == pytest.approx(0.999)
    assert interaction.source_database == "string"
    assert interaction.evidence_count == 4

    assert len(target.drug_summaries) == 17
    summary = target.drug_summaries[0]
    assert summary.drug_id == "CHEMBL4518483"
    assert summary.drug_name == "danuglipron"
    assert summary.max_clinical_stage == "PHASE_2"

    assert len(target.mouse_phenotypes) == 12
    assert target.mouse_phenotypes[0].phenotype_id == "MP:0013279"
    assert (
        target.mouse_phenotypes[0].phenotype_label
        == "increased fasting circulating glucose level"
    )

    assert len(target.safety_liabilities) == 2
    assert target.safety_liabilities[0].event == "less weight gain"
    assert target.safety_liabilities[0].datasource == "ClinPGx"

    assert len(target.genetic_constraint) == 3
    constraint = target.genetic_constraint[0]
    assert constraint.constraint_type == "syn"
    assert constraint.obs == pytest.approx(185.0)
    assert constraint.oe == pytest.approx(1.0125999450683594)
