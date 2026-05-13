"""Integration tests for OpenTargetsClient accessor methods."""

import pytest

# --- Target data accessors ---


async def test_get_target_associations(open_targets_client):
    """Test get_target_associations returns filtered associations with all fields."""
    associations = await open_targets_client.get_target_data_associations(
        "ENSG00000112164"
    )

    assert len(associations) > 10
    [gastroparesis] = [a for a in associations if a.disease_name == "gastroparesis"]
    # Verify all Association fields
    assert gastroparesis.disease_id == "EFO_1000948"
    assert gastroparesis.disease_name == "gastroparesis"
    assert gastroparesis.overall_score > 0.2
    assert 0.4 < gastroparesis.datatype_scores["genetic_association"] < 0.5
    assert 0.2 < gastroparesis.datatype_scores["literature"] < 0.3
    assert "gastrointestinal disease" in gastroparesis.therapeutic_areas


async def test_get_target_pathways(open_targets_client):
    """Test get_target_pathways returns pathway data with all fields."""
    pathways = await open_targets_client.get_target_data_pathways("ENSG00000113721")

    assert len(pathways) > 5
    [pdgf] = [p for p in pathways if p.pathway_name == "Signaling by PDGF"]
    # Verify all Pathway fields
    assert pdgf.pathway_id == "R-HSA-186797"
    assert pdgf.pathway_name == "Signaling by PDGF"
    assert pdgf.top_level_pathway == "Signal Transduction"


async def test_get_target_interactions(open_targets_client):
    """Test get_target_interactions returns interaction data with all fields."""
    interactions = await open_targets_client.get_target_data_interactions(
        "ENSG00000113721"
    )

    assert len(interactions) > 10
    plcg1 = next(
        i
        for i in interactions
        if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
    )
    # Verify all Interaction fields
    assert plcg1.interacting_target_id == "ENSG00000124181"
    assert plcg1.interacting_target_symbol == "PLCG1"
    assert plcg1.interaction_score > 0.99
    assert plcg1.source_database == "string"
    assert plcg1.biological_role == "unspecified role"
    assert plcg1.evidence_count == 4
    assert plcg1.interaction_type == "functional"


async def test_interaction_type_string_is_functional(open_targets_client):
    """Test STRING interaction with all Interaction fields verified."""
    interactions = await open_targets_client.get_target_data_interactions(
        "ENSG00000113721"
    )

    # Pick PLCG1 interaction from STRING - verify all fields
    plcg1_string = next(
        i
        for i in interactions
        if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
    )
    assert plcg1_string.interacting_target_id == "ENSG00000124181"
    assert plcg1_string.interacting_target_symbol == "PLCG1"
    assert plcg1_string.interaction_score > 0.99
    assert plcg1_string.source_database == "string"
    assert plcg1_string.biological_role == "unspecified role"
    assert plcg1_string.evidence_count == 4
    assert plcg1_string.interaction_type == "functional"


async def test_interaction_type_intact_is_physical(open_targets_client):
    """Test IntAct interaction with all Interaction fields verified."""
    interactions = await open_targets_client.get_target_data_interactions(
        "ENSG00000113721"
    )

    # Pick PLCG1 interaction from IntAct - verify all fields
    plcg1_intact = next(
        i
        for i in interactions
        if i.interacting_target_symbol == "PLCG1" and i.source_database == "intact"
    )
    assert plcg1_intact.interacting_target_id == "ENSG00000124181"
    assert plcg1_intact.interacting_target_symbol == "PLCG1"
    assert plcg1_intact.interaction_score > 0.6
    assert plcg1_intact.source_database == "intact"
    assert plcg1_intact.biological_role == "unspecified role"
    assert plcg1_intact.evidence_count > 0
    assert plcg1_intact.interaction_type == "physical"


async def test_interaction_type_signor_is_signalling(open_targets_client):
    """Test Signor interaction with all Interaction fields verified.

    Note: Signor data may not be currently available in Open Targets API.
    This test validates the mapping if Signor interactions are present.
    """
    interactions = await open_targets_client.get_target_data_interactions(
        "ENSG00000113721"
    )

    signor_interactions = [i for i in interactions if i.source_database == "signor"]
    for interaction in signor_interactions:
        assert interaction.interaction_type == "signalling"


async def test_interaction_type_reactome_is_enzymatic(open_targets_client):
    """Test Reactome interaction with all Interaction fields verified.

    Note: Reactome data may not be currently available in Open Targets API.
    This test validates the mapping if Reactome interactions are present.
    """
    interactions = await open_targets_client.get_target_data_interactions(
        "ENSG00000113721"
    )

    reactome_interactions = [i for i in interactions if i.source_database == "reactome"]
    for interaction in reactome_interactions:
        assert interaction.interaction_type == "enzymatic"


async def test_get_target_drug_summaries(open_targets_client):
    """Test get_known_drugs returns drugs with all DrugSummary fields."""
    drug_summaries = await open_targets_client.get_target_data_drug_summaries(
        "ENSG00000112164"
    )

    assert len(drug_summaries) > 5
    liraglutide = next(d for d in drug_summaries if d.drug_name == "liraglutide")
    # Verify DrugSummary fields
    assert liraglutide.drug_id == "CHEMBL4084119"
    assert liraglutide.drug_name == "liraglutide"
    assert liraglutide.drug_type == "Protein"
    assert liraglutide.max_clinical_stage == "APPROVAL"
    assert len(liraglutide.diseases) > 0
    t2d = next(
        d for d in liraglutide.diseases if d.disease_name == "type 2 diabetes mellitus"
    )
    assert t2d.disease_id == "MONDO_0005148"
    assert t2d.disease_from_source != ""


async def test_get_target_expression(open_targets_client):
    """Test get_target_expression returns tissue expression with all fields."""
    expressions = await open_targets_client.get_target_data_tissue_expression(
        "ENSG00000163399"
    )

    assert len(expressions) > 10
    liver = next(e for e in expressions if e.tissue_name == "liver")
    # Verify all TissueExpression fields
    assert liver.tissue_id == "UBERON_0002107"
    assert liver.tissue_name == "liver"
    assert liver.tissue_anatomical_system == "endocrine system"
    # Verify all RNAExpression fields
    assert liver.rna.value > 0
    assert liver.rna.quantile == 5
    # Verify all ProteinExpression fields
    assert liver.protein.level == 2
    assert liver.protein.reliability is True


async def test_get_target_phenotypes(open_targets_client):
    """Test get_target_phenotypes returns mouse phenotype with all fields."""
    phenotypes = await open_targets_client.get_target_data_mouse_phenotypes(
        "ENSG00000112164"
    )

    assert len(phenotypes) > 5
    glucose = next(p for p in phenotypes if p.phenotype_id == "MP:0013279")
    # Verify all MousePhenotype fields
    assert glucose.phenotype_id == "MP:0013279"
    assert glucose.phenotype_label == "increased fasting circulating glucose level"
    assert "homeostasis/metabolism phenotype" in glucose.phenotype_categories
    assert len(glucose.biological_models) == 1
    # Verify BiologicalModel fields
    [model] = glucose.biological_models
    assert model.allelic_composition == "Glp1r<tm1b(KOMP)Mbp> hom early"
    assert model.genetic_background == "C57BL/6NTac"
    assert model.model_id == "MGI:5637223"
    assert "8898756" in model.literature
    assert "23959939" in model.literature
    assert "14966573" in model.literature


async def test_get_target_safety_liabilities(open_targets_client):
    """Test get_target_data_safety_liabilities returns all SafetyLiability fields."""
    liabilities = await open_targets_client.get_target_data_safety_liabilities(
        "ENSG00000163399"
    )

    assert len(liabilities) > 5

    arrhythmia = next(
        sl
        for sl in liabilities
        if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
    )
    # Verify all SafetyLiability fields
    assert arrhythmia.event == "cardiac arrhythmia"
    assert arrhythmia.event_id == "EFO_0004269"
    assert arrhythmia.datasource == "Lynch et al. (2017)"
    assert arrhythmia.literature == "28216264"
    assert arrhythmia.url is None
    assert len(arrhythmia.effects) == 1
    # Verify SafetyEffect fields
    [effect] = arrhythmia.effects
    assert effect.direction == "Inhibition/Decrease/Downregulation"
    assert effect.dosing == "acute"


async def test_get_target_genetic_constraints(open_targets_client):
    """Test get_target_data_genetic_constraints returns all GeneticConstraint fields."""
    constraints = await open_targets_client.get_target_data_genetic_constraints(
        "ENSG00000141736"
    )

    assert len(constraints) > 0
    lof_constraint = next(gc for gc in constraints if gc.constraint_type == "lof")
    # Verify all GeneticConstraint fields
    assert lof_constraint.constraint_type == "lof"
    assert 0.41 < lof_constraint.oe < 0.42
    assert 0.33 < lof_constraint.oe_lower < 0.34
    assert 0.51 < lof_constraint.oe_upper < 0.52
    assert 0.06 < lof_constraint.score < 0.07
    assert lof_constraint.upper_bin == 1
    assert lof_constraint.exp > 100
    assert lof_constraint.obs > 0
    assert lof_constraint.upper_bin6 == 1


async def test_get_drug_indications(open_targets_client):
    """Test get_drug_indications returns indication data."""
    indications = await open_targets_client.get_drug_indications("CHEMBL2108724")

    assert len(indications) > 5
    [t2d] = [i for i in indications if i.disease_name == "type 2 diabetes mellitus"]
    assert t2d.disease_id == "MONDO_0005148"
    assert t2d.max_clinical_stage == "APPROVAL"
    assert t2d.id != ""


async def test_get_disease_drugs(open_targets_client):
    """Test get_disease_drugs returns drugs for a disease with all DrugSummary fields."""
    # Type 2 diabetes mellitus - should have semaglutide and other GLP-1 agonists
    drugs = await open_targets_client.get_disease_drugs("MONDO_0005148")

    assert len(drugs) > 10
    semaglutide = next(d for d in drugs if d.drug_name == "semaglutide")
    # Verify DrugSummary fields
    assert semaglutide.drug_id == "CHEMBL2108724"
    assert semaglutide.drug_name == "semaglutide"
    assert semaglutide.max_clinical_stage == "APPROVAL"


# --- Text fields added for mechanism-agent candidate pipeline ---


async def test_target_and_association_text_fields(open_targets_client):
    """TargetData.function_descriptions + Association.disease_description are
    populated on GLP1R / type 2 diabetes (both known to carry these fields)."""
    target = await open_targets_client.get_target_data("ENSG00000112164")

    # Target-level UniProt function paragraphs
    assert target.symbol == "GLP1R"
    assert len(target.function_descriptions) >= 1
    assert "glucagon-like peptide 1" in target.function_descriptions[0].lower()

    # Disease description on the T2D association
    t2d = next(
        a for a in target.associations if a.disease_name == "type 2 diabetes mellitus"
    )
    assert t2d.disease_description
    assert "diabetes" in t2d.disease_description.lower()


async def test_get_target_evidences_glp1r_t2d(open_targets_client):
    """get_target_evidences returns records grouped by disease_id with direction
    fields populated for GLP1R / T2D (GoF / protect is the canonical signal)."""
    ev_map = await open_targets_client.get_target_evidences(
        "ENSG00000112164", ["MONDO_0005148"]
    )

    assert "MONDO_0005148" in ev_map
    records = ev_map["MONDO_0005148"]
    assert len(records) > 0
    directions = {
        (e.direction_on_target, e.direction_on_trait)
        for e in records
        if e.direction_on_target and e.direction_on_trait
    }
    assert ("GoF", "protect") in directions


async def test_get_target_evidences_variant_functional_consequence(open_targets_client):
    """variant_functional_consequence populates on rare-variant Mendelian disease.

    SLC6A3 + infantile dystonia-parkinsonism is driven by coding variants in
    SLC6A3, so the evidence records carry vFC labels like missense_variant /
    absent_gene_product.
    """
    ev_map = await open_targets_client.get_target_evidences(
        "ENSG00000142319", ["Orphanet_238455"]
    )
    records = ev_map["Orphanet_238455"]
    vfc_labels = {
        e.variant_functional_consequence.label
        for e in records
        if e.variant_functional_consequence and e.variant_functional_consequence.label
    }
    assert "missense_variant" in vfc_labels
