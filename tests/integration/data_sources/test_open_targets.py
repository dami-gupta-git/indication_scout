"""Integration tests for OpenTargetsopen_targets_client."""

import logging

import pytest


from indication_scout.constants import BROADENING_BLOCKLIST
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.markers import no_review

logger = logging.getLogger(__name__)


# --- get_drug ---


@no_review
async def test_sildenafil_drug_data(open_targets_client):
    """Test fetching drug data and indications for semaglutide."""
    drug = await open_targets_client.get_drug("CHEMBL2108724")
    indications = drug.indications
    match = [i for i in indications if "kidney" in i.disease_name.lower()]
    approved = [a for a in match if a.disease_id in drug.approved_disease_ids]
    logger.info(drug.indications)

@no_review
async def test_imatinib_drug_data(open_targets_client):
    """Test fetching drug data and indications for semaglutide."""
    drug = await open_targets_client.get_drug("CHEMBL941")
    indications = drug.indications
    pass

# TODO delete
@no_review
async def test_single_drug_data(open_targets_client):
    """Test fetching drug data and indications for semaglutide."""
    drug = await open_targets_client.get_drug("CHEMBL894")
    indications = drug.indications
    assert drug.atc_classifications == ["N06AX12"]


async def test_bupropion_drug_data(open_targets_client):
    """Test bupropion DrugData fields including ChEMBL trade names from salt forms."""
    drug = await open_targets_client.get_drug("CHEMBL894")

    assert drug.atc_classifications == ["N06AX12"]


async def test_semaglutide_drug_data(open_targets_client):
    """Test all semaglutide DrugData fields: top-level, targets, MoA, indications."""
    drug = await open_targets_client.get_drug("CHEMBL2108724")

    # DrugData top-level fields
    assert drug.chembl_id == "CHEMBL2108724"
    assert drug.drug_type == "Protein"
    assert drug.maximum_clinical_stage == "APPROVAL"
    assert len(drug.indications) > 5
    assert len(drug.targets) > 0
    assert len(drug.adverse_events) > 5
    assert drug.adverse_events_critical_value is not None
    assert len(drug.warnings) == 1
    assert drug.atc_classifications == ["A10BJ06"]

    # DrugTarget — GLP1R
    [glp1r] = [t for t in drug.targets if t.target_symbol == "GLP1R"]
    assert glp1r.target_id == "ENSG00000112164"
    assert glp1r.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
    assert glp1r.action_type == "AGONIST"

    # MechanismOfAction
    assert len(drug.mechanisms_of_action) == 1
    moa = drug.mechanisms_of_action[0]
    assert moa.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
    assert moa.action_type == "AGONIST"
    assert "ENSG00000112164" in moa.target_ids
    assert "GLP1R" in moa.target_symbols

    # Indication — type 2 diabetes mellitus
    [t2d] = [
        i for i in drug.indications if i.disease_name == "type 2 diabetes mellitus"
    ]
    assert t2d.disease_id == "MONDO_0005148"
    assert t2d.max_clinical_stage == "APPROVAL"
    assert t2d.id != ""


async def test_trastuzumab_drug_data(open_targets_client):
    """Test trastuzumab: ATC classification and adverse events."""
    drug = await open_targets_client.get_drug("CHEMBL1201585")

    assert drug.atc_classifications == ["L01FD01"]

    # AdverseEvent fields
    adverse_event = next(
        ae for ae in drug.adverse_events if ae.name == "ejection fraction decreased"
    )
    assert adverse_event.meddra_code == "10050528"
    assert adverse_event.count > 0
    assert adverse_event.log_likelihood_ratio > 0


async def test_rofecoxib_drug_data(open_targets_client):
    """Test rofecoxib (Vioxx): ATC, and DrugWarning with all fields."""
    drug = await open_targets_client.get_drug("CHEMBL122")

    assert drug.atc_classifications == ["M01AH02"]
    assert len(drug.warnings) > 5

    # DrugWarning — cardiotoxicity withdrawal
    warning = next(
        w
        for w in drug.warnings
        if w.toxicity_class == "cardiotoxicity"
        and w.efo_id == "EFO:0000612"
        and "serious cardiovascular events" in (w.description or "")
    )
    assert warning.warning_type == "Withdrawn"
    assert (
        warning.description
        == "Increased risk of serious cardiovascular events, such as heart attacks and strokes"
    )
    assert warning.country == "Worldwide"
    assert warning.year == 2004


async def test_metformin_drug_data(open_targets_client):
    """Test metformin basic DrugData fields."""
    drug = await open_targets_client.get_drug("CHEMBL1431")

    assert drug.chembl_id == "CHEMBL1431"
    assert drug.drug_type == "Small molecule"
    assert drug.maximum_clinical_stage == "APPROVAL"
    assert drug.atc_classifications == ["A10BA02"]


# --- get_drug error handling ---


@pytest.mark.parametrize(
    "bad_name",
    ["xyzzy_not_a_real_drug_12345", "", "!!!@@@###$$$"],
)
async def test_get_drug_invalid_input_raises_error(open_targets_client, bad_name):
    """Invalid drug names raise DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_drug(bad_name)

    assert exc_info.value.source == "open_targets"


# --- get_disease_synonyms ---


# TODO delete
@no_review
async def test_get_disease_synonyms_nep(open_targets_client):

    result = await open_targets_client.get_disease_synonyms(
        "type 2 diabetes nephropathy"
    )
    assert result


async def test_get_disease_synonyms(open_targets_client):
    """Test fetching disease synonyms for type 2 diabetes mellitus."""
    result = await open_targets_client.get_disease_synonyms("type 2 diabetes mellitus")

    assert result.disease_id == "MONDO_0005148"
    assert result.disease_name == "type 2 diabetes mellitus"
    assert result.parent_names == ["diabetes mellitus"]
    assert len(result.exact) == 25
    assert len(result.related) == 9
    assert len(result.narrow) == 1
    assert "T2DM" in result.exact
    assert "NIDDM" in result.exact
    assert "type 2 diabetes" in result.exact
    assert "maturity-onset diabetes" in result.related
    assert "diabetes mellitus, noninsulin-dependent, 2" in result.narrow


async def test_get_disease_synonyms_nonexistent(open_targets_client):
    """Test that a nonexistent disease name raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_disease_synonyms("xyzzy_not_a_real_disease_12345")

    assert exc_info.value.source == "open_targets"
    assert "No disease found" in str(exc_info.value)


# --- get_target_data error handling ---


@pytest.mark.parametrize(
    "bad_id",
    ["ENSG99999999999", "not_an_ensembl_id", ""],
)
async def test_get_target_data_invalid_input_raises_error(open_targets_client, bad_id):
    """Invalid target IDs raise DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_target_data(bad_id)

    assert exc_info.value.source == "open_targets"


# --- get_target_data ---


async def test_glp1r_target_data(open_targets_client):
    """Test GLP1R TargetData: associations, drug summaries, and mouse phenotypes."""
    target = await open_targets_client.get_target_data("ENSG00000112164")

    # Association — gastroparesis
    assert len(target.associations) > 10
    [assoc] = [a for a in target.associations if a.disease_name == "gastroparesis"]
    assert assoc.disease_id.startswith("EFO_") or assoc.disease_id.startswith("MONDO_")
    assert assoc.overall_score > 0.2
    assert 0.4 < assoc.datatype_scores["genetic_association"] < 0.5
    assert 0.2 < assoc.datatype_scores["literature"] < 0.3
    assert "gastrointestinal disease" in assoc.therapeutic_areas

    # DrugSummary — liraglutide
    liraglutide = next(d for d in target.drug_summaries if d.drug_name == "liraglutide")
    assert liraglutide.drug_id == "CHEMBL4084119"
    assert liraglutide.max_clinical_stage == "APPROVAL"
    assert len(liraglutide.diseases) > 0
    t2d = next(
        d for d in liraglutide.diseases if d.disease_name == "type 2 diabetes mellitus"
    )
    assert t2d.disease_id == "MONDO_0005148"


async def test_pdgfrb_target_data(open_targets_client):
    """Test PDGFRB TargetData: pathways and interactions."""
    target = await open_targets_client.get_target_data("ENSG00000113721")

    # Pathway — Signaling by PDGF
    [pathway] = [p for p in target.pathways if p.pathway_name == "Signaling by PDGF"]
    assert pathway.pathway_id == "R-HSA-186797"
    assert pathway.top_level_pathway == "Signal Transduction"

    # Interaction — PLCG1 via STRING
    interaction = next(
        i
        for i in target.interactions
        if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
    )
    assert interaction.interacting_target_id == "ENSG00000124181"
    assert interaction.interaction_score > 0.99
    assert interaction.biological_role == "unspecified role"
    assert interaction.evidence_count == 4


async def test_atp1a1_target_data(open_targets_client):
    """Test ATP1A1 TargetData: tissue expression and safety liabilities."""
    target = await open_targets_client.get_target_data("ENSG00000163399")

    # TissueExpression — liver
    expression = next(e for e in target.expressions if e.tissue_name == "liver")
    assert expression.tissue_id == "UBERON_0002107"
    assert expression.tissue_anatomical_system == "endocrine system"
    assert expression.rna.value > 0
    assert expression.rna.quantile == 5
    assert expression.protein.level == 2
    assert expression.protein.reliability is True

    # SafetyLiability — cardiac arrhythmia
    assert len(target.safety_liabilities) > 5
    liability = next(
        sl
        for sl in target.safety_liabilities
        if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
    )
    assert liability.datasource == "Lynch et al. (2017)"
    assert liability.literature == "28216264"
    assert liability.url is None
    assert len(liability.effects) == 1
    [effect] = liability.effects
    assert effect.direction == "Inhibition/Decrease/Downregulation"
    assert effect.dosing == "acute"


async def test_glp1r_target_mouse_phenotype_and_erbb2_constraint(open_targets_client):
    """Test MousePhenotype (GLP1R) and GeneticConstraint (ERBB2) — two small single-field tests combined."""
    # GLP1R — MousePhenotype + BiologicalModel
    glp1r = await open_targets_client.get_target_data("ENSG00000112164")
    phenotype = next(
        p for p in glp1r.mouse_phenotypes if p.phenotype_id == "MP:0013279"
    )
    assert phenotype.phenotype_label == "increased fasting circulating glucose level"
    assert "homeostasis/metabolism phenotype" in phenotype.phenotype_categories
    assert len(phenotype.biological_models) == 1
    [model] = phenotype.biological_models
    assert model.allelic_composition == "Glp1r<tm1b(KOMP)Mbp> hom early"
    assert model.genetic_background == "C57BL/6NTac"
    assert model.model_id == "MGI:5637223"
    assert "8898756" in model.literature

    # ERBB2 — GeneticConstraint (lof)
    erbb2 = await open_targets_client.get_target_data("ENSG00000141736")
    lof_constraint = next(
        gc for gc in erbb2.genetic_constraint if gc.constraint_type == "lof"
    )
    assert 0.41 < lof_constraint.oe < 0.42
    assert 0.33 < lof_constraint.oe_lower < 0.34
    assert 0.51 < lof_constraint.oe_upper < 0.52
    assert 0.06 < lof_constraint.score < 0.07
    assert lof_constraint.upper_bin == 1
    assert lof_constraint.exp > 100
    assert lof_constraint.obs > 0
    assert lof_constraint.upper_bin6 == 1


async def test_drug_target_competitors_semaglutide(open_targets_client):
    """Test get_drug_target_competitors returns DrugSummary lists keyed by target symbol."""
    result = await open_targets_client.get_drug_target_competitors("CHEMBL2108724")

    # Semaglutide targets GLP1R — must be present
    assert "GLP1R" in result
    assert len(result["GLP1R"]) > 5

    # All values must be lists of DrugSummary; check the map has no empty lists
    for symbol, summaries in result.items():
        assert isinstance(summaries, list)
        assert len(summaries) > 0

    # Spot-check a known GLP1R competitor: LIRAGLUTIDE
    liraglutide = next(d for d in result["GLP1R"] if d.drug_name == "liraglutide")
    assert liraglutide.drug_id == "CHEMBL4084119"
    assert liraglutide.drug_name == "liraglutide"
    assert liraglutide.max_clinical_stage == "APPROVAL"
    assert len(liraglutide.diseases) > 0


# --- get_rich_drug_data ---


async def test_get_rich_drug_data_semaglutide(open_targets_client):
    """Test RichDrugData for semaglutide: all DrugData fields, all TargetData fields for GLP1R."""
    result = await open_targets_client.get_rich_drug_data("CHEMBL2108724")

    # --- RichDrugData structure ---
    assert len(result.targets) == len(result.drug.targets)

    # --- DrugData fields ---
    drug = result.drug
    assert drug.chembl_id == "CHEMBL2108724"
    assert drug.drug_type == "Protein"
    assert drug.maximum_clinical_stage == "APPROVAL"
    assert len(drug.indications) > 5
    assert len(drug.targets) > 0
    assert len(drug.adverse_events) > 5
    assert drug.adverse_events_critical_value is not None
    assert len(drug.warnings) == 1

    # DrugTarget — GLP1R
    glp1r_target = next(t for t in drug.targets if t.target_symbol == "GLP1R")
    assert glp1r_target.target_id == "ENSG00000112164"
    assert glp1r_target.target_symbol == "GLP1R"
    assert (
        glp1r_target.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
    )
    assert glp1r_target.action_type == "AGONIST"

    # Indication — type 2 diabetes mellitus
    t2d = next(
        i for i in drug.indications if i.disease_name == "type 2 diabetes mellitus"
    )
    assert t2d.disease_id == "MONDO_0005148"
    assert t2d.disease_name == "type 2 diabetes mellitus"
    assert t2d.max_clinical_stage == "APPROVAL"
    assert t2d.id != ""

    # DrugWarning — single warning for semaglutide
    [warning] = drug.warnings
    assert warning.warning_type == "Black Box Warning"

    # AdverseEvent — pick a known one
    nausea = next(ae for ae in drug.adverse_events if ae.name == "nausea")
    assert nausea.name == "nausea"
    assert nausea.count > 0
    assert nausea.log_likelihood_ratio > 300

    # --- GLP1R TargetData — all fields ---
    glp1r = next(t for t in result.targets if t.target_id == "ENSG00000112164")
    assert glp1r.target_id == "ENSG00000112164"
    assert glp1r.symbol == "GLP1R"
    assert glp1r.name == "glucagon like peptide 1 receptor"
    assert len(glp1r.associations) > 10
    assert len(glp1r.drug_summaries) > 5
    assert len(glp1r.expressions) > 0
    assert len(glp1r.mouse_phenotypes) > 5
    assert len(glp1r.pathways) > 0
    assert len(glp1r.interactions) > 0

    # Association — gastroparesis
    gastroparesis = next(
        a for a in glp1r.associations if a.disease_name == "gastroparesis"
    )
    assert gastroparesis.disease_id == "EFO_1000948"
    assert gastroparesis.disease_name == "gastroparesis"
    assert gastroparesis.overall_score > 0.2
    assert 0.4 < gastroparesis.datatype_scores["genetic_association"] < 0.5
    assert 0.2 < gastroparesis.datatype_scores["literature"] < 0.3
    assert "gastrointestinal disease" in gastroparesis.therapeutic_areas

    # DrugSummary — semaglutide on GLP1R
    sema_summary = next(d for d in glp1r.drug_summaries if d.drug_name == "semaglutide")
    assert sema_summary.drug_id == "CHEMBL2108724"
    assert sema_summary.drug_name == "semaglutide"
    assert sema_summary.drug_type == "Protein"
    assert sema_summary.max_clinical_stage == "APPROVAL"
    assert len(sema_summary.diseases) > 0
    t2d_disease = next(
        d for d in sema_summary.diseases if d.disease_name == "type 2 diabetes mellitus"
    )
    assert t2d_disease.disease_id == "MONDO_0005148"
    assert t2d_disease.disease_from_source != ""

    # MousePhenotype — increased fasting circulating glucose level
    glucose_phenotype = next(
        p for p in glp1r.mouse_phenotypes if p.phenotype_id == "MP:0013279"
    )
    assert glucose_phenotype.phenotype_id == "MP:0013279"
    assert (
        glucose_phenotype.phenotype_label
        == "increased fasting circulating glucose level"
    )
    assert "homeostasis/metabolism phenotype" in glucose_phenotype.phenotype_categories
    assert len(glucose_phenotype.biological_models) == 1
    [bio_model] = glucose_phenotype.biological_models
    assert bio_model.allelic_composition == "Glp1r<tm1b(KOMP)Mbp> hom early"
    assert bio_model.genetic_background == "C57BL/6NTac"
    assert bio_model.model_id == "MGI:5637223"
    assert len(bio_model.literature) > 0

    # Interaction — pick a STRING interaction
    string_interactions = [
        i for i in glp1r.interactions if i.source_database == "string"
    ]
    assert len(string_interactions) > 0
    assert all(i.interaction_type == "functional" for i in string_interactions)
    assert all(i.evidence_count > 0 for i in string_interactions)

    # Pathway — R-HSA-420092
    [pathway] = [p for p in glp1r.pathways if p.pathway_id == "R-HSA-420092"]
    assert pathway.pathway_id == "R-HSA-420092"
    assert pathway.pathway_name == "Glucagon-type ligand receptors"
    assert pathway.top_level_pathway == "Signal Transduction"


async def test_get_rich_drug_data_null_interactions(open_targets_client):
    """coerce_nones must convert null list fields to [] across all targets.

    metformin's PRKAA1 target is known to return null for 'interactions' from
    the Open Targets API. This guards against AttributeError on any target,
    regardless of which symbols the API returns.
    """
    result = await open_targets_client.get_rich_drug_data("CHEMBL1431")

    assert len(result.targets) > 0
    for t in result.targets:
        assert isinstance(
            t.interactions, list
        ), f"{t.symbol} interactions is not a list"
        assert isinstance(
            t.associations, list
        ), f"{t.symbol} associations is not a list"
        assert isinstance(
            t.drug_summaries, list
        ), f"{t.symbol} drug_summaries is not a list"


async def test_get_drug_competitors_bupropion(open_targets_client):
    """get_drug_competitors returns candidate diseases with correct competitor drugs.

    Bupropion acts on monoamine transporters (SLC6A2, SLC6A3, SLC6A4). Its sibling
    drugs treat fatigue, fibromyalgia, pain, and drug dependence — all verified live
    on 2026-03-10. Asserts:
    - result is a non-empty dict
    - known candidate diseases are present with expected sibling drugs
    """
    result = await open_targets_client.get_drug_competitors("CHEMBL894")
    diseases = result["diseases"]

    assert isinstance(diseases, dict)
    assert len(diseases) >= 1

    # Fatigue: armodafinil, methylphenidate, modafinil all share monoamine targets
    assert "fatigue" in diseases
    assert {"armodafinil", "methylphenidate", "modafinil"}.issubset(diseases["fatigue"])

    # Fibromyalgia: duloxetine, milnacipran, levomilnacipran (SNRI class on same targets)
    assert "fibromyalgia" in diseases
    assert {"duloxetine", "milnacipran", "levomilnacipran"}.issubset(
        diseases["fibromyalgia"]
    )


async def test_empagliflozin_candidates(open_targets_client):
    result = await open_targets_client.get_drug_competitors("CHEMBL2107830")
    diseases = set(result["diseases"].keys())
    # Should have candidate diseases
    assert len(diseases) > 0
    # Should be subtracted (APPROVAL stage)
    assert "heart failure" not in diseases
    assert "type 2 diabetes mellitus" not in diseases
