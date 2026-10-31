"""Shared fixtures for mechanism-graph unit tests: a fixed run and a stub Open Targets client."""

import pytest

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.graph.model_graph import (
    CandidateExplanation,
    GraphTarget,
    MechanismGraph,
    TargetLink,
)
from indication_scout.models.model_open_targets import (
    Association,
    MechanismOfAction,
    Pathway,
)


class StubOpenTargetsClient:
    """Returns fixed associations, pathways, and disease IDs; no network."""

    def __init__(self) -> None:
        self.associations = {
            "ENSG_A": [
                Association(
                    disease_id="EFO_1",
                    disease_name="disease one",
                    overall_score=0.5,
                    datatype_scores={"clinical": 0.4, "literature": 0.3},
                    therapeutic_areas=["metabolic disease"],
                ),
                Association(
                    disease_id="EFO_9",
                    disease_name="not a candidate",
                    overall_score=0.9,
                    datatype_scores={"clinical": 0.9},
                    therapeutic_areas=["immune system disease"],
                ),
            ],
            "ENSG_B": [
                Association(
                    disease_id="EFO_1",
                    disease_name="disease one",
                    overall_score=0.8,
                    datatype_scores={"genetic_association": 0.7},
                    therapeutic_areas=["metabolic disease"],
                ),
                Association(
                    disease_id="EFO_2",
                    disease_name="disease two official",
                    overall_score=0.6,
                    datatype_scores={"literature": 0.6, "animal_model": 0.2},
                    therapeutic_areas=["cardiovascular disease", "metabolic disease"],
                ),
            ],
            "ENSG_C": [],
        }
        self.pathways = {
            "ENSG_A": [
                Pathway(
                    pathway_id="R-1",
                    pathway_name="Pathway one",
                    top_level_pathway="Metabolism",
                ),
                Pathway(
                    pathway_id="R-2",
                    pathway_name="Pathway two",
                    top_level_pathway="Metabolism",
                ),
            ],
            "ENSG_B": [
                Pathway(
                    pathway_id="R-3",
                    pathway_name="Pathway three",
                    top_level_pathway="Signal Transduction",
                ),
            ],
            "ENSG_C": [
                Pathway(
                    pathway_id="R-4",
                    pathway_name="Pathway four",
                    top_level_pathway="Disease",
                ),
            ],
        }

    async def get_target_data_associations(self, target_id: str) -> list[Association]:
        return self.associations[target_id]

    async def get_target_data_pathways(self, target_id: str) -> list[Pathway]:
        return self.pathways[target_id]


@pytest.fixture
def stub_client() -> StubOpenTargetsClient:
    return StubOpenTargetsClient()


@pytest.fixture
def run_output() -> SupervisorOutput:
    """Three targets (TC links to nothing), three candidates (one linked twice, one once, one unresolved)."""
    return SupervisorOutput(
        drug_name="drugx",
        mechanism=MechanismOutput(
            drug_targets={"TA": "ENSG_A", "TB": "ENSG_B", "TC": "ENSG_C"},
            mechanisms_of_action=[
                MechanismOfAction(
                    mechanism_of_action="TA inhibitor",
                    action_type="INHIBITOR",
                    target_ids=["ENSG_A"],
                    target_symbols=["TA"],
                ),
                MechanismOfAction(
                    mechanism_of_action="TB and TC agonist",
                    action_type="AGONIST",
                    target_ids=["ENSG_B", "ENSG_C"],
                    target_symbols=["TB", "TC"],
                ),
            ],
        ),
        disease_findings=[
            CandidateFindings(disease="disease one", disease_id="EFO_1"),
            CandidateFindings(disease="disease two", disease_id="EFO_2"),
            CandidateFindings(disease="disease three", disease_id=None),
        ],
        top_diseases=["disease one", "disease two", "disease three"],
    )


@pytest.fixture
def expected_graph() -> MechanismGraph:
    """The graph build_graph produces for run_output and the stub client."""
    return MechanismGraph(
        drug_name="drugx",
        targets=[
            GraphTarget(
                symbol="TA",
                target_id="ENSG_A",
                action_types=["INHIBITOR"],
                pathways=[
                    Pathway(
                        pathway_id="R-1",
                        pathway_name="Pathway one",
                        top_level_pathway="Metabolism",
                    ),
                    Pathway(
                        pathway_id="R-2",
                        pathway_name="Pathway two",
                        top_level_pathway="Metabolism",
                    ),
                ],
            ),
            GraphTarget(
                symbol="TB",
                target_id="ENSG_B",
                action_types=["AGONIST"],
                pathways=[
                    Pathway(
                        pathway_id="R-3",
                        pathway_name="Pathway three",
                        top_level_pathway="Signal Transduction",
                    ),
                ],
            ),
        ],
        candidates=[
            CandidateExplanation(
                candidate="disease one",
                disease_id="EFO_1",
                disease_name="disease one",
                therapeutic_areas=["metabolic disease"],
                links=[
                    TargetLink(
                        target_symbol="TB",
                        overall_score=0.8,
                        datatype_scores={"genetic_association": 0.7},
                    ),
                    TargetLink(
                        target_symbol="TA",
                        overall_score=0.5,
                        datatype_scores={"clinical": 0.4, "literature": 0.3},
                    ),
                ],
            ),
            CandidateExplanation(
                candidate="disease two",
                disease_id="EFO_2",
                disease_name="disease two official",
                therapeutic_areas=["cardiovascular disease", "metabolic disease"],
                links=[
                    TargetLink(
                        target_symbol="TB",
                        overall_score=0.6,
                        datatype_scores={"literature": 0.6, "animal_model": 0.2},
                    ),
                ],
            ),
            CandidateExplanation(
                candidate="disease three",
                disease_id=None,
                disease_name="",
                therapeutic_areas=[],
                links=[],
            ),
        ],
    )
