"""Build the mechanism graph for a finished run from Open Targets data.

Candidate disease IDs are preserved by the supervisor from candidate discovery and joined against the associations of
every drug target. Names are display labels only and are never used as ontology identity joins.
"""

import logging

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.graph.model_graph import (
    CandidateExplanation,
    GraphTarget,
    MechanismGraph,
    TargetLink,
)
from indication_scout.models.model_open_targets import Association

logger = logging.getLogger(__name__)


async def build_graph(
    output: SupervisorOutput, client: OpenTargetsClient
) -> MechanismGraph | None:
    """Return the drug → target → candidate graph for ``output``, or None when the run has no mechanism block.

    Only associations above the client's configured minimum score count as links. Pathways are kept only for targets
    that link to at least one candidate. Raises DataSourceError from the client.
    """
    mechanism = output.mechanism
    if mechanism is None:
        return None

    action_types: dict[str, list[str]] = {}
    for moa in mechanism.mechanisms_of_action:
        for symbol in moa.target_symbols:
            types = action_types.setdefault(symbol, [])
            if moa.action_type and moa.action_type not in types:
                types.append(moa.action_type)

    # disease_id -> [(target symbol, overall score, association)] across every drug target. The client already drops
    # associations without a score; the check here only narrows the type.
    by_disease: dict[str, list[tuple[str, float, Association]]] = {}
    for symbol, target_id in mechanism.drug_targets.items():
        for assoc in await client.get_target_data_associations(target_id):
            if assoc.overall_score is not None:
                by_disease.setdefault(assoc.disease_id, []).append(
                    (symbol, assoc.overall_score, assoc)
                )

    disease_ids = {
        finding.disease.lower().strip(): finding.disease_id
        for finding in output.disease_findings
    }
    candidates: list[CandidateExplanation] = []
    linked_symbols: set[str] = set()
    for name in output.top_diseases:
        disease_id = disease_ids.get(name.lower().strip())
        hits = by_disease.get(disease_id, []) if disease_id else []
        hits = sorted(hits, key=lambda h: h[1], reverse=True)
        linked_symbols.update(symbol for symbol, _, _ in hits)
        candidates.append(
            CandidateExplanation(
                candidate=name,
                disease_id=disease_id,
                disease_name=hits[0][2].disease_name if hits else "",
                therapeutic_areas=hits[0][2].therapeutic_areas if hits else [],
                links=[
                    TargetLink(
                        target_symbol=symbol,
                        overall_score=score,
                        datatype_scores=assoc.datatype_scores,
                    )
                    for symbol, score, assoc in hits
                ],
            )
        )
        if not hits:
            logger.info(
                "[GRAPH] %s: no drug target linked to %r (disease_id=%s)",
                output.drug_name,
                name,
                disease_id,
            )

    targets = [
        GraphTarget(
            symbol=symbol,
            target_id=target_id,
            action_types=action_types.get(symbol, []),
            pathways=await client.get_target_data_pathways(target_id),
        )
        for symbol, target_id in mechanism.drug_targets.items()
        if symbol in linked_symbols
    ]
    return MechanismGraph(
        drug_name=output.drug_name, targets=targets, candidates=candidates
    )
