"""Assemble OT data into rows for mechanism_candidates.select_top_candidates.

Fetches a single target's data and per-(target, disease) evidences. Kept separate from mechanism_candidates.py so
the classifier has no OT / I/O dependencies.
"""

from datetime import date

from indication_scout.agents.mechanism.ot_score import recompute_overall
from indication_scout.constants import OT_LEAKY_DATASOURCES
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import Association


def _ranking_score(association: Association, date_before: date | None) -> float:
    """Score used to rank an association: OT's overall_score in production; in holdout, the score recomputed with
    leaky datasources (clinical_precedence) dropped so post-cutoff data cannot influence ranking.
    """
    if date_before is None:
        return association.overall_score or 0.0
    return recompute_overall(
        association.datasource_scores, exclude=OT_LEAKY_DATASOURCES
    )


async def build_candidate_rows(
    ot_client: OpenTargetsClient,
    target_id: str,
    action_types: set[str],
    top_n: int,
    date_before: date | None = None,
) -> list[dict]:
    """Fetch a target's top-N associations + per-pair evidences and return row dicts for select_top_candidates.

    Which associations survive the top_n cut is decided by `ranking_score` (see _ranking_score).

    Row keys: target_symbol, action_types, disease_name, overall_score, ranking_score, evidences,
    disease_description, target_function.
    """
    target = await ot_client.get_target_data(target_id)
    target_function = (
        target.function_descriptions[0] if target.function_descriptions else ""
    )
    top = sorted(
        target.associations,
        key=lambda a: _ranking_score(a, date_before),
        reverse=True,
    )[:top_n]
    efo_ids = [a.disease_id for a in top if a.disease_id]
    ev_map = await ot_client.get_target_evidences(target_id, efo_ids)
    return [
        {
            "target_symbol": target.symbol,
            "action_types": action_types,
            "disease_name": a.disease_name,
            "disease_id": a.disease_id,
            "overall_score": a.overall_score,
            "ranking_score": _ranking_score(a, date_before),
            "evidences": ev_map.get(a.disease_id, []),
            "disease_description": a.disease_description,
            "target_function": target_function,
        }
        for a in top
    ]
