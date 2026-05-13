"""Assemble OT data into rows for mechanism_candidates.select_top_candidates.

Data-layer glue between OpenTargetsClient and the pure-function classifier. Fetches a single target's
data + per-(target, disease) evidences and returns dict rows shaped for select_top_candidates.

Kept separate from mechanism_candidates.py so the classifier module has no OT / I/O dependencies.
"""

from indication_scout.data_sources.open_targets import OpenTargetsClient


async def build_candidate_rows(
    ot_client: OpenTargetsClient,
    target_id: str,
    action_types: set[str],
    top_n: int,
) -> list[dict]:
    """Fetch a target's top-N associations + per-pair evidences and return row dicts shaped for
    select_top_candidates.

    The row contract (keys):
        target_symbol, action_types, disease_name, overall_score, evidences, disease_description,
        target_function
    """
    target = await ot_client.get_target_data(target_id)
    target_function = (
        target.function_descriptions[0] if target.function_descriptions else ""
    )
    top = sorted(
        target.associations,
        key=lambda a: a.overall_score or 0.0,
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
            "evidences": ev_map.get(a.disease_id, []),
            "disease_description": a.disease_description,
            "target_function": target_function,
        }
        for a in top
    ]
