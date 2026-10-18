"""Pure-function helpers for selecting mechanism-agent repurposing candidates (no OT calls, async, or agent coupling).

Classifies each (target, disease) pair as POSITIVE when the drug's action direction aligns with the disease
mechanism, aggregated from directionOnTarget + directionOnTrait across evidence records. Drops FDA-approved
diseases supplied by the caller and returns the top N POSITIVE diseases by overall score, one row per disease.
"""

import re
from collections import Counter
from collections.abc import Iterable

from indication_scout.agents.mechanism.mechanism_output import MechanismCandidate
from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    GOF_ACTION_TYPES,
    LOF_ACTION_TYPES,
)

# Minimum fraction of direction-labeled evidence records that must agree on a (dirT, dirTrait) pair to count as the
# consensus. Majority voting tolerates the outlier records OT often has (e.g. 146 GoF/protect + 1 LoF/risk).
_MAJORITY_THRESHOLD: float = 0.8


def aggregate_directions(
    records: Iterable,
    min_fraction: float = _MAJORITY_THRESHOLD,
) -> tuple[set[str], set[str]]:
    """Majority-vote aggregation of direction fields across evidence records.

    Returns singleton sets for the most common (directionOnTarget, directionOnTrait) pair iff it covers at least
    `min_fraction` of records where BOTH fields are populated; otherwise empty sets (UNKNOWN downstream).
    Records with a missing field can't vote.

    Accepts any iterable of objects with `direction_on_target` and `direction_on_trait` attributes.
    """
    pairs = [
        (r.direction_on_target, r.direction_on_trait)
        for r in records
        if getattr(r, "direction_on_target", None)
        and getattr(r, "direction_on_trait", None)
    ]
    if not pairs:
        return set(), set()
    counts = Counter(pairs)
    (top_dir, top_trait), top_count = counts.most_common(1)[0]
    if top_count / len(pairs) < min_fraction:
        return set(), set()
    return {top_dir}, {top_trait}


def classify_positive(
    action_types: set[str],
    directions_on_target: set[str],
    directions_on_trait: set[str],
) -> bool:
    """Return True iff the drug's action opposes the disease mechanism.

    False if drug action or disease direction is unknown or inconclusive.
    """
    drug_dir = _drug_direction(action_types)
    if drug_dir == "unknown":
        return False

    disease_dir = _disease_direction(directions_on_target, directions_on_trait)
    if disease_dir in ("unknown", "inconclusive"):
        return False

    # Disease needs the opposite direction of what causes it.
    if disease_dir == "LoF-driven" and drug_dir == "GoF":
        return True
    if disease_dir == "GoF-driven" and drug_dir == "LoF":
        return True
    return False


def clean_function_description(text: str) -> str:
    """Strip UniProt citation/ECO noise from a function-description string.

    Removes `(PubMed:…)` citations, `{ECO:…}` blocks and `(By similarity)` / `(Probable)` qualifiers, then tidies
    whitespace and punctuation.
    """
    if not text:
        return ""
    text = re.sub(r"\{ECO:[^}]*\}", "", text)
    text = re.sub(r"\(\s*PubMed:[^)]*\)", "", text)
    text = re.sub(r"\(\s*By similarity\s*\)", "", text)
    text = re.sub(r"\(\s*Probable\s*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()


def select_top_candidates(
    rows: list[dict],
    approved_diseases: set[str],
    limit: int,
) -> list[MechanismCandidate]:
    """Filter, rank, and trim rows to the top `limit` repurposing candidate DISEASES.

    Keeps POSITIVE rows whose disease is not in `approved_diseases` (case-insensitive exact match), sorts by
    `ranking_score` descending, collapses to one row per disease, takes `limit`, and returns MechanismCandidate
    objects. `ranking_score` is OT's overall_score in production and the leak-free recomputed score in holdout mode.

    `limit` counts distinct diseases, not rows: OT lists a disease once per target, so a drug with many related
    targets (metformin's 51 complex I subunits) would otherwise spend the limit on repeats.

    Direction uses majority voting over the row's `evidences` (see `aggregate_directions`).

    Required row keys:
        target_symbol: str
        action_types: set[str]
        disease_name: str
        disease_id: str | None
        overall_score: float | None
        ranking_score: float | None      # leak-free in holdout; == overall_score in production
        evidences: list[EvidenceRecord]   # raw records, direction-voted internally
        disease_description: str
        target_function: str
    """
    approved_lower = {d.lower() for d in approved_diseases if d}

    positives: list[dict] = []
    for row in rows:
        dir_targets, dir_traits = aggregate_directions(row.get("evidences") or [])
        if not classify_positive(
            row.get("action_types") or set(),
            dir_targets,
            dir_traits,
        ):
            continue
        if _is_approved(row.get("disease_name", ""), approved_lower):
            continue
        if row.get("disease_name", "").lower().strip() in BROADENING_BLOCKLIST:
            continue
        positives.append(row)

    positives.sort(key=lambda r: r.get("ranking_score") or 0.0, reverse=True)

    # One row per disease, keyed on the EFO ID and falling back to the name for rows OT left without one.
    # Insertion order is score order, so the first row seen for a disease is its highest-scoring target.
    best_by_disease: dict[str, dict] = {}
    for row in positives:
        key = row.get("disease_id") or row.get("disease_name", "").lower().strip()
        if key and key not in best_by_disease:
            best_by_disease[key] = row

    return [
        MechanismCandidate(
            target_symbol=row.get("target_symbol", ""),
            action_type=_canonical_action(row.get("action_types") or set()),
            disease_name=row.get("disease_name", ""),
            disease_id=row.get("disease_id"),
            disease_description=row.get("disease_description", "") or "",
            target_function=clean_function_description(
                row.get("target_function", "") or ""
            ),
        )
        for row in list(best_by_disease.values())[:limit]
    ]


# --- internal helpers -------------------------------------------------------


def _drug_direction(action_types: set[str]) -> str:
    """Map drug action set to 'LoF' / 'GoF' / 'unknown'; mixed sets (e.g. INHIBITOR and AGONIST) are 'unknown'.

    Examples:
        >>> _drug_direction({"INHIBITOR"})
        'LoF'
        >>> _drug_direction({"AGONIST"})
        'GoF'
        >>> _drug_direction({"INHIBITOR", "AGONIST"})
        'unknown'
        >>> _drug_direction(set())
        'unknown'
    """
    has_lof = bool(action_types & LOF_ACTION_TYPES)
    has_gof = bool(action_types & GOF_ACTION_TYPES)
    if has_lof and has_gof:
        return "unknown"
    if has_lof:
        return "LoF"
    if has_gof:
        return "GoF"
    return "unknown"


def _disease_direction(
    directions_on_target: set[str],
    directions_on_trait: set[str],
) -> str:
    """Infer what direction of target perturbation CAUSES the disease.

    Returns 'LoF-driven' / 'GoF-driven' / 'inconclusive' (records disagree) / 'unknown' (either field empty).
    LoF-driven: loss of target function causes the disease, so it needs a GoF drug (agonist/activator).
    GoF-driven: gain of target function causes the disease, so it needs an LoF drug (inhibitor/antagonist).

    The label is derived from the two evidence-record fields:

      direction_on_target | direction_on_trait | meaning                                       | label
      --------------------|--------------------|-----------------------------------------------|-----------
      LoF                 | risk               | losing target activity → causes disease       | LoF-driven
      GoF                 | risk               | gaining target activity → causes disease      | GoF-driven
      LoF                 | protect            | losing protects ⇒ having it drives disease    | GoF-driven
      GoF                 | protect            | gaining protects ⇒ losing it drives disease   | LoF-driven

    `protect` rows flip the inference: if losing the target protects against disease, the disease is driven by the
    target being active.

    Examples:
        >>> _disease_direction({"LoF"}, {"risk"})
        'LoF-driven'
        >>> _disease_direction({"GoF"}, {"risk"})
        'GoF-driven'
        >>> _disease_direction({"LoF"}, {"protect"})
        'GoF-driven'
        >>> _disease_direction({"GoF"}, {"protect"})
        'LoF-driven'
        >>> _disease_direction(set(), {"risk"})
        'unknown'
        >>> _disease_direction({"LoF", "GoF"}, {"risk"})
        'inconclusive'
    """
    if not directions_on_target or not directions_on_trait:
        return "unknown"
    if len(directions_on_target) > 1 or len(directions_on_trait) > 1:
        return "inconclusive"
    dt = next(iter(directions_on_target))
    dtr = next(iter(directions_on_trait))
    if dtr == "risk":
        if dt == "LoF":
            return "LoF-driven"
        if dt == "GoF":
            return "GoF-driven"
    elif dtr == "protect":
        if dt == "LoF":
            return "GoF-driven"
        if dt == "GoF":
            return "LoF-driven"
    return "unknown"


def _is_approved(disease_name: str, approved_lower: set[str]) -> bool:
    """Case-insensitive exact match. `approved_diseases` is expected to contain disease names already
    matched to OT's vocabulary upstream (e.g. by services.approval_check.get_fda_approved_disease_mapping),
    so this layer just drops exact hits — no synonym or substring logic."""
    if not disease_name or not approved_lower:
        return False
    return disease_name.lower() in approved_lower


def _canonical_action(action_types: set[str]) -> str:
    if not action_types:
        return ""
    return sorted(action_types)[0]
