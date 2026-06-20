"""FDA approval check service.

Uses openFDA drug labels + LLM extraction to identify which candidate
diseases are already FDA-approved for a given drug's trade names.
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from indication_scout.config import get_settings
from indication_scout.constants import (
    CACHE_TTL,
    CURATED_FDA_APPROVED_CANDIDATES,
    CURATED_FDA_COMBINATION_ONLY_CANDIDATES,
    CURATED_FDA_CONTAMINATED_CANDIDATES,
    CURATED_FDA_REJECTED_CANDIDATES,
    DEFAULT_CACHE_DIR,
    DRUG_APPROVALS_PATH,
)
from indication_scout.data_sources.chembl import (
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.data_sources.fda import FDAClient
from indication_scout.services.llm import (
    parse_last_json_array,
    parse_last_json_object,
    query_llm,
    query_small_llm,
)
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_DRUG_APPROVAL_NS = "fda_drug_disease_approval"
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Per-candidate approval relationship, decided once from the FDA label. Only "approved" removes a
# candidate; "combination_only" demotes; "contaminated" and "none" both rank (contaminated also
# flags trial-table suppression). See PLAN_approval_labeling_upstream.md.
ApprovalLabel = Literal["approved", "combination_only", "contaminated", "none"]
APPROVAL_LABELS: frozenset[str] = frozenset(
    ("approved", "combination_only", "contaminated", "none")
)


def _coerce_label(value: Any) -> ApprovalLabel | None:
    """Validate a raw value as an ApprovalLabel. Returns None if invalid.

    Single source of truth for label validation — used by both the LLM-parse path
    and the per-drug cache loader so an invalid/legacy value is skipped, never
    silently coerced (e.g. a bool or stale string must NOT become "approved").
    """
    if isinstance(value, str) and value in APPROVAL_LABELS:
        return value  # type: ignore[return-value]
    return None


def _drug_approval_path(drug_name: str, cache_dir: Path) -> Path:
    """Return the per-drug cache file path for the approvals namespace.

    Slugifies the drug name to a filesystem-safe stem; appends an 8-char
    SHA suffix to disambiguate names that collapse to the same slug
    (e.g. "5-FU" vs "5_FU") and to handle empty slugs.
    """
    slug = _SLUG_RE.sub("_", drug_name.lower()).strip("_")
    # Fold the LLM model into the key: verdicts are LLM-generated, so a model
    # change must not serve stale verdicts from the prior model.
    keyed = f"{drug_name}\x00{_settings.llm_model}"
    suffix = hashlib.sha256(keyed.encode()).hexdigest()[:8]
    stem = f"{slug}_{suffix}" if slug else suffix
    return cache_dir / _DRUG_APPROVAL_NS / f"{stem}.json"


def _load_drug_approvals(drug_name: str, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the per-drug approvals file as {disease_lower: {label, cached_at, ttl}}.

    Returns an empty dict if the file is missing or unparseable. Expired
    entries are dropped lazily; the file is rewritten only when a caller
    invokes _save_drug_approvals. Entries whose `label` is missing or not a
    valid ApprovalLabel are skipped (stale bool-shaped caches do not survive).
    """
    path = _drug_approval_path(drug_name, cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}

    fresh: dict[str, dict[str, Any]] = {}
    now = datetime.now()
    for disease_key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            ttl = int(entry.get("ttl", CACHE_TTL))
        except (KeyError, TypeError, ValueError):
            continue
        label = _coerce_label(entry.get("label"))
        if label is None:
            continue
        if (now - cached_at).total_seconds() > ttl:
            continue
        fresh[disease_key] = {
            "label": label,
            "cached_at": entry["cached_at"],
            "ttl": ttl,
        }
    return fresh


def _save_drug_approvals(
    drug_name: str,
    new_labels: dict[str, ApprovalLabel],
    cache_dir: Path,
    ttl: int = CACHE_TTL,
) -> None:
    """Merge new_labels into the per-drug file and write it back.

    Existing unexpired entries are preserved; new labels overwrite any
    prior entry for the same disease (refreshing its cached_at).
    """
    if not new_labels:
        return
    path = _drug_approval_path(drug_name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_drug_approvals(drug_name, cache_dir)
    now_iso = datetime.now().isoformat()
    for disease, label in new_labels.items():
        existing[disease.lower()] = {
            "label": label,
            "cached_at": now_iso,
            "ttl": ttl,
        }
    payload = {
        "ns": _DRUG_APPROVAL_NS,
        "drug_name": drug_name,
        "entries": existing,
    }
    path.write_text(json.dumps(payload, default=str, indent=2))


# --------------------------------------------------------------------------
# Hardcoded FDA approval lookup (used during temporal holdouts)
#
# When the pipeline is invoked with `date_before` set, the live openFDA
# label path leaks today's approvals into a holdout (e.g. a 2020 holdout
# would see semaglutide as approved for MASH because the current label
# lists it). The lookup below replaces the live path with a hardcoded
# {drug: [{disease, approved}]} table, gated on `as_of`, so a holdout sees
# only approvals that existed on or before the cutoff.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_drug_approvals_table() -> dict[str, list[dict[str, str]]]:
    """Load the hardcoded approvals JSON file once per process."""
    if not DRUG_APPROVALS_PATH.exists():
        logger.warning(
            "drug approvals table not found at %s; lookup will return empty",
            DRUG_APPROVALS_PATH,
        )
        return {}
    raw = json.loads(DRUG_APPROVALS_PATH.read_text())
    if not isinstance(raw, dict):
        logger.error(
            "drug approvals table at %s is not a JSON object; ignoring",
            DRUG_APPROVALS_PATH,
        )
        return {}
    return raw


async def get_approved_indications(
    drug_name: str,
    candidate_diseases: list[str],
    as_of: date | None,
) -> set[str]:
    """Return the subset of candidate_diseases that the drug was FDA-approved
    for on or before `as_of`, sourced from the hardcoded approvals table.

    Matching uses the same LLM disease matcher as the competitor merge
    (`merge_duplicate_diseases` REMOVE): a candidate is dropped only when it is
    the SAME clinical condition as a pre-cutoff approved indication. This avoids
    the substring trap where a broader/sibling candidate ("leukemia",
    "depressive disorder") was wrongly stripped because it shares a substring
    with a more specific approved indication ("chronic myeloid leukemia",
    "major depressive disorder"). Subtype/parent and same-category-distinct
    pairs stay separate per the matcher's prohibition rules.

    Returns empty set when:
      - drug is not in the table (logs a warning)
      - as_of is None (callers should use the live FDA path in that case)
      - no table entry's `approved` date is < as_of
    """
    if as_of is None:
        return set()
    if not drug_name or not candidate_diseases:
        return set()

    table = _load_drug_approvals_table()
    entries = table.get(drug_name.lower().strip())
    if entries is None:
        logger.warning(
            "get_approved_indications: %r not in hardcoded approvals table; "
            "returning empty set (approval reasoning disabled for this holdout run)",
            drug_name,
        )
        return set()

    pre_cutoff_diseases: list[str] = []
    for entry in entries:
        approved_str = entry.get("approved")
        disease = entry.get("disease")
        if not approved_str or not disease:
            continue
        try:
            approved_dt = date.fromisoformat(approved_str)
        except ValueError:
            logger.warning(
                "get_approved_indications: bad date %r for %s/%s; skipping",
                approved_str,
                drug_name,
                disease,
            )
            continue
        if approved_dt < as_of:
            pre_cutoff_diseases.append(disease.lower().strip())

    if not pre_cutoff_diseases:
        return set()

    # LLM equivalence: REMOVE returns the candidate names that are the SAME
    # condition as a pre-cutoff approved indication (subtype/parent/sibling kept
    # separate by the prompt's prohibition rules).
    from indication_scout.services.disease_helper import merge_duplicate_diseases

    merge_result = await merge_duplicate_diseases(
        candidate_diseases, pre_cutoff_diseases
    )
    removed_lower = {r.lower().strip() for r in merge_result.get("remove", [])}
    return {c for c in candidate_diseases if c.lower().strip() in removed_lower}


def list_approved_indications_at(
    drug_name: str,
    as_of: date | None,
) -> list[str]:
    """Return all approved indication strings for `drug_name` on or before
    `as_of`, sourced from the hardcoded approvals table.

    Used to seed the supervisor's drug briefing (the analogue of
    `list_approved_indications_from_labels`) during temporal holdouts.
    Returns empty list when as_of is None or drug is not in the table.
    """
    if as_of is None:
        return []
    if not drug_name:
        return []

    table = _load_drug_approvals_table()
    entries = table.get(drug_name.lower().strip())
    if entries is None:
        logger.warning(
            "list_approved_indications_at: %r not in hardcoded approvals table; "
            "returning empty list",
            drug_name,
        )
        return []

    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        approved_str = entry.get("approved")
        disease = entry.get("disease")
        if not approved_str or not disease:
            continue
        try:
            approved_dt = date.fromisoformat(approved_str)
        except ValueError:
            continue
        if approved_dt >= as_of:
            continue
        key = disease.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(disease)
    return out


async def list_approved_indications_from_labels(
    label_texts: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[str]:
    """Extract the approved indications named in FDA label text, no candidate list.

    Companion to extract_approved_from_labels. That function asks "which of
    THESE candidates are approved per the label?" — useful for filtering a
    candidate list. This function asks "what indications does the label
    approve?" — useful for seeding the supervisor's drug-level briefing with
    approvals discovered up front, before any candidate list exists.

    Args:
        label_texts: Raw indications_and_usage strings from openFDA.
        cache_dir: Cache directory for storing LLM results.

    Returns:
        Deduplicated list of approved indication names extracted from the
        label text. Empty list if label_texts is empty, the LLM response is
        not parseable JSON, or the LLM returns a non-list. Order-preserving.
    """
    if not label_texts:
        return []

    cache_params = {
        "label_texts": sorted(label_texts),
        "llm_model": _settings.llm_model,
    }
    cached = cache_get("fda_label_indications", cache_params, cache_dir)
    if cached is not None:
        return list(cached)

    template = (_PROMPTS_DIR / "list_label_indications.txt").read_text()
    prompt = template.format(label_texts="\n---\n".join(label_texts))

    response = await query_llm(prompt)
    parsed = parse_last_json_array(response)

    if parsed is None:
        logger.error(
            "list_approved_indications_from_labels: failed to parse LLM response: %s",
            response,
        )
        return []

    indications: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        indications.append(cleaned)

    cache_set(
        "fda_label_indications",
        cache_params,
        indications,
        cache_dir,
        ttl=CACHE_TTL,
    )
    return indications


async def extract_approved_from_labels(
    label_texts: list[str],
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> set[str]:
    """Use an LLM to identify which candidate diseases appear as approved indications in FDA label text.

    Args:
        label_texts: Raw indications_and_usage strings from openFDA.
        candidate_diseases: Disease names to check against the labels.
        cache_dir: Cache directory for storing LLM results.

    Returns:
        Set of candidate disease names (verbatim from input) found in the labels.
    """
    if not label_texts or not candidate_diseases:
        return set()

    cache_params = {
        "label_texts": sorted(label_texts),
        "candidate_diseases": sorted(candidate_diseases),
        "small_llm_model": _settings.small_llm_model,
    }
    cached = cache_get("fda_approval_check", cache_params, cache_dir)
    if cached is not None:
        return set(cached)

    template = (_PROMPTS_DIR / "extract_fda_approvals.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=", ".join(candidate_diseases),
    )

    response = await query_small_llm(prompt)
    parsed = parse_last_json_array(response)

    if parsed is None:
        logger.error(
            "extract_approved_from_labels: failed to parse LLM response: %s", response
        )
        return set()

    candidate_lower_map = {c.lower(): c for c in candidate_diseases}
    validated: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        original = candidate_lower_map.get(item.lower())
        if original is not None:
            validated.add(original)
        else:
            logger.warning(
                "extract_approved_from_labels: LLM returned unknown disease %r, skipping",
                item,
            )

    cache_set(
        "fda_approval_check",
        cache_params,
        list(validated),
        cache_dir,
        ttl=CACHE_TTL,
    )
    return validated


# TODO delete
async def get_all_fda_approved_diseases(
    drug_names: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Any:
    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_names)

    if not label_texts:
        return set()
    return label_texts


async def get_fda_approved_disease_mapping(
    drug_name: str,
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, ApprovalLabel]:
    """Classify each candidate's relationship to the drug's approved indications.

    Two-tier lookup:
      1. Curated short-circuit — exact (case-sensitive) match against the drug's
         curated approved / contaminated / rejected lists; skips the LLM.
      2. LLM fallback — for remaining candidates, the input drug_name is
         expanded to all known aliases (generic, trade, INN, USAN, salt forms)
         via ChEMBL, all matching openFDA labels are fetched, and the
         candidates are batched into one LLM call that returns one ApprovalLabel
         per candidate.

    Args:
        drug_name: A single drug name (trade, generic/INN, or USAN).
        candidate_diseases: Disease names to check against the label.
        cache_dir: Cache directory.

    Returns:
        Dict mapping each input candidate (verbatim) to an ApprovalLabel:
        "approved" (already covered → drop), "combination_only" (demote),
        "contaminated" (keep, suspect trial counts), or "none" (keep, clean).
        Every input candidate is always present as a key. Defaults to "none"
        on any failure (chembl, FDA fetch, LLM parse) — a failure must NOT
        drop a candidate.
    """
    result: dict[str, ApprovalLabel] = {c: "none" for c in candidate_diseases}

    if not drug_name or not candidate_diseases:
        return result

    # Curated short-circuit: exact, case-sensitive match against the drug's
    # curated lists. Hits skip both the FDA fetch and the LLM call.
    approved_set = set(CURATED_FDA_APPROVED_CANDIDATES.get(drug_name, []))
    contaminated_set = set(CURATED_FDA_CONTAMINATED_CANDIDATES.get(drug_name, []))
    combination_set = set(CURATED_FDA_COMBINATION_ONLY_CANDIDATES.get(drug_name, []))
    rejected_set = set(CURATED_FDA_REJECTED_CANDIDATES.get(drug_name, []))
    uncurated: list[str] = []
    for c in candidate_diseases:
        if c in approved_set:
            result[c] = "approved"
        elif c in combination_set:
            result[c] = "combination_only"
        elif c in contaminated_set:
            result[c] = "contaminated"
        elif c in rejected_set:
            result[c] = "none"
        else:
            uncurated.append(c)

    if not uncurated:
        return result

    # Per-pair drug-disease label cache: one file per drug holding a
    # {disease_lower: {label, cached_at, ttl}} map. Apply unexpired hits
    # directly to result; only the still-missing candidates are sent to
    # ChEMBL/FDA/LLM below. New labels are merged back into the same file
    # after the LLM call. Curated entries are applied above and never cached
    # here, so curated overrides always win. TTL is per-label, preserving
    # the previous semantics where each (drug, disease) pair expired
    # independently.
    drug_cache = _load_drug_approvals(drug_name, cache_dir)
    still_missing: list[str] = []
    for c in uncurated:
        entry = drug_cache.get(c.lower())
        if entry is None:
            still_missing.append(c)
        else:
            result[c] = entry["label"]

    if not still_missing:
        return result

    # Expand the input drug name to all known aliases (generic, trade, INN,
    # USAN, salt forms, etc.) via ChEMBL. Different formulations of the same
    # drug carry distinct openFDA labels (e.g. fluoxetine generic vs Sarafem
    # for PMDD), so feeding all aliases to get_all_label_indications surfaces
    # approvals that a single-name lookup would miss. On any chembl failure,
    # fall back to the bare drug_name.
    try:
        chembl_id = await resolve_drug_name(drug_name, cache_dir)
        drug_aliases = await get_all_drug_names(chembl_id, cache_dir)
        if drug_name not in drug_aliases:
            drug_aliases = [drug_name, *drug_aliases]
        logger.info(
            "get_fda_approved_disease_mapping: %r → chembl_id=%s, %d aliases",
            drug_name,
            chembl_id,
            len(drug_aliases),
        )
    except Exception as e:
        logger.warning(
            "get_fda_approved_disease_mapping: chembl alias lookup failed for %r: %s; "
            "falling back to bare drug_name",
            drug_name,
            e,
        )
        drug_aliases = [drug_name]

    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    logger.info(
        "get_fda_approved_disease_mapping: %r → fetched %d label texts from %d aliases",
        drug_name,
        len(label_texts),
        len(drug_aliases),
    )

    if not label_texts:
        logger.warning(
            "get_fda_approved_disease_mapping: %r → no label texts found across %d aliases; "
            "returning False for %d candidate(s)",
            drug_name,
            len(drug_aliases),
            len(still_missing),
        )
        return result

    template = (_PROMPTS_DIR / "extract_fda_approval_single.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=json.dumps(still_missing),
    )

    response = await query_llm(prompt)
    parsed = parse_last_json_object(response)

    if parsed is None:
        logger.error(
            "get_fda_approved_disease_mapping: failed to parse LLM response: %s",
            response,
        )
        return result

    # Map LLM keys back to verbatim input candidates (case-insensitive), scoped
    # to the still-missing candidates so a stray key cannot overwrite a curated
    # or already-cached value.
    lower_to_verbatim = {c.lower(): c for c in still_missing}
    llm_labels: dict[str, ApprovalLabel] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        original = lower_to_verbatim.get(key.lower())
        if original is None:
            logger.warning(
                "get_fda_approved_disease_mapping: LLM returned unknown candidate %r, skipping",
                key,
            )
            continue
        label = _coerce_label(value)
        if label is None:
            logger.error(
                "get_fda_approved_disease_mapping: value for %r is not a valid label: %r",
                key,
                value,
            )
            continue
        result[original] = label
        llm_labels[original] = label

    # Merge fresh labels into the per-drug cache file. Only candidates the LLM
    # returned a valid label for are cached — parse failures or skipped
    # candidates remain uncached so a future call can retry them.
    _save_drug_approvals(drug_name, llm_labels, cache_dir, ttl=CACHE_TTL)

    return result
