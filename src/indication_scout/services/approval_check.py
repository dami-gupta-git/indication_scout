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
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

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
from indication_scout.data_sources.base_client import DataSourceError
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

# Per-candidate approval relationship, decided once from the FDA label. Only "approved" removes a candidate;
# "combination_only" demotes; "contaminated" and "none" both rank (contaminated also flags trial-table suppression).
# "unresolved" is not a judgment: the mapping returns it for a candidate it could not judge (no label text found, the FDA
# lookup failed, or the model's answer was unparseable or did not cover the candidate). It is never cached, never emitted by
# the model, and never accepted from the per-drug cache, so APPROVAL_LABELS below lists only the four judged labels.
ApprovalLabel = Literal[
    "approved", "combination_only", "contaminated", "none", "unresolved"
]
APPROVAL_LABELS: frozenset[str] = frozenset(
    ("approved", "combination_only", "contaminated", "none")
)


class _ContaminatedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["contaminated"]
    matched_approved_indication: str
    reason: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class _UnanchoredDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["approved", "combination_only", "none"]
    reason: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


_ApprovalDecision = Annotated[
    _ContaminatedDecision | _UnanchoredDecision,
    Field(discriminator="label"),
]
_APPROVAL_DECISION_ADAPTER: TypeAdapter[_ContaminatedDecision | _UnanchoredDecision] = (
    TypeAdapter(_ApprovalDecision)
)


def _coerce_label(value: Any) -> ApprovalLabel | None:
    """Validate a raw value as an ApprovalLabel; None if invalid.

    Shared by the LLM-parse path and the per-drug cache loader so an invalid or legacy value (a bool, a stale string) is
    skipped, never coerced to "approved".
    """
    if isinstance(value, str) and value in APPROVAL_LABELS:
        return value  # type: ignore[return-value]
    return None


def _parse_approval_decisions(
    parsed: dict[str, Any],
    candidates: list[str],
    approved_indications: list[str],
) -> dict[str, ApprovalLabel]:
    """Validate structured decisions and reject contaminated labels without an exact anchor."""
    lower_to_verbatim = {candidate.lower(): candidate for candidate in candidates}
    approved_set = set(approved_indications)
    labels: dict[str, ApprovalLabel] = {}

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
        try:
            decision = _APPROVAL_DECISION_ADAPTER.validate_python(value)
        except ValidationError as exc:
            logger.error(
                "get_fda_approved_disease_mapping: invalid decision for %r: %s",
                key,
                exc,
            )
            continue
        if isinstance(decision, _ContaminatedDecision):
            if decision.matched_approved_indication not in approved_set:
                logger.warning(
                    "get_fda_approved_disease_mapping: rejecting contaminated verdict for %r "
                    "with unverified anchor %r",
                    key,
                    decision.matched_approved_indication,
                )
                continue
            labels[original] = "contaminated"
        else:
            labels[original] = decision.label

    return labels


def _drug_approval_path(drug_name: str, cache_dir: Path) -> Path:
    """Return the per-drug cache file path for the approvals namespace.

    The slugified drug name carries an 8-char SHA suffix to disambiguate names that collapse to the same slug
    (e.g. "5-FU" vs "5_FU") and to handle empty slugs.
    """
    slug = _SLUG_RE.sub("_", drug_name.lower()).strip("_")
    # Fold the LLM model into the key: verdicts are LLM-generated, so a model change must not serve stale ones.
    keyed = f"{drug_name}\x00{_settings.llm_model}"
    suffix = hashlib.sha256(keyed.encode()).hexdigest()[:8]
    stem = f"{slug}_{suffix}" if slug else suffix
    return cache_dir / _DRUG_APPROVAL_NS / f"{stem}.json"


def _load_drug_approvals(drug_name: str, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the per-drug approvals file as {disease_lower: {label, cached_at, ttl}}.

    Returns an empty dict if the file is missing or unparseable. Expired entries are dropped lazily (the file is
    rewritten only by _save_drug_approvals), and entries with a missing or invalid `label` are skipped.
    """
    path = _drug_approval_path(drug_name, cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    # Older verdicts may have been judged against parent-only aliases after a failed salt-form lookup.
    if not isinstance(raw, dict) or raw.get("salt_forms_checked") is not True:
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

    Unexpired entries are preserved; a new label overwrites any prior entry for the same disease (refreshing cached_at).
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
        "salt_forms_checked": True,
        "entries": existing,
    }
    path.write_text(json.dumps(payload, default=str, indent=2))


# --------------------------------------------------------------------------
# Hardcoded FDA approval lookup (used during temporal holdouts)
#
# With `date_before` set, the live openFDA path would leak today's approvals into a holdout (a 2020 holdout would see
# semaglutide approved for MASH). This {drug: [{disease, approved, combination_with?}]} table, gated on `as_of`, replaces
# it so a holdout sees only approvals that existed on or before the cutoff. An entry with `combination_with` is an
# approval of a fixed-dose combination product (bupropion with naltrexone for obesity): it never counts as an approval of
# the drug alone, only as combination_only.
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook that raises on a repeated key instead of keeping the last value."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate drug key {key!r} in drug approvals table")
        seen[key] = value
    return seen


@lru_cache(maxsize=1)
def _load_drug_approvals_table() -> dict[str, list[dict[str, str]]]:
    """Load the hardcoded approvals JSON file once per process."""
    if not DRUG_APPROVALS_PATH.exists():
        logger.warning(
            "drug approvals table not found at %s; lookup will return empty",
            DRUG_APPROVALS_PATH,
        )
        return {}
    raw = json.loads(
        DRUG_APPROVALS_PATH.read_text(), object_pairs_hook=_reject_duplicate_keys
    )
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
    """Return the candidate_diseases the drug alone was FDA-approved for before `as_of`, per the hardcoded table.

    Combination-product entries (`combination_with`) are excluded; get_combination_only_indications reports those.

    Matching uses the same LLM disease matcher as the competitor merge (`merge_duplicate_diseases` REMOVE): a candidate
    is dropped only when it is the SAME clinical condition as a pre-cutoff approved indication. This avoids the
    substring trap where a broader or sibling candidate ("leukemia") was stripped because it shares a substring with a
    more specific approved indication ("chronic myeloid leukemia"). Subtype/parent and same-category pairs stay separate.

    Returns empty set when:
      - drug is not in the table (logs a warning)
      - as_of is None (callers should use the live FDA path in that case)
      - no single-drug table entry's `approved` date is < as_of
    """
    return await _match_table_entries(
        drug_name, candidate_diseases, as_of, combination=False
    )


async def get_combination_only_indications(
    drug_name: str,
    candidate_diseases: list[str],
    as_of: date | None,
) -> set[str]:
    """Return the candidate_diseases approved before `as_of` only inside a combination product, per the hardcoded table.

    Same matching and empty-set conditions as get_approved_indications, over the `combination_with` entries only.
    """
    return await _match_table_entries(
        drug_name, candidate_diseases, as_of, combination=True
    )


async def _match_table_entries(
    drug_name: str,
    candidate_diseases: list[str],
    as_of: date | None,
    combination: bool,
) -> set[str]:
    """Match candidates against the drug's pre-`as_of` table entries of one kind (single-drug or combination)."""
    if as_of is None:
        return set()
    if not drug_name or not candidate_diseases:
        return set()

    table = _load_drug_approvals_table()
    entries = table.get(drug_name.lower().strip())
    if entries is None:
        logger.warning(
            "get_approved_indications: %r not in hardcoded approvals table; returning empty set",
            drug_name,
        )
        return set()

    pre_cutoff_diseases: list[str] = []
    for entry in entries:
        if bool(entry.get("combination_with")) is not combination:
            continue
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

    # LLM equivalence: REMOVE returns candidates that are the SAME condition as a pre-cutoff approved indication.
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
    """Return all approved indication strings for `drug_name` on or before `as_of`, from the hardcoded approvals table.

    Seeds the supervisor's drug briefing during temporal holdouts (analogue of `list_approved_indications_from_labels`).
    Combination-product entries are not approvals of the drug alone and are left out. Returns an empty list when as_of is
    None or the drug is not in the table.
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
        if entry.get("combination_with"):
            continue
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


def _table_as_label_text(drug_name: str, as_of: date) -> str | None:
    """Render the drug's table entries approved before `as_of` as label-style text for the approval prompt.

    Combination-product entries name their partner drug so the prompt can label them combination_only. Returns None when
    the drug has no entry before `as_of`.
    """
    entries = _load_drug_approvals_table().get(drug_name.lower().strip())
    if not entries:
        return None
    lines: list[str] = []
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
        partner = entry.get("combination_with")
        if partner:
            lines.append(
                f"- {disease} (approved only in a fixed-dose combination product with {partner})"
            )
        else:
            lines.append(f"- {disease}")
    if not lines:
        return None
    return f"INDICATIONS AND USAGE ({drug_name}, from a curated FDA approvals record):\n" + "\n".join(lines)


async def list_approved_indications_from_labels(
    label_texts: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[str]:
    """Extract the approved indications named in FDA label text, with no candidate list.

    Companion to extract_approved_from_labels, which asks which of THESE candidates are approved. This asks what the
    label approves, to seed the supervisor's drug-level briefing before any candidate list exists.

    Args:
        label_texts: Raw indications_and_usage strings from openFDA.
        cache_dir: Cache directory for storing LLM results.

    Returns:
        Deduplicated, order-preserving list of approved indication names. Empty if label_texts is empty, the LLM
        response is not parseable JSON, or the LLM returns a non-list.
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
    approved_indications: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, ApprovalLabel]:
    """Classify each candidate's relationship to the drug's approved indications.

    Lookup order:
      1. Curated short-circuit: exact (case-sensitive) match against the drug's curated approved / contaminated /
         rejected lists; skips the LLM.
      2. LLM judgment: the drug_name is expanded to all known aliases via ChEMBL, all matching openFDA labels are
         fetched, and the remaining candidates go into one LLM call. Each structured decision includes an
         ApprovalLabel; contaminated decisions must also name an exact item from approved_indications.
      3. When openFDA returns no label text, the drug's hardcoded approvals-table entries stand in for the label text in
         the same LLM call (openFDA dropped baricitinib's Olumiant label from its name index, October 2026). These
         verdicts are not cached.

    Args:
        drug_name: A single drug name (trade, generic/INN, or USAN).
        candidate_diseases: Disease names to check against the label.
        approved_indications: Label-derived indications from the run's shared drug intake.
        cache_dir: Cache directory.

    Returns:
        Dict mapping each input candidate (verbatim) to an ApprovalLabel: "approved" (already covered → drop),
        "combination_only" (demote), "contaminated" (keep, suspect trial counts), "none" (keep, clean) or "unresolved"
        (keep; the candidate could not be judged). Every input candidate is a key. A failure (no label text, FDA fetch,
        LLM parse, an answer that omits the candidate) yields "unresolved", never "none": a failure must NOT drop a
        candidate and must not read as a judged "not approved". Unresolved labels are not cached.
    """
    result: dict[str, ApprovalLabel] = dict.fromkeys(candidate_diseases, "none")

    if not drug_name or not candidate_diseases:
        return result

    # Curated short-circuit: exact, case-sensitive match against the drug's curated lists; hits skip the FDA fetch and LLM.
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

    # Per-pair label cache: one file per drug holding {disease_lower: {label, cached_at, ttl}}. Unexpired hits go straight
    # into result; only the rest are sent to ChEMBL/FDA/LLM, and new labels are merged back after the call. Curated
    # entries are applied above and never cached, so curated overrides always win. TTL is per label.
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

    # Nothing below has judged these candidates yet; any early return leaves them unresolved, not "none".
    for c in still_missing:
        result[c] = "unresolved"

    # Expand the drug name to all known aliases via ChEMBL: different formulations carry distinct openFDA labels
    # (fluoxetine generic vs Sarafem for PMDD), so all aliases surface approvals a single-name lookup would miss. On a
    # chembl failure the remaining candidates stay unresolved: a bare-name lookup can return an incomplete label set
    # (sildenafil without Revatio misses PAH) and the model would then cache a conclusive "none" from partial evidence.
    try:
        chembl_id = await resolve_drug_name(drug_name, cache_dir)
        drug_aliases = await get_all_drug_names(chembl_id, cache_dir)
        if drug_name not in drug_aliases:
            drug_aliases = [drug_name, *drug_aliases]
        logger.debug(
            "get_fda_approved_disease_mapping: %r → chembl_id=%s, %d aliases",
            drug_name,
            chembl_id,
            len(drug_aliases),
        )
    except Exception as e:
        logger.warning(
            "get_fda_approved_disease_mapping: chembl alias lookup failed for %r: %s; "
            "leaving %d candidate(s) unresolved",
            drug_name,
            e,
            len(still_missing),
        )
        return result

    try:
        async with FDAClient(cache_dir=cache_dir) as client:
            label_texts = await client.get_all_label_indications(drug_aliases)
    except DataSourceError as e:
        logger.warning(
            "get_fda_approved_disease_mapping: FDA label fetch failed for %r: %s; leaving %d candidate(s) unresolved",
            drug_name,
            e,
            len(still_missing),
        )
        return result

    logger.debug(
        "get_fda_approved_disease_mapping: %r → fetched %d label texts from %d aliases",
        drug_name,
        len(label_texts),
        len(drug_aliases),
    )

    # openFDA can drop a current label from its name index (baricitinib's Olumiant label, October 2026). The hardcoded
    # approvals table then stands in for the label text, judged by the same prompt. Verdicts from the table are not
    # cached: once openFDA serves the label again, the next call judges the real label.
    from_table = False
    if not label_texts:
        table_text = _table_as_label_text(drug_name, as_of=date.today())
        if table_text is None:
            logger.warning(
                "get_fda_approved_disease_mapping: %r → no label texts found across %d aliases and no table entry; "
                "leaving %d candidate(s) unresolved",
                drug_name,
                len(drug_aliases),
                len(still_missing),
            )
            return result
        logger.warning(
            "get_fda_approved_disease_mapping: %r → no label texts found across %d aliases; judging %d candidate(s) "
            "against the hardcoded approvals table",
            drug_name,
            len(drug_aliases),
            len(still_missing),
        )
        label_texts = [table_text]
        from_table = True

    template = (_PROMPTS_DIR / "extract_fda_approval_single.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        approved_indications=json.dumps(approved_indications),
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

    llm_labels = _parse_approval_decisions(
        parsed,
        still_missing,
        approved_indications,
    )
    result.update(llm_labels)

    # Merge fresh labels into the per-drug cache file. Only candidates with a valid LLM label are cached; parse failures
    # and skipped candidates stay uncached so a later call can retry them.
    if not from_table:
        _save_drug_approvals(drug_name, llm_labels, cache_dir, ttl=CACHE_TTL)

    return result
