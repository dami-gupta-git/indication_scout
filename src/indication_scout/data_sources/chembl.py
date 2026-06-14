"""ChEMBL API client."""

import json
import logging

from datetime import datetime
from pathlib import Path

from indication_scout.constants import (
    CACHE_TTL,
    CHEMBL_BASE_URL,
    DEFAULT_CACHE_DIR,
    OPEN_TARGETS_BASE_URL,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_chembl import (
    ATCDescription,
    MoleculeData,
    MoleculeSynonym,
)
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Combined per-ChEMBL-ID cache. One file per parent ChEMBL ID holds
# {"chembl_id", "names", "cached_at", "ttl"}. Replaces the prior
# `chembl_drug_names` (chembl_id → names) and `drug_name_to_chembl`
# (name → chembl_id) namespaces, which together produced one tiny file
# per alias. Forward lookup is a direct file read; reverse lookup is a
# directory scan of all per-drug files (cheap at the tens-to-hundreds
# scale this cache reaches in practice).
_CHEMBL_NAMES_NS = "chembl_id_to_names"
_CHEMBL_MOLECULE_NS = "chembl_molecule"


def _chembl_names_path(chembl_id: str, cache_dir: Path) -> Path:
    """Return the per-ChEMBL-ID cache file path."""
    return cache_dir / _CHEMBL_NAMES_NS / f"{chembl_id}.json"


def _load_chembl_names(chembl_id: str, cache_dir: Path) -> list[str] | None:
    """Return the cached names list for a ChEMBL ID, or None if missing/expired."""
    path = _chembl_names_path(chembl_id, cache_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(raw["cached_at"])
        ttl = int(raw.get("ttl", CACHE_TTL))
        names = raw["names"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if (datetime.now() - cached_at).total_seconds() > ttl:
        return None
    if not isinstance(names, list):
        return None
    return list(names)


def _save_chembl_names(
    chembl_id: str,
    names: list[str],
    cache_dir: Path,
    ttl: int = CACHE_TTL,
) -> None:
    """Write the names list for a ChEMBL ID."""
    path = _chembl_names_path(chembl_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ns": _CHEMBL_NAMES_NS,
        "chembl_id": chembl_id,
        "names": list(names),
        "cached_at": datetime.now().isoformat(),
        "ttl": ttl,
    }
    path.write_text(json.dumps(payload, default=str, indent=2))


def _lookup_chembl_id_by_name(name: str, cache_dir: Path) -> str | None:
    """Reverse-index lookup: scan per-drug files for a name match.

    Returns the parent ChEMBL ID whose cached names list contains `name`
    (case-insensitive), or None if no cached drug owns that alias.
    Expired files are skipped (handled by _load_chembl_names) but not
    pruned here — they'll be overwritten on the next get_all_drug_names
    call for that ChEMBL ID.
    """
    ns_dir = cache_dir / _CHEMBL_NAMES_NS
    if not ns_dir.exists():
        return None
    target = name.lower()
    for path in ns_dir.glob("*.json"):
        try:
            raw = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(raw["cached_at"])
            ttl = int(raw.get("ttl", CACHE_TTL))
            chembl_id = raw["chembl_id"]
            names = raw["names"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if (datetime.now() - cached_at).total_seconds() > ttl:
            continue
        if not isinstance(names, list):
            continue
        if any(isinstance(n, str) and n.lower() == target for n in names):
            return chembl_id
    return None


class ChEMBLClient(BaseClient):
    """Client for querying ChEMBL database."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    @property
    def _source_name(self) -> str:
        return "chembl"

    async def get_atc_description(self, atc_code: str) -> ATCDescription:
        """Fetch ATC classification hierarchy for a single ATC code.

        Hits GET /atc_class/{atc_code}.json and returns an ATCDescription instance.
        Results are cached under namespace "atc_description" for CACHE_TTL seconds.
        Raises DataSourceError if the code is not found or the response is malformed.
        """
        cached = cache_get("atc_description", {"atc_code": atc_code}, self.cache_dir)
        if cached is not None:
            return ATCDescription.model_validate(cached)

        url = f"{CHEMBL_BASE_URL}/atc_class/{atc_code}.json"
        try:
            raw = await self._rest_get(url, params={})
        except Exception as e:
            raise DataSourceError(
                self._source_name,
                f"Unexpected error fetching ATC code '{atc_code}': {e}",
            )

        if not isinstance(raw, dict):
            raise DataSourceError(
                self._source_name,
                f"Unexpected response shape for ATC code '{atc_code}'",
            )

        result = ATCDescription(
            level1=raw["level1"],
            level1_description=raw["level1_description"],
            level2=raw["level2"],
            level2_description=raw["level2_description"],
            level3=raw["level3"],
            level3_description=raw["level3_description"],
            level4=raw["level4"],
            level4_description=raw["level4_description"],
            level5=raw["level5"],
            who_name=raw["who_name"],
        )

        cache_set(
            "atc_description",
            {"atc_code": atc_code},
            result.model_dump(),
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return result

    async def get_molecule(self, chembl_id: str) -> MoleculeData:
        """Fetch molecule data by ChEMBL ID.

        Hits GET /molecule/{chembl_id}.json and returns a MoleculeData instance.
        Raises DataSourceError if the molecule is not found or the response is malformed.

        The raw response is file-cached under `chembl_molecule` (molecule facts like
        first_approval are immutable), so repeat runs skip the network — and the EBI
        timeout — entirely.
        """
        cache_params = {"chembl_id": chembl_id}
        raw = cache_get(_CHEMBL_MOLECULE_NS, cache_params, self.cache_dir)
        if raw is None:
            url = f"{CHEMBL_BASE_URL}/molecule/{chembl_id}.json"
            try:
                raw = await self._rest_get(url, params={})
            except Exception as e:
                raise DataSourceError(
                    self._source_name, f"Unexpected error fetching {chembl_id}: {e}"
                )
            if isinstance(raw, dict):
                cache_set(_CHEMBL_MOLECULE_NS, cache_params, raw, self.cache_dir)

        if not isinstance(raw, dict):
            raise DataSourceError(
                self._source_name, f"Unexpected response shape for '{chembl_id}'"
            )

        synonyms = [
            MoleculeSynonym(
                molecule_synonym=s.get("molecule_synonym", "").lower(),
                syn_type=s.get("syn_type", ""),
                synonyms=s.get("synonyms", "").lower(),
            )
            for s in raw.get("molecule_synonyms") or []
        ]

        hierarchy = raw.get("molecule_hierarchy") or {}

        return MoleculeData(
            molecule_chembl_id=raw["molecule_chembl_id"],
            pref_name=(raw.get("pref_name") or "").lower(),
            parent_chembl_id=hierarchy.get("parent_chembl_id", ""),
            molecule_type=raw["molecule_type"],
            max_phase=raw["max_phase"],
            atc_classifications=raw.get("atc_classifications") or [],
            black_box_warning=raw["black_box_warning"],
            first_approval=raw.get("first_approval"),
            oral=raw["oral"],
            molecule_synonyms=synonyms,
        )


class _OTSearchClient(BaseClient):
    """Minimal client for Open Targets GraphQL search.

    Avoids importing OpenTargetsClient (which imports chembl → circular).
    """

    @property
    def _source_name(self) -> str:
        return "open_targets"


_OT_DRUG_SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
        hits { id entity }
    }
}
"""


async def resolve_drug_name(drug_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """Resolve a free-text drug name to a canonical (parent) ChEMBL ID.

    1. Checks the `resolve_drug_name` cache (previously-resolved inputs).
    2. Scans the `chembl_id_to_names` per-drug cache files for a name match
       (covers any synonym of a drug we've fetched before).
    3. Searches Open Targets GraphQL for the drug name → some ChEMBL ID.
    4. Fetches the molecule from ChEMBL to read parent_chembl_id.
    5. If the result is a salt (parent differs), follows to the parent.

    Results are cached under namespace "resolve_drug_name".
    Raises DataSourceError if the drug is not found.

    Args:
        drug_name: Free-text drug name (e.g. "metformin", "Glucophage").
        cache_dir: Directory for file-based cache.

    Returns:
        Canonical parent ChEMBL ID (e.g. "CHEMBL1431").
    """
    normalized = drug_name.lower()

    cached = cache_get("resolve_drug_name", {"drug_name": normalized}, cache_dir)
    if cached is not None:
        return cached

    # Reverse-index hit: any synonym we've already seen maps to its parent ChEMBL ID
    reverse_hit = _lookup_chembl_id_by_name(normalized, cache_dir)
    if reverse_hit is not None:
        cache_set(
            "resolve_drug_name",
            {"drug_name": normalized},
            reverse_hit,
            cache_dir,
            ttl=CACHE_TTL,
        )
        return reverse_hit

    # Step 1: OT search → some ChEMBL ID
    async with _OTSearchClient() as client:
        data = await client._graphql(
            OPEN_TARGETS_BASE_URL, _OT_DRUG_SEARCH_QUERY, {"q": drug_name}
        )

    hits = data["data"]["search"]["hits"]
    drug_hits = [h for h in hits if h["entity"] == "drug"]
    if not drug_hits:
        raise DataSourceError(
            "chembl",
            f"No drug found for '{drug_name}'",
        )
    search_chembl_id = drug_hits[0]["id"]

    # Step 2: Check if this is a salt — follow to parent if so
    async with ChEMBLClient(cache_dir=cache_dir) as client:
        molecule = await client.get_molecule(search_chembl_id)

    parent_id = molecule.parent_chembl_id
    if parent_id and parent_id != search_chembl_id:
        logger.info("Salt detected: %s -> parent %s", search_chembl_id, parent_id)
        canonical_id = parent_id
    else:
        canonical_id = search_chembl_id

    cache_set(
        "resolve_drug_name",
        {"drug_name": normalized},
        canonical_id,
        cache_dir,
        ttl=CACHE_TTL,
    )

    return canonical_id


async def get_all_drug_names(
    chembl_id: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> list[str]:
    """Fetch all known names for a drug from ChEMBL.

    Returns a list where the first element is always pref_name (the
    generic/preferred name), followed by every other synonym from the
    parent molecule and its salt forms, regardless of syn_type
    (TRADE_NAME, RESEARCH_CODE, OTHER, USAN, INN, INN_FRENCH, BAN, FDA,
    JAN, etc.). All names are lowercased.

    For small molecules, trade names often live on salt forms (e.g. HCl,
    HBr) rather than the parent molecule. Fetches the parent molecule's
    synonyms and discovers salt forms via the molecule_hierarchy
    endpoint, collecting all synonyms from both.

    Filters out "component of" entries (combination products).
    Results are cached under namespace "chembl_id_to_names" — one file
    per parent ChEMBL ID containing the names list. The same file also
    serves as the reverse index used by resolve_drug_name.

    Args:
        chembl_id: ChEMBL ID of the parent molecule (e.g. "CHEMBL894").
        cache_dir: Directory for file-based cache.

    Returns:
        Deduplicated list of drug names, pref_name first. All lowercase.
    """
    cached = _load_chembl_names(chembl_id, cache_dir)
    if cached is not None:
        return cached

    all_names: list[str] = []

    async with ChEMBLClient(cache_dir=cache_dir) as client:
        # 1. Get pref_name and trade names from the parent molecule itself
        parent = await client.get_molecule(chembl_id)
        pref_name = parent.pref_name if parent.pref_name else chembl_id.lower()

        all_names.extend(s.molecule_synonym for s in parent.molecule_synonyms)

        # 2. Discover salt forms via molecule_hierarchy
        hierarchy_url = f"{CHEMBL_BASE_URL}/molecule.json"
        try:
            hierarchy_raw = await client._rest_get(
                hierarchy_url,
                params={
                    "molecule_hierarchy__parent_chembl_id": chembl_id,
                    "limit": 50,
                },
            )
        except DataSourceError as e:
            logger.warning("Failed to fetch salt forms for %s: %s", chembl_id, e)
            hierarchy_raw = {}

    salt_molecules = (hierarchy_raw or {}).get("molecules", [])
    for mol in salt_molecules:
        mol_id = mol.get("molecule_chembl_id", "")
        if mol_id == chembl_id:
            continue  # skip the parent, already processed
        for s in mol.get("molecule_synonyms") or []:
            all_names.append(s.get("molecule_synonym", "").lower())

    # 3. Filter out "component of" entries and deduplicate with pref_name first
    filtered = [name for name in all_names if name and "component of" not in name]
    result = [pref_name] + [n for n in list(dict.fromkeys(filtered)) if n != pref_name]

    logger.info("ChEMBL drug names for %s: %d found", chembl_id, len(result))

    # Single per-ChEMBL-ID file. Forward lookup (chembl_id → names) is a
    # direct read; reverse lookup (name → chembl_id) is a scan over all
    # files in this namespace, performed by _lookup_chembl_id_by_name.
    _save_chembl_names(chembl_id, result, cache_dir, ttl=CACHE_TTL)

    return result
