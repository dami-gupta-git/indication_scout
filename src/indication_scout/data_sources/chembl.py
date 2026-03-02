"""ChEMBL API client."""

import logging

from pathlib import Path

from indication_scout.constants import CACHE_TTL, CHEMBL_BASE_URL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_chembl import ATCDescription, MoleculeData
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


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
        """
        url = f"{CHEMBL_BASE_URL}/molecule/{chembl_id}.json"
        try:
            raw = await self._rest_get(url, params={})
        except Exception as e:
            raise DataSourceError(
                self._source_name, f"Unexpected error fetching {chembl_id}: {e}"
            )

        if not isinstance(raw, dict):
            raise DataSourceError(
                self._source_name, f"Unexpected response shape for '{chembl_id}'"
            )

        return MoleculeData(
            molecule_chembl_id=raw["molecule_chembl_id"],
            molecule_type=raw["molecule_type"],
            max_phase=raw["max_phase"],
            atc_classifications=raw.get("atc_classifications") or [],
            black_box_warning=raw["black_box_warning"],
            first_approval=raw.get("first_approval"),
            oral=raw["oral"],
        )
