"""DrugBank data client."""

from indication_scout.data_sources.base_client import BaseClient


class DrugBankClient(BaseClient):
    """Client for accessing DrugBank data."""

    @property
    def _source_name(self) -> str:
        return "drugbank"

    async def get_drug(self, drug_name: str) -> dict | None:
        """Get drug information by name."""
        raise NotImplementedError

    async def get_interactions(self, drug_id: str) -> list[dict]:
        """Get drug-drug interactions."""
        raise NotImplementedError
