import asyncio

from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.markers import no_review


@no_review
def _expand(drug_name: str) -> None:
    pass


@no_review
async def run_candidate(drug_name: str) -> None:
    async with OpenTargetsClient() as client:
        chembl_id = await resolve_drug_name(drug_name, client.cache_dir)
        x = await client.get_drug(chembl_id)

    print(x)


if __name__ == "__main__":
    asyncio.run(run_candidate("bupropion"))
