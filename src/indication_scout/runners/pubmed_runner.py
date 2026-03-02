import asyncio

from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.markers import no_review


@no_review
def _expand(drug_name: str) -> None:
    pass

@no_review
async def run_candidate(drug_name: str) -> None:
    async with OpenTargetsClient() as client:
        x = await client.get_drug(drug_name)

    print(x)


if __name__ == "__main__":
    asyncio.run(run_candidate("bupropion"))
