"""Check whether querying by drug alias (brand vs INN vs other synonyms) changes
CT.gov / PubMed results.

Resolves the drug's ChEMBL id, pulls all known aliases (INN, trade names, USAN,
BAN, etc.), then runs CT.gov's drug-only trial sweep AND a PubMed search
separately per alias. Reports, per source:
  - NCT ids / PMIDs returned by each alias
  - which ids are alias-exclusive (only found via that name, missed by others)
  - a "union vs single-alias-as-typed" delta, to show what a user typing just
    one name (e.g. the brand) would miss vs a canonicalized/expanded query.

Investigation only — does not change any query logic in the app.

Run:
    python scripts/alias_query_impact.py wegovy
    python scripts/alias_query_impact.py wegovy --indication "type 2 diabetes mellitus"
"""

import argparse
import asyncio
import logging

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.helpers.drug_helpers import normalize_drug_name

logging.basicConfig(level=logging.WARNING)

CT_MAX_FETCH = 500
PUBMED_MAX_RESULTS = 200


async def ct_trials_for_alias(client: ClinicalTrialsClient, alias: str) -> set[str]:
    trials, _saturated = await client._paginated_search(
        drug=alias,
        indication=None,
        max_results=CT_MAX_FETCH,
        sort="EnrollmentCount:desc",
    )
    return {t.nct_id for t in trials}


async def pubmed_pmids_for_alias(client: PubMedClient, alias: str) -> set[str]:
    pmids = await client.search(query=alias, max_results=PUBMED_MAX_RESULTS)
    return set(pmids)


def _print_set_report(title: str, per_alias: dict[str, set[str]], typed_alias: str) -> None:
    union = set().union(*per_alias.values()) if per_alias else set()
    typed_set = per_alias.get(typed_alias, set())

    print(f"\n--- {title} ---")
    for alias, ids in per_alias.items():
        print(f"  {alias!r:<30} -> {len(ids):>4} ids")
    print(f"  {'UNION (all aliases)':<30} -> {len(union):>4} ids")

    missed = union - typed_set
    print(f"\n  As typed ({typed_alias!r}): {len(typed_set)} ids")
    print(f"  Missed by typing only {typed_alias!r}: {len(missed)} ids"
          f" ({len(missed) / len(union):.0%} of union)" if union else "")

    if missed:
        # Which alias(es) uniquely found each missed id
        contributors: dict[str, list[str]] = {}
        for alias, ids in per_alias.items():
            if alias == typed_alias:
                continue
            for i in ids & missed:
                contributors.setdefault(i, []).append(alias)
        sample = list(contributors.items())[:15]
        print(f"  Sample of ids missed (id -> found via alias):")
        for i, aliases in sample:
            print(f"    {i:<15} <- {aliases}")
        if len(contributors) > len(sample):
            print(f"    ... and {len(contributors) - len(sample)} more")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drug", help="Drug name as a user would type it (e.g. 'wegovy')")
    parser.add_argument(
        "--aliases",
        nargs="*",
        default=None,
        help="Override alias list instead of resolving via ChEMBL (e.g. --aliases wegovy semaglutide ozempic)",
    )
    args = parser.parse_args()

    typed = normalize_drug_name(args.drug)

    if args.aliases:
        aliases = [normalize_drug_name(a) for a in args.aliases]
        if typed not in aliases:
            aliases.append(typed)
    else:
        chembl_id = await resolve_drug_name(typed, cache_dir=DEFAULT_CACHE_DIR)
        resolved_aliases = await get_all_drug_names(chembl_id, cache_dir=DEFAULT_CACHE_DIR)
        aliases = list(dict.fromkeys([typed] + [normalize_drug_name(a) for a in resolved_aliases]))
        print(f"Resolved ChEMBL id: {chembl_id}")

    print(f"Typed drug name: {typed!r}")
    print(f"Aliases to compare ({len(aliases)}): {aliases}")

    async with ClinicalTrialsClient() as ct_client:
        ct_results = {}
        for alias in aliases:
            ct_results[alias] = await ct_trials_for_alias(ct_client, alias)

    async with PubMedClient() as pm_client:
        pm_results = {}
        for alias in aliases:
            pm_results[alias] = await pubmed_pmids_for_alias(pm_client, alias)

    _print_set_report("ClinicalTrials.gov (drug-only, query.intr)", ct_results, typed)
    _print_set_report("PubMed", pm_results, typed)


if __name__ == "__main__":
    asyncio.run(main())
