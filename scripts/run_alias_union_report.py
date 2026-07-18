"""Run `scout find` for a drug, but with CT.gov's `query.intr` widened to an
OR of all ChEMBL-known aliases (brand names, INN, synonyms) instead of just
the single name the user typed.

Investigation-only monkeypatch: patches ClinicalTrialsClient._build_search_params
in-process for the duration of this run so we can diff the resulting report
against an unpatched `scout find` run on the same drug. Does not touch any
committed source.

Run:
    python scripts/run_alias_union_report.py wegovy
    python scripts/run_alias_union_report.py sildenafil --date-before 2005-01-01
"""

import argparse
import asyncio
import logging
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_REPORTS_DIR = PROJECT_ROOT / "test_reports"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    constants_file = os.environ.get("CONSTANTS_FILE", ".env.constants")
    constants_path = PROJECT_ROOT / constants_file
    load_dotenv(constants_path)
    os.environ["CONSTANTS_FILE"] = str(constants_path)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drug", help="Drug name as a user would type it (e.g. 'wegovy')")
    parser.add_argument(
        "--date-before",
        type=str,
        default=None,
        help="Holdout cutoff (YYYY-MM-DD). PubMed/CT.gov queries restricted to before this date.",
    )
    args = parser.parse_args()
    date_before = date.fromisoformat(args.date_before) if args.date_before else None

    _load_env()

    from indication_scout.constants import DEFAULT_CACHE_DIR
    from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
    from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
    from indication_scout.helpers.drug_helpers import normalize_drug_name
    from indication_scout.services.analysis_runner import run_analysis

    typed = normalize_drug_name(args.drug)
    chembl_id = await resolve_drug_name(typed, cache_dir=DEFAULT_CACHE_DIR)
    resolved_aliases = await get_all_drug_names(chembl_id, cache_dir=DEFAULT_CACHE_DIR)
    aliases = list(dict.fromkeys([typed] + [normalize_drug_name(a) for a in resolved_aliases]))
    print(f"Widening CT.gov query.intr for {typed!r} to union of: {aliases}")

    or_term = " OR ".join(f'"{a}"' for a in aliases)

    original_build = ClinicalTrialsClient._build_search_params

    def patched_build(self, *, drug=None, **kwargs):
        if drug is not None and normalize_drug_name(drug) == typed:
            drug = or_term
        return original_build(self, drug=drug, **kwargs)

    ClinicalTrialsClient._build_search_params = patched_build

    # Bypass the file cache: cache keys are built from the pre-patch `drug` string
    # ("wegovy"), so a prior wegovy-only run's cached CT.gov results would otherwise
    # be returned unchanged, silently defeating the alias-union patch above.
    scratch_cache_dir = PROJECT_ROOT / "scratch_cache_alias_union"
    original_init = ClinicalTrialsClient.__init__

    def patched_init(self, cache_dir=None, **kwargs):
        original_init(self, cache_dir=scratch_cache_dir, **kwargs)

    ClinicalTrialsClient.__init__ = patched_init

    try:
        output, report_md = await run_analysis(typed, date_before=date_before)
    finally:
        ClinicalTrialsClient._build_search_params = original_build
        ClinicalTrialsClient.__init__ = original_init

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout_tag = f"_holdout_{date_before.isoformat()}" if date_before else ""
    payload_path = TEST_REPORTS_DIR / f"{typed}_ALIASUNION{holdout_tag}_{timestamp}.json"
    payload_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    md_path = TEST_REPORTS_DIR / f"{typed}_ALIASUNION{holdout_tag}_{timestamp}.md"
    if date_before is not None:
        report_md = f"> **HOLDOUT** — date_before={date_before.isoformat()}\n\n" + report_md
    md_path.write_text(report_md, encoding="utf-8")
    print(f"Payload: {payload_path}")
    print(f"Report:  {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
