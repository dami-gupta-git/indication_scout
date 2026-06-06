"""IndicationScout CLI entry point.

Exposes the `scout` command declared in `pyproject.toml` under `[project.scripts]`.

Usage:
    scout find -d <drug> [--out-dir DIR] [--no-write]
    scout render -i <payload.json> [--out-dir DIR] [--no-write]
"""

import asyncio
import logging
import os
from datetime import date, datetime
from pathlib import Path

import click
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "snapshots"
TEST_REPORTS_DIR = PROJECT_ROOT / "test_reports"

# Manually set this to point at a different constants file (e.g. ".env.constants.experiment").
# Path is resolved relative to PROJECT_ROOT.
CONSTANTS_FILE = ".env.constants.test"


def _load_env() -> None:
    """Load .env files before importing modules that read settings at import time."""
    load_dotenv(PROJECT_ROOT / ".env")
    # Shell-level CONSTANTS_FILE wins over the module-level default so the CLI
    # and the regression test can be pointed at the same constants.
    constants_file = os.environ.get("CONSTANTS_FILE", CONSTANTS_FILE)
    constants_path = PROJECT_ROOT / constants_file
    load_dotenv(constants_path)
    # Also export so config.py's pydantic Settings picks up the same file.
    os.environ["CONSTANTS_FILE"] = str(constants_path)


async def _run_for_drug(
    drug: str, out_dir: Path, write: bool, date_before: date | None = None
) -> None:
    # Imports are deferred until after _load_env() runs in main(), because
    # base_client.py calls get_settings() at import time.
    from indication_scout.helpers.drug_helpers import normalize_drug_name
    from indication_scout.services.analysis_runner import run_analysis
    from indication_scout.tracing import setup_tracing, shutdown_tracing

    setup_tracing()
    try:
        # Normalize at the entry point so filenames/logs below see the same lowercased form
        # run_analysis derives internally (normalize_drug_name is idempotent).
        drug = normalize_drug_name(drug)

        logger.info("Starting %s (date_before=%s)", drug, date_before)
        output, report_md = await run_analysis(drug, date_before=date_before)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Persist the SupervisorOutput payload so the report can be re-rendered later
        # via `scout render` without re-running the pipeline. Skip for holdout runs.
        if date_before is None:
            TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            payload_path = TEST_REPORTS_DIR / f"{drug}_{timestamp}.json"
            payload_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Saved payload -> %s", payload_path)
            click.echo(f"Payload:   {payload_path}")

        if not write:
            click.echo(report_md)
            return

        write_dir = out_dir / "holdouts" if date_before else out_dir
        write_dir.mkdir(parents=True, exist_ok=True)
        cutoff_tag = f"_holdout_{date_before.isoformat()}" if date_before else ""
        md_path = write_dir / f"{drug}{cutoff_tag}_{timestamp}.md"
        if date_before is not None:
            banner = f"> **HOLDOUT** — date_before={date_before.isoformat()}\n\n"
            report_md = banner + report_md
        md_path.write_text(report_md, encoding="utf-8")
        logger.info("Finished %s -> %s", drug, md_path)
        click.echo(f"Report:    {md_path}")

    finally:
        shutdown_tracing()


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """IndicationScout — agentic drug repurposing."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Quiet third-party per-request chatter that drowns out our own banners.
    # Keep WARNING+ so genuine failures still surface.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@cli.command()
@click.option(
    "-d",
    "--drug",
    required=True,
    help="Drug name to run the supervisor pipeline on.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUT_DIR,
    show_default=True,
    help="Directory to write the report into.",
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Print the markdown report to stdout instead of writing to disk.",
)
@click.option(
    "--date-before",
    "date_before",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help=(
        "Temporal cutoff (YYYY-MM-DD). PubMed and ClinicalTrials.gov queries are restricted to "
        "records dated strictly before this date. Mechanism (OpenTargets) data has no date "
        "filter and is always current."
    ),
)
def find(
    drug: str, out_dir: Path, no_write: bool, date_before: datetime | None
) -> None:
    """Run the supervisor pipeline on DRUG and produce a repurposing report."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise click.ClickException(
            "ANTHROPIC_API_KEY is not set. Add it to your .env or environment."
        )
    cutoff = date_before.date() if date_before is not None else None
    asyncio.run(_run_for_drug(drug, out_dir, write=not no_write, date_before=cutoff))


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a SupervisorOutput JSON payload saved by `scout find`.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=TEST_REPORTS_DIR,
    show_default=True,
    help="Directory to write the rendered markdown report into.",
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Print the markdown report to stdout instead of writing to disk.",
)
def render(input_path: Path, out_dir: Path, no_write: bool) -> None:
    """Re-render a markdown report from a saved SupervisorOutput JSON payload."""
    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
    from indication_scout.report.format_report import format_report

    output = SupervisorOutput.model_validate_json(input_path.read_text(encoding="utf-8"))
    report_md = format_report(output)

    if no_write:
        click.echo(report_md)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{input_path.stem}.md"
    md_path.write_text(report_md, encoding="utf-8")
    logger.info("Rendered %s -> %s", input_path, md_path)
    click.echo(f"Report:    {md_path}")


@cli.command("diff-report")
@click.argument("golden", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("current", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def diff_report(golden: Path, current: Path) -> None:
    """Diff two SupervisorOutput JSON snapshots and print the result.

    Both arguments must be paths to JSON files produced by serializing a
    SupervisorOutput (`model_dump_json`). The diff is the same one the
    regression test runs, so a clean output here means the regression test
    would pass.
    """
    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
    from indication_scout.regression.harness import (
        compare_reports,
        has_errors,
        render_diffs,
    )

    g = SupervisorOutput.model_validate_json(golden.read_text())
    c = SupervisorOutput.model_validate_json(current.read_text())
    diffs = compare_reports(g, c)
    click.echo(render_diffs(diffs))
    if has_errors(diffs):
        raise click.exceptions.Exit(code=1)


def main() -> None:
    """Console-script entry point referenced by `pyproject.toml`."""
    _load_env()
    cli()


if __name__ == "__main__":
    main()
