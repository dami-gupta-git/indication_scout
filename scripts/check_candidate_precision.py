"""Score candidate-selection precision from the newest live reports."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.services.precision_metrics import (
    CandidatePrecisionSpec,
    filter_spec_to_drugs,
    load_candidate_precision_spec,
    score_ranked_candidate_precision,
    write_candidate_precision_report,
)

logger = logging.getLogger("check_candidate_precision")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--drugs",
        nargs="+",
        help="Score only these drugs (default: every drug in the label file)",
    )
    return parser


def _latest_reports(
    reports_dir: Path, spec: CandidatePrecisionSpec
) -> tuple[list[SupervisorOutput], list[Path]]:
    reports: list[SupervisorOutput] = []
    paths: list[Path] = []
    drugs = sorted({label.drug.lower().strip() for label in spec.labels})
    for drug in drugs:
        candidates = sorted(
            reports_dir.glob(f"{drug}_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise ValueError(f"no generated report found for {drug!r}")
        path = candidates[0]
        report = SupervisorOutput.model_validate_json(path.read_text(encoding="utf-8"))
        if report.drug_name.lower().strip() != drug:
            raise ValueError(
                f"report {path.name!r} contains drug {report.drug_name!r}, expected {drug!r}"
            )
        reports.append(report)
        paths.append(path)
    return reports, paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args()
    spec = load_candidate_precision_spec(args.labels)
    try:
        if args.drugs:
            spec = filter_spec_to_drugs(spec, args.drugs)
        reports, report_paths = _latest_reports(args.reports_dir, spec)
        result = score_ranked_candidate_precision(reports, spec)
    except ValueError as error:
        logger.error("Candidate precision unavailable: %s", error)
        raise SystemExit(2) from error

    write_candidate_precision_report(result, report_paths, args.output)
    logger.info(
        "Candidate precision@%d: %d/%d (%.1f%%); threshold %.1f%%",
        result.top_k,
        result.valid,
        result.valid + result.invalid,
        result.precision * 100,
        result.minimum_precision * 100,
    )
    logger.info("Wrote %s", args.output)
    if result.precision < result.minimum_precision:
        logger.error("Candidate precision is below the CI threshold")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
