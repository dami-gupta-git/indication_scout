"""Scaffold a Layer 2 regression spec from a gold_standard snapshot.

Reads a frozen `tests/regression/gold_standard/<drug>_<date>.json`
(`SupervisorOutput`) and emits a starter `tests/regression/specs/<drug>.yaml`
pre-filled with every extractable invariant:

  - ranked_order            <- top_diseases (in order)
  - candidate_set_contains  <- candidate_diseases
  - required_ncts_surfaced  <- clinical_trials.relevant_nct_ids (per ranked disease)
  - required_pmids_cited    <- evidence_summary supporting + contradicting (per ranked disease)

The scaffold is a STARTING POINT, not a finished spec. It cannot make the
domain calls a good spec needs — which NCTs/PMIDs are load-bearing anchors vs.
incidental, which demotions matter, which phrases to forbid. After generating:

  1. Prune required_ncts_surfaced / required_pmids_cited to 2-3 high-signal
     anchors per disease (relevant_nct_ids can run to dozens).
  2. Trim candidate_set_contains to the diseases that must not silently drop.
  3. Fill in the commented forbidden_in_ranked / forbidden_phrases stubs if the
     drug has a known demotion (e.g. combination-product) or factual guard.

Then verify:  pytest tests/regression/layer2_structural/test_per_drug.py \
                     -m regression_layer2 -k <drug>

Usage:
    python scripts/scaffold_regression_spec.py <drug>
    python scripts/scaffold_regression_spec.py <drug> --snapshot path/to.json
    python scripts/scaffold_regression_spec.py <drug> --ncts 3 --force --stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scaffold_regression_spec")

REGRESSION_DIR = Path(__file__).resolve().parent.parent / "tests" / "regression"
GOLD_STANDARD_DIR = REGRESSION_DIR / "gold_standard"
SPECS_DIR = REGRESSION_DIR / "specs"

# How many relevant NCTs to seed per disease. relevant_nct_ids can run to
# dozens (metformin/PCOS has 70+); seed a handful and let the author prune.
DEFAULT_NCTS_PER_DISEASE = 3


def _resolve_snapshot(drug: str, explicit: str | None) -> Path:
    """The gold_standard snapshot for `drug`, or an explicit path."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"snapshot not found: {p}")
        return p
    candidates = sorted(GOLD_STANDARD_DIR.glob(f"{drug}_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no gold_standard/{drug}_*.json found. Drop the SupervisorOutput "
            f"snapshot there first, or pass --snapshot."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"expected exactly one gold_standard/{drug}_*.json, found "
            f"{[p.name for p in candidates]}. Pass --snapshot to disambiguate."
        )
    return candidates[0]


def _yaml_str_list(items: list[str], indent: int) -> list[str]:
    """Render a list of strings as indented YAML `- item` lines (quoted)."""
    pad = " " * indent
    return [f'{pad}- "{i}"' if _needs_quote(i) else f"{pad}- {i}" for i in items]


def _needs_quote(s: str) -> bool:
    # Quote purely-numeric strings (PMIDs) so YAML keeps them as strings.
    return s.isdigit()


def _finding_by_disease(report: dict, disease: str) -> dict | None:
    for f in report.get("disease_findings") or []:
        if f.get("disease") == disease:
            return f
    return None


def _relevant_ncts(finding: dict, limit: int) -> list[str]:
    ct = finding.get("clinical_trials") or {}
    return list(ct.get("relevant_nct_ids") or [])[:limit]


def _cited_pmids(finding: dict) -> list[str]:
    es = (finding.get("literature") or {}).get("evidence_summary") or {}
    supporting = es.get("supporting_pmids") or []
    contradicting = es.get("contradicting_pmids") or []
    # Order-preserving dedup: supporting first, then any new contradicting.
    return list(dict.fromkeys([*supporting, *contradicting]))


def build_spec_yaml(drug: str, report: dict, snapshot_name: str, ncts_per_disease: int) -> str:
    top = report.get("top_diseases") or []
    candidates = report.get("candidate_diseases") or []

    lines: list[str] = []
    lines.append(f"# {drug.capitalize()} regression spec.")
    lines.append("#")
    lines.append(f"# Scaffolded from gold_standard/{snapshot_name} by")
    lines.append("# scripts/scaffold_regression_spec.py. PRUNE before committing — the")
    lines.append("# NCT/PMID/candidate lists are seeded from the snapshot and should be")
    lines.append("# trimmed to the high-signal invariants that must not silently flip.")
    lines.append("#")
    lines.append("# Bucket values must match `tests/regression/failure_buckets.Bucket`.")
    lines.append("")
    lines.append(f"drug: {drug}")
    lines.append("")

    # ranked_order — top_diseases, in order.
    lines.append("ranked_order:")
    lines.append("  bucket: ranking")
    lines.append("  indications:")
    lines.extend(_yaml_str_list(top, indent=4))
    lines.append("")

    # candidate_set_contains — seed with all candidates; author trims.
    lines.append("candidate_set_contains:")
    lines.append("  bucket: structural_integrity")
    lines.append("  indications:")
    lines.extend(_yaml_str_list(candidates, indent=4))
    lines.append("")

    # required_ncts_surfaced — per ranked disease, from relevant_nct_ids.
    lines.append("required_ncts_surfaced:")
    any_ncts = False
    for disease in top:
        finding = _finding_by_disease(report, disease)
        if finding is None:
            continue
        ncts = _relevant_ncts(finding, ncts_per_disease)
        if not ncts:
            continue
        any_ncts = True
        lines.append("  - bucket: literature_coverage")
        lines.append(f"    indication: {disease}")
        lines.append("    section: relevant")
        lines.append("    ncts:")
        lines.extend(_yaml_str_list(ncts, indent=6))
    if not any_ncts:
        lines.append("  []  # no relevant_nct_ids in any ranked disease")
    lines.append("")

    # required_pmids_cited — per ranked disease, from evidence_summary.
    lines.append("required_pmids_cited:")
    any_pmids = False
    for disease in top:
        finding = _finding_by_disease(report, disease)
        if finding is None:
            continue
        pmids = _cited_pmids(finding)
        if not pmids:
            continue
        any_pmids = True
        lines.append("  - bucket: literature_coverage")
        lines.append(f"    indication: {disease}")
        lines.append("    mode: cited")
        lines.append("    pmids:")
        lines.extend(_yaml_str_list(pmids, indent=6))
    if not any_pmids:
        lines.append("  []  # no cited PMIDs in any ranked disease")
    lines.append("")

    # Stubs the scaffolder can't infer — fill in by hand if applicable.
    lines.append("# forbidden_in_ranked:  # diseases that must stay demoted (e.g. combination-product)")
    lines.append("#   - bucket: demotion_logic")
    lines.append("#     indication: <disease>")
    lines.append("")
    lines.append("# forbidden_phrases:  # factual guards")
    lines.append("#   - bucket: factual_accuracy")
    lines.append('#     phrase: "<phrase that must not appear>"')
    lines.append("#     scope: summary  # summary | blurb | anywhere")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("drug", help="drug name, e.g. bupropion")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="explicit path to the gold_standard JSON (default: glob by drug)",
    )
    parser.add_argument(
        "--ncts",
        type=int,
        default=DEFAULT_NCTS_PER_DISEASE,
        help=f"relevant NCTs to seed per disease (default: {DEFAULT_NCTS_PER_DISEASE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing specs/<drug>.yaml",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print to stdout instead of writing the spec file",
    )
    args = parser.parse_args()

    drug = args.drug.strip().lower()
    snapshot = _resolve_snapshot(drug, args.snapshot)
    report = json.loads(snapshot.read_text())

    spec_yaml = build_spec_yaml(drug, report, snapshot.name, args.ncts)

    if args.stdout:
        sys.stdout.write(spec_yaml)
        return 0

    out_path = SPECS_DIR / f"{drug}.yaml"
    if out_path.exists() and not args.force:
        logger.error("%s already exists; pass --force to overwrite", out_path)
        return 1

    out_path.write_text(spec_yaml)
    logger.info("wrote %s", out_path)
    logger.info("Next: prune the NCT/PMID/candidate lists, then verify with:")
    logger.info(
        "  pytest tests/regression/layer2_structural/test_per_drug.py "
        "-m regression_layer2 -k %s",
        drug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
