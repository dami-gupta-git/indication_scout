"""Failure-mode taxonomy for the regression suite.

Every assertion in `layer2_structural/assertions.py` is tagged with a bucket.
When an assertion fails, the bucket is attached to the resulting `BucketedDiff`
so failure distributions can roll up over time. The taxonomy lines up with the
failure-mode buckets used in the bioRxiv submission, so the same artifact
serves regression hygiene and the paper's failure-mode analysis.

Adding a new bucket: add it to `Bucket` and document what kind of failure it
represents. Specs reference buckets by string; the loader validates that the
string matches the enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Severity = Literal["error", "warn", "info"]


class Bucket(str, Enum):
    """The failure-mode taxonomy.

    Keep this list compact — every new bucket dilutes the rollup. If a new
    failure type doesn't fit cleanly into an existing bucket, that's a signal
    worth a conversation, not an automatic addition.
    """

    LITERATURE_COVERAGE = "literature_coverage"
    # Required PMID or NCT didn't appear in the relevant section. The signal
    # is in the upstream data; the supervisor failed to surface it.

    RANKING = "ranking"
    # An indication that should be in the top-N ranked signals isn't, or one
    # that shouldn't be is. The candidate set may be fine; the rank ordering
    # logic dropped or promoted incorrectly.

    DEMOTION_LOGIC = "demotion_logic"
    # A demotion that should fire didn't, or vice versa. Includes combination-
    # product (bupropion × obesity), parent-indication (T2DM under rosi), and
    # narrower-approval (sildenafil × Eisenmenger) cases.

    EVIDENCE_GATE = "evidence_gate"
    # A candidate that should have been gated out (zero trials AND weak/no
    # literature) survived, or a legitimate untested-but-rationale-supported
    # candidate got incorrectly gated.

    TERMINATION_CLASSIFICATION = "termination_classification"
    # A terminated trial's stop reason was classified into the wrong bucket
    # (operational / efficacy_futility / business / enrollment / safety).

    FACTUAL_ACCURACY = "factual_accuracy"
    # A forbidden phrase appeared (e.g. "approved for obesity" on bupropion
    # monotherapy), or a known-true fact is contradicted in the report.

    STRUCTURAL_INTEGRITY = "structural_integrity"
    # Required nested field missing, summary empty, invariant violated. The
    # report's shape itself is wrong, independent of content.


@dataclass(frozen=True)
class BucketedDiff:
    """A failed assertion, tagged with its failure-mode bucket.

    `path` is a JSON-pointer-ish locator into the SupervisorOutput.
    `detail` is the human-readable explanation rendered into the failure
    message and bucket rollup.
    """

    bucket: Bucket
    path: str
    severity: Severity
    detail: str
    spec_ref: str = ""
    # Optional pointer back to the YAML spec entry that produced this diff,
    # so when a Layer 2 test fails you can navigate from the assertion error
    # to the line in the spec that drove it.


def has_errors(diffs: list[BucketedDiff]) -> bool:
    return any(d.severity == "error" for d in diffs)


def render(diffs: list[BucketedDiff]) -> str:
    """Compact rendering for pytest failure output and the rollup CLI."""
    if not diffs:
        return "no diffs"
    lines = [f"{'SEV':<6} {'BUCKET':<28} {'PATH':<48} DETAIL"]
    for d in diffs:
        lines.append(
            f"{d.severity:<6} {d.bucket.value:<28} {d.path:<48} {d.detail}"
        )
    return "\n".join(lines)


def summarize_buckets(diffs: list[BucketedDiff]) -> dict[Bucket, int]:
    """Count error-severity diffs by bucket. Drives the bioRxiv rollup."""
    counts: dict[Bucket, int] = {}
    for d in diffs:
        if d.severity != "error":
            continue
        counts[d.bucket] = counts.get(d.bucket, 0) + 1
    return counts
