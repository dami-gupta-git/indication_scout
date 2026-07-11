"""Full-pipeline regression test against a committed gold-standard SupervisorOutput.

Marked `regression` and excluded from the default `pytest` run via the
addopts marker filter in `pytest.ini`. To run:

    pytest -m regression                                # replay (default mode)
    SCOUT_CASSETTE_MODE=record  pytest -m regression    # re-record
    SCOUT_CASSETTE_MODE=live    pytest -m regression    # bypass cassette

The test requires a live database connection (Postgres + pgvector) just like
the rest of the integration suite — the cassette only stubs external HTTP /
LLM traffic.

The pinned snapshot is the datestamped `gold_standard/<drug>_*.json` file —
the same frozen `SupervisorOutput` the Layer 2 spec harness asserts against.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from indication_scout.regression.harness import compare_reports, has_errors, render_diffs

from tests.regression.cassette import use_cassette
from tests.regression.constants import CASSETTE_DIR, GOLD_STANDARD_DIR

PINNED_DRUGS = ["bupropion"]


def _gold_path(drug: str) -> Path | None:
    """The single frozen gold_standard/<drug>_*.json snapshot, or None.

    Raises if more than one matches — an ambiguous snapshot is a setup error,
    not something to silently pick from.
    """
    candidates = sorted(GOLD_STANDARD_DIR.glob(f"{drug}_*.json"))
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"expected at most one gold_standard/{drug}_*.json, "
            f"found {[p.name for p in candidates]}"
        )
    return candidates[0] if candidates else None


@pytest.mark.regression
@pytest.mark.parametrize("drug", PINNED_DRUGS)
async def test_pipeline_regression(drug: str) -> None:
    gold_path = _gold_path(drug)
    cassette_path = CASSETTE_DIR / drug / "pipeline.yaml"
    record_mode = os.environ.get("SCOUT_CASSETTE_MODE", "").lower() == "record"

    # Skip only when we have nothing to compare against AND aren't in record
    # mode. Record mode is the bootstrap path — it produces the snapshot, so a
    # missing file is expected, not a reason to skip.
    if gold_path is None and not record_mode:
        pytest.skip(
            f"No gold-standard snapshot at {GOLD_STANDARD_DIR}/{drug}_*.json. "
            f"Record one with SCOUT_CASSETTE_MODE=record pytest -m regression -k {drug}"
        )

    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput

    current = await _run_pipeline(drug, cassette_path)

    # In record mode, overwrite the existing snapshot with the freshly recorded
    # run so cassette + snapshot stay in lockstep. This is the only place that
    # writes to the gold_standard directory; the test still runs the comparison
    # after so a record-mode run can catch a structural regression. Record mode
    # requires an existing datestamped snapshot to overwrite — bootstrapping a
    # brand-new drug's snapshot filename is a manual step.
    if record_mode:
        if gold_path is None:
            raise FileNotFoundError(
                f"record mode needs an existing gold_standard/{drug}_*.json to "
                f"overwrite; create the datestamped snapshot file first"
            )
        gold_path.write_text(current.model_dump_json(indent=2))
        golden = current
    else:
        golden = SupervisorOutput.model_validate_json(gold_path.read_text())

    diffs = compare_reports(golden, current)
    error_diffs = [d for d in diffs if d.severity == "error"]
    assert not has_errors(diffs), (
        f"\nRegression detected for {drug}:\n{render_diffs(error_diffs)}\n"
        f"(Run `scout diff-report {gold_path} <current.json>` for a full diff.)"
    )


async def _run_pipeline(drug: str, cassette_path: Path):
    """Run the full supervisor pipeline under the cassette context.

    Mirrors `_run_for_drug` in cli/cli.py but skips the markdown/file output —
    we want the SupervisorOutput object directly.
    """
    from langchain_anthropic import ChatAnthropic

    from indication_scout.agents.supervisor.supervisor_agent import (
        build_supervisor_agent,
        run_supervisor_agent,
    )
    from indication_scout.constants import DEFAULT_CACHE_DIR
    from indication_scout.db.session import get_db
    from indication_scout.helpers.drug_helpers import normalize_drug_name
    from indication_scout.services.retrieval import RetrievalService

    drug = normalize_drug_name(drug)
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=4096,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "test-key-unused-in-replay"),
    )
    db = next(get_db())
    svc = RetrievalService(DEFAULT_CACHE_DIR)

    with use_cassette(cassette_path):
        agent, get_merged_allowlist, get_auto_findings, get_approval_labels = (
            build_supervisor_agent(llm=llm, svc=svc, db=db, date_before=None)
        )
        return await run_supervisor_agent(
            agent,
            get_merged_allowlist,
            drug,
            get_auto_findings=get_auto_findings,
            get_approval_labels=get_approval_labels,
        )
