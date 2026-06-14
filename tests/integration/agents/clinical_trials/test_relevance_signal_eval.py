"""Gated eval: rich-field relevance signal beats MeSH on the labeled pair.

Promoted from `scratch/trial_relevance_harness.py`. Asserts the premise behind
2b (PLAN_pertrial_relevance.md): when the LLM classifies each trial relevant vs
contaminated, feeding title + interventions + brief_summary is at least as
accurate as feeding MeSH conditions, and covers every trial. The 68 trials and
their hand labels (sildenafil × SYSTEMIC hypertension; the registry over-recalls
pulmonary hypertension / PAH and other-drug trials) are the oracle.

Hits the real LLM — opt-in via `-m live` (excluded from the default run).

Run: pytest -m live tests/integration/agents/clinical_trials/test_relevance_signal_eval.py
"""

import json
import logging
from pathlib import Path

import pytest

from indication_scout.services.llm import parse_last_json_object, query_llm

from ._sild_htn_labels import LABELS

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live

# Field snapshot the labels were assigned against (mesh/interv/title/summary per nct).
_DATA_PATH = Path(__file__).parents[4] / "scratch" / "sild_htn_trials.json"

SYSTEM = (
    "You are a clinical-trials relevance classifier. The candidate repurposing "
    "indication is SYSTEMIC HYPERTENSION (ordinary high blood pressure). The "
    "registry search over-recalls: it returns trials for PULMONARY hypertension / "
    "PAH (a DISTINCT disease) and trials whose primary drug is NOT sildenafil. "
    "For EACH trial id given, output a verdict: 'relevant' if it studies SILDENAFIL "
    "for SYSTEMIC hypertension, else 'contaminated'. You MUST return a verdict for "
    "every id — omit none. Return JSON: "
    '{"verdicts": [{"nct": str, "verdict": "relevant"|"contaminated"}]}.'
)


def _load() -> list[dict]:
    return json.loads(_DATA_PATH.read_text())


def _render(rows: list[dict], condition: str) -> str:
    lines = []
    for r in rows:
        if condition == "mesh":
            lines.append(
                f"{r['nct']} | phase {r['phase']} | "
                f"mesh: {'; '.join(r['mesh']) or '(none)'}"
            )
        else:  # rich
            lines.append(
                f"{r['nct']} | phase {r['phase']} | "
                f"drugs: {'; '.join(r['interv']) or '(none)'} | "
                f"{r['title']} | {r['summary']}"
            )
    return "\n".join(lines)


async def _score(rows: list[dict], condition: str) -> tuple[float, float]:
    """Return (coverage, accuracy) of the LLM's verdicts vs LABELS."""
    prompt = f"Classify these {len(rows)} trials:\n\n{_render(rows, condition)}"
    text = await query_llm(prompt, system=SYSTEM)
    parsed = parse_last_json_object(text)
    assert parsed is not None, f"LLM returned no JSON object for {condition}"
    verdicts = {v["nct"]: v["verdict"] for v in parsed["verdicts"]}

    all_ncts = {r["nct"] for r in rows}
    tagged = set(verdicts) & all_ncts
    coverage = len(tagged) / len(all_ncts)
    scored = [n for n in all_ncts if n in verdicts]
    correct = sum(
        (verdicts[n] == "relevant") == (LABELS[n] == "R") for n in scored
    )
    accuracy = correct / len(scored) if scored else 0.0
    return coverage, accuracy


async def test_rich_signal_at_least_as_accurate_as_mesh_with_full_coverage():
    """The rich (title + interventions + summary) signal classifies every trial
    and is at least as accurate as the MeSH signal on the labeled pair. This is
    the empirical basis for dropping MeSH from the classification view (2b)."""
    rows = _load()
    assert len(rows) == len(LABELS)

    mesh_cov, mesh_acc = await _score(rows, "mesh")
    rich_cov, rich_acc = await _score(rows, "rich")

    logger.info(
        "relevance eval — mesh: cov=%.0f%% acc=%.0f%%  rich: cov=%.0f%% acc=%.0f%%",
        mesh_cov * 100,
        mesh_acc * 100,
        rich_cov * 100,
        rich_acc * 100,
    )

    # Forced per-trial verdict → full coverage for the rich signal.
    assert rich_cov == 1.0
    # Rich must not be worse than MeSH; the harness measured 99% vs 94%.
    assert rich_acc >= mesh_acc
