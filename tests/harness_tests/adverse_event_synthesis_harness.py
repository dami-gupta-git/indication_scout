"""Standalone harness: does the new ADVERSE-EVENT grounding rule (candidate prompt) change the
synthesis verdict, and where — across the whole seed corpus?

The bug it targets (snapshot minoxidil_2026-06-15_23-27-28.md, Diabetes Mellitus): synthesize cited
PMID 6985752 ("Minoxidil." review) as a *contradicting* signal because the abstract lists "diabetes
mellitus" inside an adverse-effect enumeration — a bare AE-list term, not a causal/clinical study.
The candidate prompt (tests/harness_tests/prompts/adverse_event_synthesis_prompt.txt) adds rule #5: an AE-list disease
mention is NOT evidence for/against repurposing and must not anchor the assessment.

WIDE TEST: sweep every disease in every seed_examples/*.json, run BOTH prompts (current
synthesize.txt vs candidate) N times each, and report only the pairs where they DIVERGE — by
modal direction, or by supporting/contradicting PMID sets. This surfaces both the intended fixes
(AE-list → none) and any REGRESSIONS (the rule wrongly suppressing genuine harm evidence, e.g.
minoxidil x atherosclerosis, which is a real contraindication and must stay "contradicts").

Abstracts are read straight from the seed JSON (the same text synthesize saw — no DB / no
embeddings). Each pair's modal verdict across N runs is compared.

Run: .venv/bin/python tests/harness_tests/adverse_event_synthesis_harness.py [model] [runs] [--all]
       --all  print every pair, not just divergent ones
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

settings = get_settings()
client = AsyncAnthropic(api_key=settings.anthropic_api_key)

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = _args[0] if len(_args) > 0 else settings.llm_model
RUNS_PER_CASE = int(_args[1]) if len(_args) > 1 else 3
SHOW_ALL = "--all" in sys.argv
MAX_CONCURRENCY = 6

ROOT = Path(__file__).parent.parent
SEED_DIR = ROOT / "seed_examples"
CURRENT_PROMPT = (ROOT / "src/indication_scout/prompts/synthesize.txt").read_text()
CANDIDATE_PROMPT = (Path(__file__).parent / "prompts" / "adverse_event_synthesis_prompt.txt").read_text()

_sem = asyncio.Semaphore(MAX_CONCURRENCY)


def all_pairs() -> list[tuple[str, str, str, str]]:
    """(seed_file, drug, disease, formatted_abstracts) for every disease with >=1 abstract."""
    pairs = []
    for seed in sorted(SEED_DIR.glob("*.json")):
        d = json.loads(seed.read_text())
        drug = d.get("drug_name")
        if not drug:
            continue
        for f in d.get("disease_findings", []):
            results = (f.get("literature") or {}).get("semantic_search_results") or []
            if not results:
                continue
            formatted = "\n\n".join(
                f"PMID: {r['pmid']}\nTitle: {r['title']}\nAbstract: {r['abstract']}"
                for r in results
            )
            pairs.append((seed.name, drug, f["disease"], formatted))
    return pairs


async def synth(template: str, drug: str, disease: str, abstracts: str) -> dict:
    async with _sem:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": template.format(
                        drug_name=drug, disease_name=disease, abstracts=abstracts
                    ),
                }
            ],
        )
    out = resp.content[0].text.strip()
    if out.startswith("```"):
        out = out.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_parse_fail": True}


def modal(results: list[dict], key: str) -> str:
    return Counter(r.get(key, "PARSE_FAIL") for r in results).most_common(1)[0][0]


def pmid_set(results: list[dict], key: str) -> frozenset:
    """Union of a PMID-list field across runs (stable signal despite run variance)."""
    s: set[str] = set()
    for r in results:
        s.update(r.get(key) or [])
    return frozenset(s)


async def eval_pair(template, drug, disease, abstracts):
    return await asyncio.gather(
        *(synth(template, drug, disease, abstracts) for _ in range(RUNS_PER_CASE))
    )


async def main():
    pairs = all_pairs()
    print(
        f"=== model: {MODEL}  runs/pair: {RUNS_PER_CASE}  "
        f"pairs: {len(pairs)}  (current vs candidate) ===\n"
    )
    n_div = 0
    for seed, drug, disease, abstracts in pairs:
        cur, cand = await asyncio.gather(
            eval_pair(CURRENT_PROMPT, drug, disease, abstracts),
            eval_pair(CANDIDATE_PROMPT, drug, disease, abstracts),
        )
        cur_dir, cand_dir = modal(cur, "direction"), modal(cand, "direction")
        cur_str, cand_str = modal(cur, "strength"), modal(cand, "strength")
        cur_contra, cand_contra = pmid_set(cur, "contradicting_pmids"), pmid_set(cand, "contradicting_pmids")
        cur_supp, cand_supp = pmid_set(cur, "supporting_pmids"), pmid_set(cand, "supporting_pmids")

        diverged = (
            cur_dir != cand_dir
            or cur_str != cand_str
            or cur_contra != cand_contra
            or cur_supp != cand_supp
        )
        if diverged:
            n_div += 1
        if not (diverged or SHOW_ALL):
            continue

        tag = "DIVERGE" if diverged else "same"
        print(f"[{tag}] {drug} x {disease}  ({seed})")
        print(f"        direction:  {cur_dir:>10} -> {cand_dir}")
        print(f"        strength:   {cur_str:>10} -> {cand_str}")
        if cur_contra != cand_contra:
            print(f"        contra:     {sorted(cur_contra)} -> {sorted(cand_contra)}")
        if cur_supp != cand_supp:
            print(f"        support:    {sorted(cur_supp)} -> {sorted(cand_supp)}")
        print()

    print(f"=== {n_div}/{len(pairs)} pairs diverged ===")


if __name__ == "__main__":
    asyncio.run(main())
