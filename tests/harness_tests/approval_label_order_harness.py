"""Compare the approval prompt and a coverage-first variant against both FDA excerpt orders.

Run: .venv/bin/python tests/harness_tests/approval_label_order_harness.py --runs 3
"""

import argparse
import asyncio
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from indication_scout.services.approval_check import _parse_approval_decisions
from indication_scout.services.llm import _model, parse_last_json_object, query_llm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SAVED_LABELS = (
    ROOT / "cache/fda_label_indications"
    / "2ab7bd5c2e8db017df555f05e5522a0038021407efcd67598ef666cfd067c765.json"
)
PROMPT_PATH = ROOT / "src/indication_scout/prompts/extract_fda_approval_single.txt"
CANDIDATE = "eisenmenger syndrome"
REASON_RULE = "The reason must be one sentence."
COVERAGE_RULE = (
    "For every candidate, including a candidate ultimately labelled none, write the reason before the label. "
    "In one short sentence, quote the closest indication from the FDA label text, give a short phrase explaining "
    "why it could cover this candidate (or state why no credible coverage case exists), and quote any explicit "
    "label restriction that rules out coverage. If no such restriction is stated, say so. "
    "If there is no relevant indication to quote, say so. Then write the verdict as the final field."
)


async def replay(prompt: str, indications: list[str]) -> tuple[str | None, str | None]:
    parsed = parse_last_json_object(await query_llm(prompt))
    if parsed is None:
        return None, None
    verdict = _parse_approval_decisions(parsed, [CANDIDATE], indications).get(CANDIDATE)
    for name, decision in parsed.items():
        if name.lower() == CANDIDATE and isinstance(decision, dict):
            reason = decision.get("reason")
            return verdict, reason if isinstance(reason, str) else None
    return verdict, None


async def run(label_cache: Path, runs: int) -> int:
    record = json.loads(label_cache.read_text())
    excerpts: list[str] = record["params"]["label_texts"]
    indications: list[str] = record["data"]
    if not excerpts or not indications:
        raise ValueError("The replay requires saved FDA excerpts and extracted indications.")
    baseline = PROMPT_PATH.read_text()
    if baseline.count(REASON_RULE) != 1:
        raise ValueError("The production prompt's reason instruction has changed.")
    coverage_first = baseline.replace(REASON_RULE, COVERAGE_RULE, 1)

    prompts: dict[str, str] = {}
    for variant, template in (("baseline", baseline), ("coverage-first", coverage_first)):
        for order, labels in (("original", excerpts), ("reversed", excerpts[::-1])):
            prompts[f"{variant}/{order}"] = template.format(
                label_texts="\n---\n".join(labels),
                approved_indications=json.dumps(indications),
                candidate_diseases=json.dumps([CANDIDATE]),
            )

    logger.info("Model=%s | candidate=%s | FDA excerpts=%d", _model, CANDIDATE, len(excerpts))
    for case, prompt in prompts.items():
        logger.info("%s | prompt SHA-256=%s", case, hashlib.sha256(prompt.encode()).hexdigest())

    counts: dict[str, Counter[str | None]] = {case: Counter() for case in prompts}
    for index in range(1, runs + 1):
        results = await asyncio.gather(
            *(replay(prompt, indications) for prompt in prompts.values()),
            return_exceptions=True,
        )
        for case, result in zip(prompts, results, strict=True):
            if isinstance(result, BaseException):
                counts[case][None] += 1
                logger.error("%s run %d | request failed: %s", case, index, type(result).__name__)
                continue
            verdict, reason = result
            counts[case][verdict] += 1
            logger.info("%s run %d | verdict=%s | reason=%s", case, index, verdict, reason)

    for case, verdicts in counts.items():
        logger.info("%s | %s", case, dict(verdicts))
    return int(any(verdicts[None] for verdicts in counts.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Live calls per prompt and excerpt order.")
    parser.add_argument("--label-cache", type=Path, default=SAVED_LABELS, help="Saved sildenafil FDA extraction JSON.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(run(args.label_cache, args.runs))


if __name__ == "__main__":
    raise SystemExit(main())
