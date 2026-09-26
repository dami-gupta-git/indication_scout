"""Replay only the supervisor turn that drafts the ranked blurbs, and count dropped candidates.

Takes the recorded request (full conversation, system prompt, tool schemas) for the turn whose response
called critique_ranking, and resends it live N times. By default it reads `ranking_step.yaml` next to the
drug's pipeline-replay cassette: a frozen extract of that turn from before the must-include list existed,
kept separate so re-recording the pipeline cassette does not change the baseline. Each
response's blurbs are compared with the candidates that should be ranked: every investigated
candidate except those the evidence gate removes (0 relevant trials and literature strength none)
and those whose trial search was unresolved (both go to the report footers, not the ranking).

Variants (each changes exactly one thing in the recorded request):
  recorded      — the request exactly as recorded.
  current       — the `blurbs` description swapped for the one in the current source.
  no_count      — the `blurbs` description with no number ("for the ranked candidates").
  no_notable    — the system prompt without the "pairs with notable evidence" sentence.
  explicit_list — the investigation result ends with the names of every candidate that must be ranked.
  current_with_list — `current` and `explicit_list` together (what the current code sends).

Each run also reports how many numbered lines the model wrote in its text before the tool call.

Usage:
  .venv/bin/python scripts/replay_ranking_step.py [--drug semaglutide] [--repeats 5] [--variant recorded current ...]
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

logger = logging.getLogger("indication_scout.scripts.replay_ranking_step")

CASSETTE_DIR = PROJECT_ROOT / "tests" / "regression" / "pipeline_replay" / "cassettes"

SUPERVISOR_TOOLS = (
    PROJECT_ROOT
    / "src"
    / "indication_scout"
    / "agents"
    / "supervisor"
    / "supervisor_tools.py"
)
# The `blurbs` argument lines of finalize_supervisor's docstring, 8-space indented in source.
_SOURCE_BLURBS_DESC = re.compile(
    r"^ {8}(- blurbs: [^\n]*)\n {8}(  [^\n]*?in rank order\.)", re.MULTILINE
)


def _current_blurbs_desc() -> str:
    """The `blurbs` description as the current source renders it into the tool schema."""
    m = _SOURCE_BLURBS_DESC.search(SUPERVISOR_TOOLS.read_text())
    if m is None:
        raise ValueError(
            "could not find the blurbs description in finalize_supervisor's docstring"
        )
    return f"{m.group(1)}\n{m.group(2)}"


_NO_COUNT_BLURBS_DESC = (
    "- blurbs: a list of structured per-candidate entries for the\n"
    "  ranked candidates in your summary, in rank order."
)
_NOTABLE_SENTENCE = (
    " This list surfaces pairs with notable evidence (positive, mixed,\n"
    "stalled, or adverse), not a recommendation list."
)
_NUMBERED_LINE = re.compile(r"^\s*\**\d+\.", re.MULTILINE)
VARIANTS = ["recorded", "current", "no_count", "no_notable", "explicit_list", "current_with_list"]

_CANDIDATE_LINE = re.compile(
    r"^\s+-\s+(?P<disease>[^:]+):\s+literature\s+(?P<lit>\S+?),.*?;\s+trials\s+(?P<trials>[^;]+);"
)


def _body(part: dict) -> str:
    b = part["body"]
    b = b.get("string", b) if isinstance(b, dict) else b
    return b.decode() if isinstance(b, bytes) else b


def load_ranking_requests(drug: str, cassette_path: Path | None = None) -> list[dict]:
    """Every recorded supervisor request whose response called critique_ranking."""
    path = cassette_path or CASSETTE_DIR / drug / "ranking_step.yaml"
    cassette = yaml.safe_load(path.read_text())
    found: list[dict] = []
    for it in cassette["interactions"]:
        if "api.anthropic.com" not in it["request"]["uri"]:
            continue
        resp = json.loads(_body(it["response"]))
        if any(
            p.get("type") == "tool_use" and p.get("name") == "critique_ranking"
            for p in resp.get("content", [])
        ):
            found.append(json.loads(_body(it["request"])))
    return found


def expected_candidates(request: dict) -> tuple[list[str], list[str], list[str]]:
    """(should be ranked, gate-excluded, unresolved) from the investigate_top_candidates tool result."""
    for msg in request["messages"]:
        if not isinstance(msg["content"], list):
            continue
        for part in msg["content"]:
            if part.get("type") != "tool_result":
                continue
            content = part["content"]
            text = (
                content
                if isinstance(content, str)
                else " ".join(p.get("text", "") for p in content)
            )
            if not text.startswith("Auto-investigated"):
                continue
            ranked, gated, unresolved = [], [], []
            for line in text.splitlines():
                m = _CANDIDATE_LINE.match(line)
                if not m:
                    continue
                disease = m.group("disease").strip().lower()
                trials = m.group("trials")
                if "could not be resolved" in trials:
                    unresolved.append(disease)
                elif trials.startswith("0 relevant") and m.group("lit") == "none":
                    gated.append(disease)
                else:
                    ranked.append(disease)
            return ranked, gated, unresolved
    raise ValueError("no investigate_top_candidates result in the recorded request")


# The two-line `blurbs` description in a recorded tool schema; the second line's indent differs between recordings.
_RECORDED_BLURBS_DESC = re.compile(r"- blurbs: [^\n]*\n(?P<indent>[ ]*)[^\n]*?in rank order\.")


def _swap_blurbs_desc(description: str, first: str, second: str) -> str:
    """Replace the recorded two-line `blurbs` description, keeping the recording's indentation."""
    m = _RECORDED_BLURBS_DESC.search(description)
    if m is None:
        raise ValueError("expected blurbs description not found in the recorded tool description")
    indent = m.group("indent")
    return description[: m.start()] + f"{first}\n{indent}{second.lstrip()}" + description[m.end() :]


def make_variant(request: dict, variant: str, should_rank: list[str]) -> dict:
    """Copy of the recorded request with exactly one change for `variant`."""
    if variant == "current_with_list":
        return make_variant(make_variant(request, "current", should_rank), "explicit_list", should_rank)
    req = json.loads(json.dumps(request))
    tool = next(t for t in req["tools"] if t["name"] == "finalize_supervisor")
    if variant == "current":
        first, second = _current_blurbs_desc().split("\n", 1)
        tool["description"] = _swap_blurbs_desc(tool["description"], first, second)
    elif variant == "no_count":
        first, second = _NO_COUNT_BLURBS_DESC.split("\n", 1)
        tool["description"] = _swap_blurbs_desc(tool["description"], first, second)
    elif variant == "no_notable":
        block = req["system"][0]
        if _NOTABLE_SENTENCE not in block["text"]:
            raise ValueError("expected sentence not found in the recorded system prompt")
        block["text"] = block["text"].replace(_NOTABLE_SENTENCE, "")
    elif variant == "explicit_list":
        for msg in req["messages"]:
            if not isinstance(msg["content"], list):
                continue
            for part in msg["content"]:
                if (
                    part.get("type") == "tool_result"
                    and isinstance(part.get("content"), str)
                    and part["content"].startswith("Auto-investigated")
                ):
                    part["content"] += (
                        f"\n\nYour ranking (summary and blurbs) must include every one of these "
                        f"{len(should_rank)} candidates: {', '.join(should_rank)}."
                    )
    return req


async def one_call(client: AsyncAnthropic, req: dict) -> tuple[list[str] | None, int]:
    """(blurb diseases from critique_ranking or None, numbered lines in the model's text)."""
    # This SDK version has no `temperature` keyword; send the recorded value in the request body unchanged.
    req = dict(req)
    extra_body = {"temperature": req.pop("temperature")} if "temperature" in req else {}
    resp = await client.messages.create(**req, extra_body=extra_body)
    text = "\n".join(p.text for p in resp.content if p.type == "text")
    text_lines = len(_NUMBERED_LINE.findall(text))
    for part in resp.content:
        if part.type == "tool_use" and part.name == "critique_ranking":
            blurbs = [(b.get("disease") or "").strip().lower() for b in part.input.get("blurbs") or []]
            return blurbs, text_lines
    return None, text_lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", default="semaglutide")
    parser.add_argument("--cassette", type=Path, help="recording to replay (default: the drug's frozen ranking_step.yaml)")
    parser.add_argument("--recording", type=int, default=0, help="which recorded ranking turn to replay")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=["recorded", "current"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    requests = load_ranking_requests(args.drug, args.cassette)
    if not requests:
        print(f"no recorded ranking turn for {args.drug}")
        return 1
    base = requests[args.recording]
    should_rank, gated, unresolved = expected_candidates(base)
    print(f"recording {args.recording} of {len(requests)}: model={base['model']} temperature={base.get('temperature')}")
    print(f"should be ranked: {len(should_rank)}  gate-excluded: {gated}  unresolved: {unresolved}")
    print(f"current blurbs wording: {_current_blurbs_desc()!r}\n")

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for variant in args.variant:
        req = make_variant(base, variant, should_rank)
        results = await asyncio.gather(*(one_call(client, req) for _ in range(args.repeats)))
        drops = 0
        print(f"== {variant}")
        for i, (blurbs, text_lines) in enumerate(results, 1):
            if blurbs is None:
                print(f"  run {i}: did not call critique_ranking (text list {text_lines})")
                continue
            missing = [d for d in should_rank if d not in blurbs]
            drops += bool(missing)
            print(f"  run {i}: text list {text_lines}, {len(blurbs)} blurbs, missing {missing or 'none'}")
        print(f"  runs with a drop: {drops}/{args.repeats}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
