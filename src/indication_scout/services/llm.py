"""Generic LLM call helpers."""

import json
import logging
import re

from anthropic import AsyncAnthropic, NOT_GIVEN
from dotenv import load_dotenv

from indication_scout.config import get_settings
from indication_scout.data_sources.base_client import DataSourceError

load_dotenv()

logger = logging.getLogger(__name__)

_settings = get_settings()
_model = _settings.llm_model
_small_model = _settings.small_llm_model
client = AsyncAnthropic()


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
        stripped = stripped.rsplit("```", 1)[0].strip()
    return stripped


def parse_llm_response(response: str) -> list[str]:
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    # Fallback: return the raw response as a single-item list
    return [response.strip()]


def _scan_json_blocks(text: str, opener: str, closer: str) -> list[str]:
    """Return all balanced JSON blocks of the given bracket type, in source order.

    Walks the text once, tracking depth and string-quote state so brackets inside
    strings don't break balance. Skips escaped quotes. Returns each complete
    top-level block as a substring; nested blocks are not separately returned.
    """
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            if depth == 0:
                start = i
            depth += 1
        elif ch == closer and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def parse_last_json_array(response: str) -> list | None:
    """Find the LAST complete JSON array in `response` and return it as a list.

    Tolerant of LLM responses that include reasoning text, self-corrections, or
    multiple candidate arrays. The last balanced `[...]` block that parses as a
    list wins. Returns None if no parseable array is found.
    """
    text = strip_markdown_fences(response)
    for block in reversed(_scan_json_blocks(text, "[", "]")):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def parse_last_json_object(response: str) -> dict | None:
    """Find the LAST complete JSON object in `response` and return it as a dict.

    Same tolerance as parse_last_json_array but for `{...}` blocks. Returns None
    if no parseable object is found.
    """
    text = strip_markdown_fences(response)
    for block in reversed(_scan_json_blocks(text, "{", "}")):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


async def query_llm(prompt: str, system: str = "") -> str:
    response = await client.messages.create(
        model=_model,
        max_tokens=_settings.llm_max_tokens,
        temperature=0,
        system=system or NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        raise DataSourceError(
            "llm", f"Empty content in LLM response (stop_reason={response.stop_reason})"
        )
    return response.content[0].text


async def query_small_llm(prompt: str, system: str = "") -> str:
    response = await client.messages.create(
        model=_small_model,
        max_tokens=_settings.small_llm_max_tokens,
        temperature=0,
        system=system or NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        raise DataSourceError(
            "llm", f"Empty content in LLM response (stop_reason={response.stop_reason})"
        )
    return response.content[0].text
