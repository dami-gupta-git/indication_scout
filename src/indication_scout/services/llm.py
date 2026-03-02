"""Generic LLM call helpers."""

import json
import logging
import re

from anthropic import AsyncAnthropic, NOT_GIVEN
from dotenv import load_dotenv

from indication_scout.config import get_settings

load_dotenv()

logger = logging.getLogger(__name__)

_model = get_settings().llm_model
_small_model = get_settings().small_llm_model
client = AsyncAnthropic()


def parse_llm_response(response: str) -> list[str]:
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    # Fallback: return the raw response as a single-item list
    return [response.strip()]


async def query_llm(prompt: str, system: str = "") -> str:
    response = await client.messages.create(
        model=_model,
        max_tokens=1024,
        system=system or NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def query_small_llm(prompt: str, system: str = "") -> str:
    response = await client.messages.create(
        model=_small_model,
        max_tokens=1024,
        system=system or NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
