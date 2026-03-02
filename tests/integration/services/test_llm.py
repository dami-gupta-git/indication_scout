"""Integration tests for services/llm."""

import logging


from indication_scout.services.llm import query_llm, query_small_llm

logger = logging.getLogger(__name__)


async def test_query_small_llm_returns_string():
    """Test that query_small_llm returns a non-empty string."""
    result = await query_small_llm("Say the word 'hello' and nothing else.")

    assert isinstance(result, str)
    assert len(result) > 0


async def test_query_llm_returns_string():
    """Test that query_llm returns a non-empty string."""
    result = await query_llm("Say the word 'hello' and nothing else.")

    assert isinstance(result, str)
    assert len(result) > 0
