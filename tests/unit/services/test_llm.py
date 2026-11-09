"""Pin the sampling settings on the direct SDK helpers.

The 1.x SDK dropped ``temperature`` from the create signature, so it travels in ``extra_body``. A
non-zero temperature makes the critic, gates and judgments non-deterministic run to run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import TextBlock

from indication_scout.services import llm


def _response(text: str):
    return type(
        "R",
        (),
        {"content": [TextBlock(type="text", text=text)], "stop_reason": "end_turn"},
    )()


class _Stream:
    def __init__(self, response) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def get_final_message(self):
        return self._response


@pytest.mark.parametrize(
    "max_tokens, expected_max_tokens",
    [(None, llm._settings.llm_max_tokens), (32000, 32000)],
)
async def test_query_llm_streams_with_temperature_zero(max_tokens, expected_max_tokens):
    stream = MagicMock(return_value=_Stream(_response("ok")))
    with patch.object(llm.client.messages, "stream", stream):
        out = await llm.query_llm("prompt", system="sys", max_tokens=max_tokens)
    assert out == "ok"
    kwargs = stream.call_args.kwargs
    assert kwargs["model"] == llm._model
    assert kwargs["max_tokens"] == expected_max_tokens
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert kwargs["extra_body"] == {"temperature": 0}


@pytest.mark.parametrize(
    "fn, model, max_tokens",
    [
        (llm.query_small_llm, llm._small_model, llm._settings.small_llm_max_tokens),
        (llm.query_big_llm, llm._big_model, llm._settings.llm_max_tokens),
    ],
)
async def test_helpers_send_temperature_zero(fn, model, max_tokens):
    create = AsyncMock(return_value=_response("ok"))
    with patch.object(llm.client.messages, "create", create):
        out = await fn("prompt", system="sys")
    assert out == "ok"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == model
    assert kwargs["max_tokens"] == max_tokens
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert kwargs["extra_body"] == {"temperature": 0}
