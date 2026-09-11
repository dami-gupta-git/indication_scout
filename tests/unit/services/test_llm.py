"""Pin the sampling settings on the direct SDK helpers.

The 1.x SDK dropped ``temperature`` from the create signature, so it travels in ``extra_body``. A
non-zero temperature makes the critic, gates and judgments non-deterministic run to run.
"""

from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import TextBlock

from indication_scout.services import llm


def _response(text: str):
    return type(
        "R",
        (),
        {"content": [TextBlock(type="text", text=text)], "stop_reason": "end_turn"},
    )()


@pytest.mark.parametrize(
    "fn, model, max_tokens",
    [
        (llm.query_llm, llm._model, llm._settings.llm_max_tokens),
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
