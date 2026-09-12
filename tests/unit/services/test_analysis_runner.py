"""Pin the agent model's sampling settings."""

from unittest.mock import MagicMock, patch

from indication_scout.config import get_settings
from indication_scout.services import analysis_runner


def test_build_agent_uses_temperature_zero():
    settings = get_settings()
    chat = MagicMock(return_value="llm")
    with (
        patch.object(analysis_runner, "ChatAnthropic", chat),
        patch.object(
            analysis_runner, "build_supervisor_agent", return_value=("a", "b", "c", "d")
        ),
    ):
        result = analysis_runner.build_agent(db=MagicMock())
    assert result == ("a", "b", "c", "d")
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == settings.llm_model
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == settings.llm_max_tokens
    assert kwargs["anthropic_api_key"] == settings.anthropic_api_key
    assert len(kwargs["callbacks"]) == 1
    assert kwargs["callbacks"][0]._requested_model == settings.llm_model
