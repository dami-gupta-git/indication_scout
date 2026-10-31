"""Pin the agent model's sampling settings."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

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


async def test_attach_mechanism_graph_sets_built_graph():
    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
    from indication_scout.graph.model_graph import MechanismGraph

    output = SupervisorOutput(drug_name="drugx", top_diseases=["disease one"])
    built = MechanismGraph(drug_name="drugx")
    with patch(
        "indication_scout.graph.build.build_graph", AsyncMock(return_value=built)
    ):
        result = await analysis_runner.attach_mechanism_graph(output)
    assert result is output
    assert result.mechanism_graph == built
    assert result.drug_name == "drugx"
    assert result.top_diseases == ["disease one"]


async def test_attach_mechanism_graph_leaves_none_on_data_source_error():
    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
    from indication_scout.data_sources.base_client import DataSourceError

    output = SupervisorOutput(drug_name="drugx", top_diseases=["disease one"])
    before = output.model_dump()
    with patch(
        "indication_scout.graph.build.build_graph",
        AsyncMock(side_effect=DataSourceError("open_targets", "boom")),
    ):
        result = await analysis_runner.attach_mechanism_graph(output)
    assert result.mechanism_graph is None
    assert result.model_dump() == before


async def test_attach_mechanism_graph_skips_temporal_holdout():
    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput

    output = SupervisorOutput(
        drug_name="drugx",
        top_diseases=["disease one"],
        date_before=date(2020, 1, 1),
    )
    build = AsyncMock()
    with patch("indication_scout.graph.build.build_graph", build):
        result = await analysis_runner.attach_mechanism_graph(output)
    assert result is output
    assert result.drug_name == "drugx"
    assert result.top_diseases == ["disease one"]
    assert result.date_before == date(2020, 1, 1)
    assert result.mechanism_graph is None
    build.assert_not_awaited()
