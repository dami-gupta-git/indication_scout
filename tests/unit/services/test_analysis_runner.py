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


async def test_pair_analysis_resolves_the_approval_verdict_before_the_sub_agents_start():
    """The pair's FDA verdict is resolved once, ahead of the sub-agents, and handed to the trials agent, so every consumer
    reads the same verdict. An approved pair still runs (the user asked for it) and carries no relationship tag.
    """
    from indication_scout.helpers.drug_helpers import DrugIntake

    events: list[str] = []

    async def mapping(**kwargs):
        events.append("mapping")
        return {"disease x": "approved"}

    def agent_runner(name: str):
        async def run(*args, **kwargs):
            events.append(name)
            return None

        return run

    intake = DrugIntake(
        chembl_id="CHEMBL1", aliases=["drugx"], approved_indications=["disease x"]
    )
    session_factory = MagicMock()
    with (
        patch(
            "indication_scout.helpers.drug_helpers.seed_drug_intake",
            new=AsyncMock(return_value=intake),
        ),
        patch.object(analysis_runner, "ChatAnthropic", MagicMock()),
        patch.object(
            analysis_runner, "make_session_factory", return_value=session_factory
        ),
        patch(
            "indication_scout.services.embeddings.embed_async",
            new=AsyncMock(return_value=[[0.0]]),
        ),
        patch(
            "indication_scout.agents.mechanism.mechanism_agent.build_mechanism_agent",
            return_value=MagicMock(),
        ),
        patch(
            "indication_scout.agents.mechanism.mechanism_agent.run_mechanism_agent",
            new=agent_runner("mechanism"),
        ),
        patch(
            "indication_scout.agents.literature.literature_agent.build_literature_agent",
            return_value=MagicMock(),
        ),
        patch(
            "indication_scout.agents.literature.literature_agent.run_literature_agent",
            new=agent_runner("literature"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_agent.build_clinical_trials_agent",
            return_value=MagicMock(),
        ) as build_ct,
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_agent.run_clinical_trials_agent",
            new=agent_runner("trials"),
        ),
        patch(
            "indication_scout.services.approval_check.get_fda_approved_disease_mapping",
            new=mapping,
        ),
        patch.object(
            analysis_runner, "attach_mechanism_graph", new=AsyncMock(return_value=None)
        ),
        patch.object(analysis_runner, "format_report", return_value="report"),
    ):
        output, report = await analysis_runner.run_pair_analysis("drugx", "disease x")

    assert events[0] == "mapping"
    assert set(events[1:]) == {"mechanism", "literature", "trials"}
    assert output.disease_findings[0].approval_relationship == "none"
    assert report == "report"
    # The trials tool receives the pair's verdict from the same mapping call.
    assert build_ct.call_count == 1
    assert build_ct.call_args.kwargs["approval_label"] == "approved"
    assert build_ct.call_args.kwargs["date_before"] is None
