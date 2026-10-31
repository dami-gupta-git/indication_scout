"""Unit tests for build_graph."""

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.graph.build import build_graph


async def test_build_graph_links_candidates_to_targets(
    run_output, stub_client, expected_graph
):
    graph = await build_graph(run_output, stub_client)

    assert graph == expected_graph


async def test_build_graph_without_mechanism_returns_none(stub_client):
    output = SupervisorOutput(drug_name="drugx", top_diseases=["disease one"])

    assert await build_graph(output, stub_client) is None
