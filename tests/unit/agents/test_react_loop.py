"""Unit tests for the gated ReAct loop.

Drives build_gated_react_loop with a scripted fake model (no network, no LLM) to assert:
- the loop ends at the tools node when finalize succeeds — no discarded trailing model turn;
- a rejected finalize (falsy artifact) loops back to the model to retry;
- the per-agent finalize predicates enforce their gates (CT: non-empty summary;
  supervisor: critique_ranking must have run before finalize).
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from indication_scout.agents._react_loop import (
    _trailing_tool_messages,
    build_gated_react_loop,
)
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    _finalize_done as ct_finalize_done,
)
from indication_scout.agents.supervisor.supervisor_agent import (
    _finalize_done as sup_finalize_done,
)


class _ScriptedModel:
    """Returns a pre-scripted AIMessage per ainvoke call; records call count."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = responses
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id}


def _ct_finalize_tool():
    @tool(response_format="content_and_artifact")
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """finalize"""
        if not summary.strip():
            return "REJECTED: empty summary.", ""
        return "Analysis complete.", summary

    return finalize_analysis


async def test_finalize_success_ends_loop_with_no_trailing_turn():
    """A successful finalize ends the loop at the tools node — the model is not
    re-invoked to generate the discarded trailing message."""
    finalize = _ct_finalize_tool()
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("finalize_analysis", {"summary": "real"}, "a")],
            ),
            # Tripwire: should never run after a successful finalize.
            AIMessage(content="TRAILING TURN"),
        ]
    )
    graph = build_gated_react_loop(model, [finalize], "sys", ct_finalize_done)

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

    assert model.calls == 1
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].name == "finalize_analysis"
    assert result["messages"][-1].artifact == "real"
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert not any(m.content == "TRAILING TURN" for m in ai_messages)


async def test_rejected_finalize_loops_back_then_succeeds():
    """An empty-summary finalize is rejected (falsy artifact) and the loop returns to
    the model, which retries with a real summary and then ends."""
    finalize = _ct_finalize_tool()
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("finalize_analysis", {"summary": "  "}, "a")],
            ),
            AIMessage(
                content="",
                tool_calls=[_tool_call("finalize_analysis", {"summary": "real"}, "b")],
            ),
            AIMessage(content="TRAILING TURN"),
        ]
    )
    graph = build_gated_react_loop(model, [finalize], "sys", ct_finalize_done)

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

    assert model.calls == 2
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].artifact == "real"
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2


async def test_supervisor_gate_requires_critique_before_finalize():
    """finalize_supervisor before critique_ranking is rejected (empty-dict artifact)
    and loops back; after critique runs, finalize succeeds and the loop ends."""
    critique_ran = {"value": False}

    @tool
    async def critique_ranking() -> str:
        """critique"""
        critique_ran["value"] = True
        return "ok"

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(summary: str) -> tuple[str, dict]:
        """finalize"""
        if not critique_ran["value"]:
            return "Cannot finalize yet: call critique_ranking first.", {}
        return "Supervisor analysis complete.", {"summary": summary, "blurbs": []}

    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("finalize_supervisor", {"summary": "s"}, "a")],
            ),
            AIMessage(
                content="",
                tool_calls=[_tool_call("critique_ranking", {}, "b")],
            ),
            AIMessage(
                content="",
                tool_calls=[_tool_call("finalize_supervisor", {"summary": "s"}, "c")],
            ),
            AIMessage(content="TRAILING TURN"),
        ]
    )
    graph = build_gated_react_loop(
        model, [critique_ranking, finalize_supervisor], "sys", sup_finalize_done
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

    assert model.calls == 3
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].name == "finalize_supervisor"
    assert result["messages"][-1].artifact == {"summary": "s", "blurbs": []}
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 3


def test_trailing_tool_messages_returns_only_contiguous_end_block():
    """_trailing_tool_messages returns the trailing ToolMessage block, stopping at the
    first non-ToolMessage so a prior turn's finalize can't leak in."""
    messages = [
        HumanMessage(content="go"),
        ToolMessage(content="old", name="finalize_analysis", tool_call_id="x"),
        AIMessage(content="", tool_calls=[_tool_call("search_trials", {}, "y")]),
        ToolMessage(content="a", name="search_trials", tool_call_id="y"),
        ToolMessage(content="b", name="get_completed", tool_call_id="z"),
    ]

    trailing = _trailing_tool_messages(messages)

    assert [m.name for m in trailing] == ["get_completed", "search_trials"]
