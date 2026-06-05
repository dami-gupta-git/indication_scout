"""Unit tests for the supervisor streaming refactor.

`run_supervisor_agent` was switched from `agent.ainvoke` to a streaming collector
(`_stream_and_collect`, stream_mode="updates") so progress events can be emitted. The CLI calls
`run_supervisor_agent` with NO `on_event`, so the no-callback path MUST reproduce the exact message
list `ainvoke` would have returned — otherwise the CLI breaks silently. These tests lock that in,
and verify the progress events fire in order when a callback is supplied.

No network / no real LLM: a scripted tool-binding fake model drives the real LangGraph react agent.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.supervisor.progress import ProgressEvent
from indication_scout.agents.supervisor.supervisor_agent import _stream_and_collect


class _ToolFakeModel(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools (the real one raises NotImplementedError)."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@tool
def analyze_literature(disease_name: str) -> str:
    """Analyze literature for a disease."""
    return f"lit:{disease_name}"


@tool
def analyze_clinical_trials(disease_name: str) -> str:
    """Analyze clinical trials for a disease."""
    return f"ct:{disease_name}"


def _scripted_agent():
    """Real react agent driven by a fake model that calls two sub-agent tools then finishes."""
    scripted = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_literature",
                        "args": {"disease_name": "psoriasis"},
                        "id": "call_lit",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_clinical_trials",
                        "args": {"disease_name": "psoriasis"},
                        "id": "call_ct",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    model = _ToolFakeModel(messages=scripted)
    return create_react_agent(
        model=model, tools=[analyze_literature, analyze_clinical_trials]
    )


async def test_stream_collect_matches_ainvoke_when_no_callback():
    """No-callback streaming path reproduces ainvoke's exact message list (CLI parity guard)."""
    drug = "metformin"
    seed = {"messages": [HumanMessage(content=f"Find repurposing opportunities for {drug}")]}

    baseline = await _scripted_agent().ainvoke(seed)
    collected = await _stream_and_collect(_scripted_agent(), drug, on_event=None)

    base_msgs = baseline["messages"]
    coll_msgs = collected["messages"]

    assert [type(m).__name__ for m in coll_msgs] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert [type(m).__name__ for m in coll_msgs] == [type(m).__name__ for m in base_msgs]
    assert coll_msgs[0].content == base_msgs[0].content
    assert [
        (m.name, m.content) for m in coll_msgs if isinstance(m, ToolMessage)
    ] == [(m.name, m.content) for m in base_msgs if isinstance(m, ToolMessage)]
    assert coll_msgs[-1].content == "done"


async def test_stream_collect_emits_ordered_progress_events():
    """With a callback, events fire in lifecycle order with monotonic seq."""
    events: list[ProgressEvent] = []

    async def on_event(ev: ProgressEvent) -> None:
        events.append(ev)

    result = await _stream_and_collect(_scripted_agent(), "metformin", on_event=on_event)

    types = [e.type for e in events]
    assert types == [
        "started",
        "agent_start",
        "agent_done",
        "agent_start",
        "agent_done",
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert events[0].drug == "metformin"
    assert events[1].agent == "literature"
    assert events[1].disease == "psoriasis"
    assert events[2].agent == "literature"
    assert events[3].agent == "clinical_trials"
    assert events[4].agent == "clinical_trials"
    assert result["_seq"] == 5
