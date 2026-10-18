"""Gated ReAct loop: like LangGraph's create_react_agent, but ends as soon as the finalize tool succeeds.

Used by the clinical_trials and supervisor agents, whose finalize tools can reject and loop back to the model.
literature and mechanism have no reject path, so they use the prebuilt's return_direct=True.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def cached_system_message(prompt: str) -> SystemMessage:
    """A SystemMessage with an Anthropic ephemeral cache_control breakpoint on the static prefix."""
    return SystemMessage(
        content=[
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
        ]
    )


def _with_history_breakpoint(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return `messages` with a cache_control breakpoint on the last message.

    Applied to a copy of the last message only; stored state is never mutated, since a stale interior
    breakpoint would use up one of Anthropic's 4 slots.
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last.content
    if isinstance(content, str) and content:
        new_content: Any = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        new_content = [
            *content[:-1],
            {**content[-1], "cache_control": {"type": "ephemeral"}},
        ]
    else:
        # Nothing to mark (e.g. an AIMessage with only tool_calls).
        return messages
    return [*messages[:-1], last.model_copy(update={"content": new_content})]


def history_cache_pre_model_hook(state: MessagesState) -> dict:
    """pre_model_hook for create_react_agent that adds the history cache breakpoint.

    Returned under `llm_input_messages`, so the breakpoint is a per-turn copy and never persists into state.
    """
    return {"llm_input_messages": _with_history_breakpoint(state["messages"])}


def build_gated_react_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    prompt: str,
    finalize_done: Callable[[list], bool],
) -> Any:
    """Compile a ReAct loop that ends when `finalize_done(messages)` is True.

    Same as create_react_agent for a str prompt, except the tools->model edge ends the loop if finalize succeeded.
    """
    model = llm.bind_tools(tools)
    # Prompt caching: one breakpoint on the system block, a second on the last history message.
    system_message = cached_system_message(prompt)

    async def call_model(state: MessagesState) -> dict:
        messages = _with_history_breakpoint(state["messages"])
        response = await model.ainvoke([system_message, *messages])
        return {"messages": [response]}

    def after_model(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: MessagesState) -> str:
        if finalize_done(state["messages"]):
            return END
        return "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def _trailing_tool_messages(messages: list) -> list[ToolMessage]:
    """The contiguous block of ToolMessages at the end of `messages`.

    This is the current turn's tool results, so a finalize from an earlier turn can't end the loop.
    """
    block: list[ToolMessage] = []
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            break
        block.append(m)
    return block
