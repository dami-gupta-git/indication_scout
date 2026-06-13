"""Gated ReAct loop.

A minimal two-node ReAct graph (model + tools) that mirrors LangGraph's prebuilt
create_react_agent, with one difference: the loop ends as soon as the finalize tool
SUCCEEDS, instead of feeding the finalize ToolMessage back to the model for one more
(discarded) trailing turn. That trailing turn is pure latency — assembly reads typed
artifacts off the ToolMessages, never the final AIMessage.

Used by the clinical_trials and supervisor agents, whose finalize tools have a
non-terminal reject path (empty-summary / critique-not-run) that must loop back to the
model to retry. literature and mechanism finalize tools have no reject path, so they use
the prebuilt's return_direct=True instead.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def cached_system_message(prompt: str) -> SystemMessage:
    """A SystemMessage carrying an Anthropic ephemeral cache_control breakpoint.

    The breakpoint caches the static prefix (tool definitions + system prompt), so turns
    2+ of an agent loop read it from cache instead of reprocessing it. Output is unchanged.
    Accepted as the `prompt` arg by both build_gated_react_loop and create_react_agent.
    """
    return SystemMessage(
        content=[
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
        ]
    )


def _with_history_breakpoint(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return `messages` with a cache_control breakpoint on the last message's content.

    The system block carries a breakpoint for the static prefix; this adds a second one
    on the most-recently-appended message, so each turn reads the entire prior
    conversation (tool results included) from cache instead of reprocessing it. The
    breakpoint must move with the tail every turn, so it's applied to a shallow copy of
    the last message only — the stored state messages are never mutated (a stale interior
    breakpoint would consume one of Anthropic's 4 slots and drift the prefix).
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
        # Empty string or empty/non-dict list content (e.g. an AIMessage with only
        # tool_calls) — nothing to mark; leave the system-prefix breakpoint as the sole
        # cache point this turn.
        return messages
    return [*messages[:-1], last.model_copy(update={"content": new_content})]


def history_cache_pre_model_hook(state: MessagesState) -> dict:
    """pre_model_hook for create_react_agent that caches the growing conversation.

    The gated loop applies _with_history_breakpoint inline; the prebuilt create_react_agent
    builds the message list internally, so the equivalent seam is its pre_model_hook. We
    return the marked messages under `llm_input_messages` — used as the model input without
    UPDATING stored `messages`, so the breakpoint stays a transient, per-turn copy (no
    interior breakpoint persists into state). literature and mechanism wire this in to get
    the same history caching as the gated supervisor / clinical_trials loops.
    """
    return {"llm_input_messages": _with_history_breakpoint(state["messages"])}


def build_gated_react_loop(
    llm,
    tools: list,
    prompt: str,
    finalize_done: Callable[[list], bool],
) -> Any:
    """Compile a ReAct loop that ends when `finalize_done(messages)` is True.

    Equivalent to create_react_agent(model=llm, tools=tools, prompt=prompt) for a str
    prompt — model node prepends the prompt as a SystemMessage on every call (matching the
    prebuilt) — except the tools->model edge is conditional: after the tools node, end if
    finalize succeeded, otherwise loop back to the model.
    """
    model = llm.bind_tools(tools)
    # Anthropic prompt caching: a cache_control breakpoint on the system block caches
    # the (static) system prompt + tool definitions, so turns 2+ of the agent loop read
    # that prefix from cache instead of reprocessing it. A second breakpoint on the last
    # history message (see _with_history_breakpoint) caches the growing conversation —
    # the accumulated tool results — so each turn reprocesses only the new tail. Output
    # is unchanged; it cuts per-turn input-processing latency on a multi-turn loop.
    system_message = cached_system_message(prompt)

    async def call_model(state: MessagesState) -> dict:
        messages = _with_history_breakpoint(state["messages"])
        response = await model.ainvoke([system_message] + messages)
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

    ToolNode appends one ToolMessage per tool call in a turn, so the trailing block is
    exactly this turn's tool results. Scanning only the trailing block (not all history)
    means a successful finalize from a PRIOR turn — there is none in normal runs, but be
    safe — can't end the loop on a later, non-finalize turn.
    """
    block: list[ToolMessage] = []
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            break
        block.append(m)
    return block
