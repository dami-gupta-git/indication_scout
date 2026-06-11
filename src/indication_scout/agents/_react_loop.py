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

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


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
    # that prefix from cache instead of reprocessing it. Output is unchanged; it cuts
    # per-turn input-processing latency on a multi-turn loop.
    system_message = SystemMessage(
        content=[
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
        ]
    )

    async def call_model(state: MessagesState) -> dict:
        response = await model.ainvoke([system_message] + state["messages"])
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
