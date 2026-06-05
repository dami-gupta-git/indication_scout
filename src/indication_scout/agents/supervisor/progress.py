"""Progress events emitted while a supervisor run streams.

`run_supervisor_agent` streams the agent with `stream_mode="updates"` and, per message, calls an
optional `on_event` callback with one of these events; the API layer re-emits them over SSE. They
need no change to the agent or its tools — they're derived from AIMessage tool-calls (a sub-agent
is about to run) and ToolMessage names/artifacts (it finished).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Sub-agent tool names → the short agent label used in events.
_AGENT_TOOL_LABELS: dict[str, str] = {
    "analyze_literature": "literature",
    "analyze_clinical_trials": "clinical_trials",
}


class ProgressEvent(BaseModel):
    """A single progress event in a supervisor run's lifecycle.

    `type` discriminates the event; remaining fields are populated per type (empty/None otherwise).
    `seq` is a monotonic counter assigned by the runner; `ts` is set by the API layer on enqueue.
    """

    type: Literal[
        "started",
        "candidates",
        "mechanism_start",
        "mechanism_done",
        "agent_start",
        "agent_done",
        "finalizing",
        "done",
        "error",
        "cancelled",
    ]
    seq: int = 0
    drug: str = ""
    diseases: list[str] = Field(default_factory=list)
    agent: str = ""
    disease: str = ""
    summary: str = ""
    target_count: int | None = None
    job_id: str = ""
    message: str = ""
    where: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None:
                if field_info.default_factory is not None:
                    values[field_name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[field_name] = field_info.default
        return values


def agent_label(tool_name: str) -> str | None:
    """Return the short agent label for a sub-agent tool name, or None.

    None means the tool isn't a per-disease sub-agent delegation (e.g. `analyze_mechanism`,
    `finalize_supervisor`), which the runner handles with dedicated event types.
    """
    return _AGENT_TOOL_LABELS.get(tool_name)
