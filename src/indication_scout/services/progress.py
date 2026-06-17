"""Per-run progress feed for the live "what's happening now" UI.

A run is driven inside one asyncio task (`_execute` in the analyses route). That task sets the
active job's `emit` callback in a contextvar; the supervisor tools and the retrieval service
call `emit_progress(...)` at a handful of user-facing milestones. The frontend already polls
`GET /api/analyses/{job_id}` every ~1.5s, so the emitted events ride that existing poll — no
SSE, no new endpoint, no added latency (emit is a contextvar lookup + list append).

Emission is best-effort and side-channel only: when no callback is bound (CLI runs, tests),
`emit_progress` is a no-op. It must never raise into the pipeline.
"""

import logging
from contextvars import ContextVar
from typing import Callable

logger = logging.getLogger(__name__)

# The five user-facing pipeline phases, in order. The frontend renders these as a fixed
# checklist; `emit_progress(phase, message)` fills in the live line/count for one of them.
PHASE_CANDIDATES = "candidates"
PHASE_MECHANISM = "mechanism"
PHASE_TRIALS = "trials"
PHASE_LITERATURE = "literature"
PHASE_SUMMARY = "summary"

PHASES = (
    PHASE_CANDIDATES,
    PHASE_MECHANISM,
    PHASE_TRIALS,
    PHASE_LITERATURE,
    PHASE_SUMMARY,
)

# Bound by the run task to the active job's append callback; absent everywhere else.
_emitter: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "progress_emitter", default=None
)


def set_emitter(emit: Callable[[str, str], None] | None):
    """Bind the active run's progress callback. Returns the contextvar token for reset()."""
    return _emitter.set(emit)


def reset_emitter(token) -> None:
    """Restore the previous emitter binding (call in a finally after the run)."""
    _emitter.reset(token)


def emit_progress(phase: str, message: str) -> None:
    """Record a progress milestone for the active run. No-op when no run is bound.

    `phase` must be one of PHASES; `message` is the human-readable live line (e.g.
    "Found 18 candidate diseases"). Never raises into the pipeline.
    """
    emit = _emitter.get()
    if emit is None:
        return
    try:
        emit(phase, message)
    except Exception:  # noqa: BLE001 — progress is side-channel; never break the run
        logger.debug("emit_progress failed (non-fatal)", exc_info=True)
