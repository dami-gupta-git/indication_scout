"""Progress event schema for the live run feed: one emitted milestone for one pipeline phase.

Returned in the analyses poll response and rendered by the frontend's LoadingState as a checklist line. Phase values
are the PHASES constants in `services/progress.py`.
"""

from pydantic import BaseModel


class ProgressEvent(BaseModel):
    """A single user-facing progress milestone within a run."""

    phase: str
    message: str
