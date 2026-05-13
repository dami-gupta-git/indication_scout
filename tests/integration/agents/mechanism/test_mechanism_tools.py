"""Integration tests for the mechanism tool layer.

Hits real ChEMBL / Open Targets. Verifies the tool-level contract without
going through the agent loop, so tool regressions can't hide behind an LLM
that happens to paper over them.
"""

import logging

from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools
from indication_scout.models.model_open_targets import MechanismOfAction

logger = logging.getLogger(__name__)


async def test_get_drug_unresolved_name_returns_empty_moas():
    """Unknown drug name → resolve_drug_name raises DataSourceError →
    get_drug catches it and returns an empty MoA artifact instead of
    propagating the exception out of the tool.

    This is the precondition for the prompt's "no mechanisms" short-circuit
    rule. Without the try/except in get_drug, the agent loop crashes and
    the short-circuit is unreachable. Same pattern as check_fda_approval
    in clinical_trials_tools.
    """
    tools = build_mechanism_tools()
    get_drug = next(t for t in tools if t.name == "get_drug")

    msg = await get_drug.ainvoke(
        {
            "name": "get_drug",
            "args": {"drug_name": "xyzzy-not-a-drug-2026"},
            "id": "test-call-1",
            "type": "tool_call",
        }
    )

    # Artifact contract: empty list, correct type — the short-circuit
    # rule reads len(mechanisms_of_action) == 0.
    assert isinstance(msg.artifact, list)
    assert msg.artifact == []

    # Content string: plain-English, attributes the absence to our tools
    # (matches the prompt's REPORTING rule so the summary can quote it).
    assert "xyzzy-not-a-drug-2026" in msg.content
    assert "not" in msg.content.lower() and (
        "resolve" in msg.content.lower() or "found" in msg.content.lower()
    )
