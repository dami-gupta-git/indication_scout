"""Integration tests for runners/rag_runner."""

import logging

import pytest

from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.runners.rag_runner import run_rag

logger = logging.getLogger(__name__)


async def test_run_rag_empagliflozin(db_session_truncating, test_cache_dir):
    results = await run_rag("empagliflozin", db_session_truncating, cache_dir=test_cache_dir)
    summaries = " ".join(r.summary.lower() for r in results.values())

    # At least some positive and some negative signals
    assert "support" in summaries or "evidence" in summaries
    assert "not support" in summaries or "no significant" in summaries or "contradicts" in summaries

    # Basic structure checks
    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)


