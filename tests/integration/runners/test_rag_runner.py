"""Integration tests for runners/rag_runner."""

import logging

import pytest

from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.runners.rag_runner import run_rag

logger = logging.getLogger(__name__)


async def test_run_rag_empagliflozin(db_session_truncating, test_cache_dir):
    results = await run_rag(
        "empagliflozin", db_session_truncating, cache_dir=test_cache_dir
    )
    summaries = " ".join(r.summary.lower() for r in results.values())

    # Empagliflozin has broad positive evidence
    assert "support" in summaries or "evidence" in summaries

    # Basic structure checks
    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)


async def test_run_rag_colchicine_mixed_signals(db_session_truncating, test_cache_dir):
    results = await run_rag(
        "colchicine", db_session_truncating, cache_dir=test_cache_dir
    )

    # Colchicine has only preclinical cancer evidence — all indications should be weak or none.
    strengths = [s.strength for s in results.values()]
    assert all(
        s in ("weak", "none") for s in strengths
    ), f"Expected only weak/none signals for colchicine, got: {strengths}"

    # Breast cancer, lung cancer, and sarcoma flag adverse effects (toxicity to normal cells,
    # myelosuppression) — expect at least 2.
    adverse_flags = {d for d, s in results.items() if s.has_adverse_effects}
    assert len(adverse_flags) >= 2, (
        f"Expected at least 2 indications to flag adverse effects, got: {adverse_flags}"
    )

    # Summaries should reflect the preclinical-only, no-clinical-trials nature of the evidence.
    summaries = " ".join(r.summary.lower() for r in results.values())
    cautionary_terms = [
        "no clinical",
        "preclinical",
        "limited",
        "toxicity",
        "not colchicine",
        "in vitro",
    ]
    matches = [t for t in cautionary_terms if t in summaries]
    assert (
        len(matches) >= 3
    ), f"Expected at least 3 cautionary terms in colchicine summaries, found: {matches}"

    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)
