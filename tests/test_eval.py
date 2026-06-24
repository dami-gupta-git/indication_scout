"""DeepEval report-quality test for a real end-to-end IndicationScout run.

This runs the full supervisor pipeline for a drug (real data-source calls + real
LLM agents), renders the Markdown report, then judges it with several Claude-as-
judge metrics — each one checking a different way the report could drift from the
upstream data it was built on. The throughline is the core project invariant:
accuracy over coverage. Every claim in the report must be traceable to the
structured upstream facts; a smaller faithful report is preferred to a larger
embellished one.

The judge LLM is Claude (the app's provider), not deepeval's default OpenAI.

Metrics (all GEval rubrics judged by Claude):
  - DiseaseGroundedness        — no candidate disease the report names is absent
                                 from the surfaced candidate list.
  - CitationGrounding          — every PMID cited in prose is one the upstream
                                 EvidenceSummary actually collected (no fabricated
                                 citations).
  - EvidenceStrengthFaithful   — narrative strength/direction language matches the
                                 structured strength/direction (no overstatement).
  - NoEfficacyOverclaim        — no efficacy/approval claim beyond what the data
                                 supports (untested ≠ proven; not-on-label ≠ approved).
  - DevStageConsistency        — prose doesn't contradict the authoritative
                                 development-stage line.

Integration test — slow (a full scout run) and it costs tokens. Marked so it
doesn't run with the default unit suite:

    pytest -m eval tests/test_eval.py          # run it
    deepeval test run tests/test_eval.py       # run it via the deepeval CLI

Requires ANTHROPIC_API_KEY (and the usual data-source keys) in the environment.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.services.analysis_runner import run_analysis
from indication_scout.services.dev_stage import dev_stage_phrase

load_dotenv()

# Drug to run end-to-end. metformin is a well-characterised repurposing subject
# with stable upstream signal, so the run is representative without being exotic.
_DRUG = "metformin"

# Judge with Claude — the app has no OpenAI dependency, and deepeval defaults its
# judge to OpenAI. AnthropicModel reads ANTHROPIC_API_KEY from the environment.
# The report is large; raise the judge's output budget so the structured-output
# JSON (verdict + reason) isn't truncated mid-string.
_JUDGE = AnthropicModel(
    model="claude-sonnet-4-6",
    generation_kwargs={"max_tokens": 8192},
)

_P = LLMTestCaseParams  # alias for brevity in metric definitions


def _finding_context(f: CandidateFindings) -> str:
    """A flat block of the structured upstream facts for one disease finding.

    This is the ground truth the report's prose for this disease must stay within.
    Pulls the fields a reviewer would cross-check the rendered prose against:
    evidence strength/direction, collected PMIDs, and the authoritative dev-stage
    line. None/absent sub-agent output is stated explicitly so the judge knows the
    report had nothing to draw on there.
    """
    lines = [f"DISEASE: {f.disease}", f"approval_relationship: {f.approval_relationship}"]

    es = f.literature.evidence_summary if f.literature else None
    if es is not None:
        collected_pmids = sorted(
            set(es.supporting_pmids)
            | set(es.contradicting_pmids)
            | set(es.neutral_pmids)
            | set(es.relevant_pmids)
        )
        lines += [
            f"evidence_strength: {es.strength}",
            f"evidence_direction: {es.direction}",
            f"evidence_basis: {es.evidence_basis}",
            f"study_count: {es.study_count}",
            f"collected_PMIDs: {', '.join(collected_pmids) if collected_pmids else 'none'}",
        ]
    else:
        lines.append("literature: no evidence summary (sub-agent produced nothing)")

    ct = f.clinical_trials
    if ct is not None:
        stage = dev_stage_phrase(ct.signals) if ct.signals else None
        lines.append(f"development_stage: {stage or 'undetermined'}")
        if ct.approval is not None:
            lines.append(
                f"fda_approved_for_this_indication: {bool(ct.approval.is_approved)}"
            )
    else:
        lines.append("clinical_trials: no output (sub-agent produced nothing)")

    return "\n".join(lines)


def _context(output: SupervisorOutput) -> list[str]:
    """Full retrieval context: the candidate list plus per-disease structured facts.

    The first chunk enumerates every surfaced candidate (for DiseaseGroundedness);
    each subsequent chunk is one disease's structured upstream facts (for the
    citation / strength / overclaim / dev-stage metrics).
    """
    chunks = [
        "SURFACED CANDIDATE DISEASES for "
        f"{output.drug_name}: {', '.join(output.candidate_diseases)}"
    ]
    chunks += [_finding_context(f) for f in output.disease_findings]
    return chunks


# --- Metrics --------------------------------------------------------------

_DISEASE_GROUNDEDNESS = GEval(
    name="DiseaseGroundedness",
    criteria=(
        "The retrieval context lists the candidate diseases the pipeline surfaced. "
        "Check that every disease the report presents as a repurposing candidate "
        "appears in that surfaced list. Penalize any candidate disease asserted in "
        "the report but absent from the list. Boilerplate disclaimers and the drug "
        "name itself are not candidate diseases — ignore them."
    ),
    evaluation_params=[_P.ACTUAL_OUTPUT, _P.RETRIEVAL_CONTEXT],
    model=_JUDGE,
    threshold=0.8,
)

_CITATION_GROUNDING = GEval(
    name="CitationGrounding",
    criteria=(
        "The context lists, per disease, the PMIDs the pipeline actually collected "
        "(collected_PMIDs). Check that every PMID cited in the report's prose is "
        "present in the collected_PMIDs for some disease in the context. Penalize "
        "any PMID that appears in the report but in no collected_PMIDs list — that "
        "is a fabricated citation. A disease whose context says collected_PMIDs: "
        "none must cite no PMIDs."
    ),
    evaluation_params=[_P.ACTUAL_OUTPUT, _P.RETRIEVAL_CONTEXT],
    model=_JUDGE,
    threshold=0.9,
)

_EVIDENCE_STRENGTH_FAITHFUL = GEval(
    name="EvidenceStrengthFaithful",
    criteria=(
        "For each disease, the context gives the structured evidence_strength "
        "(strong/moderate/weak/none) and evidence_direction (supports/contradicts/"
        "mixed/none). Check that the report's narrative description of the evidence "
        "for that disease does not overstate it relative to those values — e.g. it "
        "must not describe weak or none evidence as strong, or describe a "
        "contradicting/mixed body as straightforwardly supportive. Matching or "
        "understating is fine; overstating is penalized."
    ),
    evaluation_params=[_P.ACTUAL_OUTPUT, _P.RETRIEVAL_CONTEXT],
    model=_JUDGE,
    threshold=0.8,
)

_NO_EFFICACY_OVERCLAIM = GEval(
    name="NoEfficacyOverclaim",
    criteria=(
        "Penalize any claim in the report that asserts efficacy, proven benefit, or "
        "approval beyond what the context supports. Specifically: a disease with a "
        "development_stage indicating no/early trials must not be described as "
        "proven or clinically established; and the report must not state the drug is "
        "FDA-approved for an indication unless the context says "
        "fda_approved_for_this_indication: True. Hedged, rationale-level language "
        "('may', 'supports the hypothesis', 'untested') is acceptable."
    ),
    evaluation_params=[_P.ACTUAL_OUTPUT, _P.RETRIEVAL_CONTEXT],
    model=_JUDGE,
    threshold=0.8,
)

_DEV_STAGE_CONSISTENCY = GEval(
    name="DevStageConsistency",
    criteria=(
        "The context gives each disease's authoritative development_stage. Check "
        "that the report's prose and risk/assessment language for that disease are "
        "consistent with it — e.g. prose must not describe active late-phase "
        "programs for a disease whose development_stage says no trials are on "
        "record. Penalize contradictions between the prose and the stated stage."
    ),
    evaluation_params=[_P.ACTUAL_OUTPUT, _P.RETRIEVAL_CONTEXT],
    model=_JUDGE,
    threshold=0.8,
)

_METRICS = [
    _DISEASE_GROUNDEDNESS,
    _CITATION_GROUNDING,
    _EVIDENCE_STRENGTH_FAITHFUL,
    _NO_EFFICACY_OVERCLAIM,
    _DEV_STAGE_CONSISTENCY,
]


@pytest.mark.eval
async def test_real_run_report_quality():
    output, report = await run_analysis(_DRUG)

    # No candidates surfaced means there is nothing to be grounded in — that is a
    # pipeline/data problem, not a report-quality failure. Skip rather than judge
    # an empty context (which would make the metrics meaningless).
    if not output.candidate_diseases:
        pytest.skip(f"No candidate diseases surfaced for {_DRUG}; nothing to evaluate.")

    test_case = LLMTestCase(
        input=f"Repurposing report for {output.drug_name}",
        actual_output=report,
        retrieval_context=_context(output),
    )

    assert_test(test_case, _METRICS)
