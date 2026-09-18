"""Shadow evaluation for one bounded agentic literature follow-up.

Run this script with the Python environment from the agentic-literature worktree. It does not
modify production code. The baseline and oracle arms call the same tools deterministically; the
agent arm lets the existing ReAct agent decide whether to spend one disease-term search.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolCall
from pydantic import BaseModel, model_validator

from indication_scout.agents.literature import literature_agent, literature_tools
from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.config import get_settings
from indication_scout.db.session import make_session_factory
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.cost_tracking import (
    CostTracker,
    CostTrackingCallback,
    bind_cost_tracker,
    candidate_cost_scope,
    reset_cost_tracker,
)
from indication_scout.services.retrieval import AbstractResult, RetrievalService

logger = logging.getLogger(__name__)

Arm = Literal["baseline", "oracle", "agent"]
Direction = Literal["supports", "contradicts", "mixed", "none"]


class EvalCase(BaseModel):
    case_id: str
    drug: str
    disease: str
    decisive_pmids: list[str]
    expected_direction: Direction
    oracle_disease_terms: list[str]
    label_source: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class EvalCohort(BaseModel):
    cases: list[EvalCase]

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class ExtraSearchAudit(BaseModel):
    terms: list[str]
    reason: str
    queries: list[str]
    new_pmids: int
    entered_shortlist: list[str]


class EvalCost(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    cost_usd: str | None
    unpriced_models: list[str]


class EvalMetrics(BaseModel):
    shortlist_recall: float
    cited_recall: float
    exact_pair_precision: float | None
    direction_match: bool
    unsupported_absence_claim: bool


class EvalCaseRun(BaseModel):
    case: EvalCase
    arm: Arm
    repeat: int
    elapsed_seconds: float
    queries: list[str]
    pool_pmids: list[str]
    shortlist: list[AbstractResult]
    evidence: EvidenceSummary | None
    extra_searches: list[ExtraSearchAudit]
    metrics: EvalMetrics
    cost: EvalCost


class EvalRun(BaseModel):
    generated_at: str
    arm: Arm
    repeat: int
    cases: list[EvalCaseRun]


def _tool_call(name: str, call_id: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args, id=call_id, type="tool_call")


def _cost_snapshot(tracker: CostTracker) -> EvalCost:
    totals = tracker.snapshot().total
    cost: str | None = None
    if not totals.unpriced_models:
        cost = str(totals.cost_usd.quantize(Decimal("0.000001")))
    return EvalCost(
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cache_read_tokens=totals.cache_read_tokens,
        cache_write_5m_tokens=totals.cache_write_5m_tokens,
        cache_write_1h_tokens=totals.cache_write_1h_tokens,
        cost_usd=cost,
        unpriced_models=sorted(totals.unpriced_models),
    )


def _extra_searches(output: LiteratureOutput) -> list[ExtraSearchAudit]:
    records = getattr(output, "extra_searches", [])
    return [
        ExtraSearchAudit.model_validate(record, from_attributes=True)
        for record in records
    ]


def _absence_claim(evidence: EvidenceSummary | None, missing_decisive: bool) -> bool:
    if evidence is None or not missing_decisive:
        return False
    text = evidence.summary.lower()
    phrases = (
        "no published",
        "no clinical evidence",
        "no evidence",
        "no studies",
        "has not been studied",
        "no efficacy",
    )
    return any(phrase in text for phrase in phrases)


def _metrics(case: EvalCase, output: LiteratureOutput) -> EvalMetrics:
    decisive = set(case.decisive_pmids)
    shortlist_pmids = {result.pmid for result in output.semantic_search_results}
    evidence = output.evidence_summary
    cited_pmids: set[str] = set()
    exact_pair_precision: float | None = None
    direction_match = False
    if evidence is not None:
        cited_pmids = (
            set(evidence.supporting_pmids)
            | set(evidence.contradicting_pmids)
            | set(evidence.neutral_pmids)
        )
        judged = len(evidence.relevant_pmids) + len(evidence.contaminated_pmids)
        if judged:
            exact_pair_precision = len(evidence.relevant_pmids) / judged
        direction_match = evidence.direction == case.expected_direction
    shortlist_recall = len(decisive & shortlist_pmids) / len(decisive)
    cited_recall = len(decisive & cited_pmids) / len(decisive)
    return EvalMetrics(
        shortlist_recall=shortlist_recall,
        cited_recall=cited_recall,
        exact_pair_precision=exact_pair_precision,
        direction_match=direction_match,
        unsupported_absence_claim=_absence_claim(
            evidence, missing_decisive=not decisive.issubset(cited_pmids)
        ),
    )


async def _run_fixed_arm(
    arm: Literal["baseline", "oracle"],
    case: EvalCase,
    llm: ChatAnthropic,
    svc: RetrievalService,
    db: Any,
) -> LiteratureOutput:
    tools = {
        tool.name: tool
        for tool in literature_tools.build_literature_tools(svc=svc, db=db)
    }
    profile = await tools["build_drug_profile"].ainvoke(
        _tool_call("build_drug_profile", "profile", drug_name=case.drug)
    )
    if profile.artifact is None:
        raise RuntimeError(f"No drug profile returned for {case.case_id}")
    expanded = await tools["expand_search_terms"].ainvoke(
        _tool_call(
            "expand_search_terms",
            "expand",
            drug_name=case.drug,
            disease_name=case.disease,
        )
    )
    fetched = await tools["fetch_and_cache"].ainvoke(
        _tool_call("fetch_and_cache", "fetch", drug_name=case.drug)
    )
    searched = await tools["semantic_search"].ainvoke(
        _tool_call(
            "semantic_search",
            "search",
            drug_name=case.drug,
            disease_name=case.disease,
        )
    )
    extra: list[ExtraSearchAudit] = []
    if arm == "oracle":
        followed = await tools["search_additional_terms"].ainvoke(
            _tool_call(
                "search_additional_terms",
                "followup",
                drug_name=case.drug,
                disease_terms=case.oracle_disease_terms,
                reason="Verified disease terminology supplied by the evaluation cohort.",
            )
        )
        record = followed.artifact.record
        extra = [ExtraSearchAudit.model_validate(record, from_attributes=True)]
        fetched = followed.artifact
        searched = followed.artifact
    await tools["safety_search"].ainvoke(
        _tool_call(
            "safety_search",
            "safety",
            drug_name=case.drug,
            disease_name=case.disease,
        )
    )
    synthesized = await tools["synthesize"].ainvoke(
        _tool_call(
            "synthesize",
            "synthesize",
            drug_name=case.drug,
            disease_name=case.disease,
        )
    )
    if arm == "oracle":
        pool_pmids = fetched.pmids
        abstracts = searched.abstracts
    else:
        pool_pmids = fetched.artifact
        abstracts = searched.artifact
    output = LiteratureOutput(
        search_results=expanded.artifact,
        pmids=pool_pmids,
        semantic_search_results=abstracts,
        evidence_summary=synthesized.artifact,
        summary="",
    )
    if hasattr(output, "extra_searches"):
        output.extra_searches = extra
    return output


async def _run_case(
    arm: Arm,
    repeat: int,
    case: EvalCase,
    llm: ChatAnthropic,
    svc: RetrievalService,
    session_factory: Any,
) -> EvalCaseRun:
    tracker = CostTracker()
    token = bind_cost_tracker(tracker)
    started = time.perf_counter()
    try:
        with candidate_cost_scope(case.case_id), session_factory() as db:
            if arm == "agent":
                agent = build_literature_agent(llm=llm, svc=svc, db=db)
                output = await run_literature_agent(agent, case.drug, case.disease)
            else:
                output = await _run_fixed_arm(arm, case, llm, svc, db)
    finally:
        reset_cost_tracker(token)
    elapsed = time.perf_counter() - started
    logger.info(
        "Completed %s repeat=%d case=%s in %.1fs",
        arm,
        repeat,
        case.case_id,
        elapsed,
    )
    return EvalCaseRun(
        case=case,
        arm=arm,
        repeat=repeat,
        elapsed_seconds=elapsed,
        queries=output.search_results,
        pool_pmids=output.pmids,
        shortlist=output.semantic_search_results,
        evidence=output.evidence_summary,
        extra_searches=_extra_searches(output),
        metrics=_metrics(case, output),
        cost=_cost_snapshot(tracker),
    )


async def _run(args: argparse.Namespace) -> None:
    cohort = EvalCohort.model_validate_json(args.cohort.read_text(encoding="utf-8"))
    selected = cohort.cases
    if args.case:
        selected = [case for case in selected if case.case_id in set(args.case)]
    if not selected:
        raise ValueError("No evaluation cases matched --case")

    # The prototype allows two calls. The evaluated intervention is intentionally one bounded
    # decision, so both the runtime budget and the agent-facing prompt are narrowed in memory.
    literature_tools.__dict__["LITERATURE_EXTRA_SEARCH_BUDGET"] = 1
    literature_agent.SYSTEM_PROMPT = literature_agent.SYSTEM_PROMPT.replace(
        "at most 2 calls", "at most 1 call"
    ).replace("Up to two calls", "At most one call")

    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
        callbacks=[CostTrackingCallback(settings.llm_model)],
    )
    svc = RetrievalService(args.cache_dir)
    session_factory = make_session_factory()
    try:
        runs = []
        for case in selected:
            runs.append(
                await _run_case(
                    args.arm,
                    args.repeat,
                    case,
                    llm,
                    svc,
                    session_factory,
                )
            )
        result = EvalRun(
            generated_at=datetime.now(UTC).isoformat(),
            arm=args.arm,
            repeat=args.repeat,
            cases=runs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.output)
    finally:
        session_factory.kw["bind"].dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "oracle", "agent"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
