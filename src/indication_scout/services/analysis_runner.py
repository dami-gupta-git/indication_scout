"""Analysis runner — invokes the supervisor agent for a drug, outside any UI.

The single shared entry point for running the pipeline: both the CLI (`scout find`) and the
FastAPI layer call `run_analysis`. No duplicate agent-building wiring. Phase 1 of PLAN_react.md
uses the existing blocking `run_supervisor_agent` (`ainvoke`) — no orchestration touch. Phase 9
swaps the inner call to a streaming variant.
"""

import logging
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from sqlalchemy.orm import Session

from indication_scout.agents.supervisor.supervisor_agent import (
    build_supervisor_agent,
    run_supervisor_agent,
)
from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import get_db
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.report.format_report import format_report
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_agent(
    db: Session,
    date_before: date | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
):
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(cache_dir)
    return build_supervisor_agent(llm, svc=svc, db=db, date_before=date_before)


async def run_analysis(
    drug_name: str, *, date_before: date | None = None
) -> tuple[SupervisorOutput, str]:
    """Run the supervisor agent for `drug_name`; return its output and the Markdown report.

    Normalizes the drug name at the entry point (cache keys, tools, sub-agents, logs all see the
    same lowercased form), owns the DB session lifecycle (opened per run, always closed), and
    threads `date_before` for holdout runs. Blocking `ainvoke` path — no orchestration touch
    (PLAN_react.md Phase 1).
    """
    drug = normalize_drug_name(drug_name)
    db = next(get_db())
    try:
        agent, get_merged_allowlist, get_auto_findings = build_agent(db, date_before=date_before)
        output = await run_supervisor_agent(
            agent, get_merged_allowlist, drug, get_auto_findings=get_auto_findings
        )
        return output, format_report(output)
    finally:
        db.close()
