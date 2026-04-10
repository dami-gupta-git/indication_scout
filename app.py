"""IndicationScout — Streamlit frontend.

Single-page app: enter a drug name, run the supervisor agent, download
the Markdown report.

Run with:
    streamlit run app.py
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from indication_scout.agents.supervisor.supervisor_agent import (
    build_supervisor_agent,
    run_supervisor_agent,
)
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.report.format_report import format_report
from indication_scout.services.retrieval import RetrievalService

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="IndicationScout", layout="centered")
st.title("IndicationScout")
st.caption("Drug repurposing analysis powered by AI agents.")


def _make_db_session():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def _build_agent(db):
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    return build_supervisor_agent(llm, svc=svc, db=db)


async def _run(drug_name: str) -> str:
    db = _make_db_session()
    try:
        agent = _build_agent(db)
        output = await run_supervisor_agent(agent, drug_name)
        return format_report(output)
    finally:
        db.close()


drug_name = st.text_input(
    "Drug name",
    placeholder="e.g. metformin",
    help="Enter the name of the drug to analyse for repurposing opportunities.",
)

if st.button("Analyse", type="primary", disabled=not drug_name.strip()):
    drug_name = drug_name.strip()
    with st.spinner(f"Running repurposing analysis for **{drug_name}**… this may take several minutes."):
        try:
            report_md = asyncio.run(_run(drug_name))
        except Exception as exc:
            logger.exception("Analysis failed for %s", drug_name)
            st.error(f"Analysis failed: {exc}")
            st.stop()

    st.success("Analysis complete.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    filename = f"indication_scout_{drug_name.replace(' ', '_')}_{timestamp}.md"

    st.download_button(
        label="Download report (.md)",
        data=report_md,
        file_name=filename,
        mime="text/markdown",
    )

    with st.expander("Preview report", expanded=True):
        st.markdown(report_md)
