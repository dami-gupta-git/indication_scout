"""IndicationScout — Streamlit frontend.

Single-page app: enter a drug name, run the supervisor agent, download
the Markdown report.

Run with:
    streamlit run app.py
"""

# healthcheck MUST be imported first so the HTTP server on port 8000 is up
# before Streamlit initialises. Railway polls /health to decide when the
# deployment is ready to receive traffic.
import healthcheck  # noqa: F401

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# load_dotenv MUST run before any indication_scout imports, because
# module-level code in base_client.py calls get_settings() at import time.
# Manually edit the constants filename below to switch tunable-limit profiles.
CONSTANTS_FILE = ".env.constants"
load_dotenv(Path(__file__).parent / ".env")
constants_path = Path(__file__).parent / CONSTANTS_FILE
load_dotenv(constants_path)
# Also export so config.py's pydantic Settings reads the same file.
os.environ["CONSTANTS_FILE"] = str(constants_path)

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.report.format_report import _splice_blurbs_into_summary
from indication_scout.services.analysis_runner import run_analysis
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="IndicationScout", page_icon="static/favicon.png", layout="wide")


def _title_case(name: str) -> str:
    """Capitalize the first letter of each whitespace-separated word, leaving the
    rest of each word untouched so acronyms (e.g. NSCLC) and possessives (e.g.
    Alzheimer's) are preserved."""
    return " ".join(w[:1].upper() + w[1:] if w else w for w in name.split(" "))


# ---------------------------------------------------------------------------
# Sidebar — inputs and run controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("static/favicon.png", width=60)
    st.markdown("## Indication Scout")
    st.caption("Drug repurposing analysis powered by AI agents.")

    st.markdown("---")

    drug_name = st.text_input(
        "Drug name",
        placeholder="e.g. metformin",
        help="Enter the name of the drug to analyse for repurposing opportunities.",
    )

    run_clicked = st.button(
        "Analyse",
        type="primary",
        disabled=not drug_name.strip(),
        width="stretch",
    )

    if run_clicked:
        drug_name = drug_name.strip()
        with st.spinner(f"Running analysis for **{drug_name}**… this may take several minutes."):
            try:
                output, report_md = asyncio.run(run_analysis(drug_name))
            except Exception as exc:
                logger.exception("Analysis failed for %s", drug_name)
                st.error(f"Analysis failed: {exc}")
                st.stop()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename = f"indication_scout_{drug_name.replace(' ', '_')}_{timestamp}.md"

        st.session_state["supervisor_output"] = output
        st.session_state["report_md"] = report_md
        st.session_state["report_filename"] = filename
        st.session_state["analysed_drug"] = drug_name

    # Disease focus selector — only shown once a run has produced findings
    if "supervisor_output" in st.session_state:
        output = st.session_state["supervisor_output"]
        investigated = [f.disease for f in output.disease_findings]
        if investigated:
            st.markdown("---")
            st.markdown("### Disease focus")
            st.caption("Drives Clinical Trials and Literature panels.")
            st.radio(
                "Investigated diseases",
                options=investigated,
                key="selected_disease",
                label_visibility="collapsed",
                format_func=_title_case,
            )

        st.markdown("---")
        st.download_button(
            label="Download report (.md)",
            data=st.session_state["report_md"],
            file_name=st.session_state["report_filename"],
            mime="text/markdown",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Main area — top band and tabs
# ---------------------------------------------------------------------------

if "supervisor_output" not in st.session_state:
    st.title("Indication Scout")
    st.caption("Drug repurposing analysis powered by AI agents.")
    st.info("Enter a drug name in the sidebar and click **Analyse** to begin.")
    st.stop()

output = st.session_state["supervisor_output"]
analysed_drug = st.session_state["analysed_drug"]

st.title(f"Indication Scout — {analysed_drug}")
st.caption(f"_Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")

# Top-band KPIs
total_trials = sum(
    f.clinical_trials.search.total_count
    for f in output.disease_findings
    if f.clinical_trials and f.clinical_trials.search
)
total_studies = sum(
    f.literature.evidence_summary.study_count
    for f in output.disease_findings
    if f.literature and f.literature.evidence_summary
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidate diseases", len(output.candidate_diseases))
m2.metric("Investigated", len(output.disease_findings))
m3.metric("Total trials", total_trials)
m4.metric("Total studies", total_studies)

st.markdown("---")

tab_overview, tab_mech, tab_trials, tab_lit = st.tabs(
    ["Overview", "Mechanism", "Clinical Trials", "Literature"]
)

investigated_set = {f.disease.lower().strip() for f in output.disease_findings}
findings_by_disease = {f.disease: f for f in output.disease_findings}
selected_disease = st.session_state.get("selected_disease")


# ----- Overview tab -----
with tab_overview:
    st.markdown("### Summary")
    summary_text = (
        _splice_blurbs_into_summary(output.summary, output.disease_findings)
        if output.summary
        else "_No summary produced._"
    )
    st.markdown(summary_text)

    st.markdown("### Candidate diseases")
    if output.candidate_diseases:
        for c in output.candidate_diseases:
            marker = "✓ investigated" if c.lower().strip() in investigated_set else "_not investigated_"
            st.markdown(f"- **{c}** — {marker}")
    else:
        st.markdown("_No candidates surfaced._")

    if selected_disease:
        st.info(
            f"Use the sidebar to switch focus disease. Currently viewing: "
            f"**{selected_disease}** in the Clinical Trials and Literature tabs."
        )

    with st.expander("Full markdown report", expanded=False):
        st.markdown(st.session_state["report_md"])


# ----- Mechanism tab -----
with tab_mech:
    if output.mechanism is None:
        st.markdown("_Mechanism analysis not run._")
    else:
        mech = output.mechanism
        st.markdown("### Mechanistic analysis")
        if mech.summary:
            st.write(mech.summary)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("#### Molecular targets")
            if mech.drug_targets:
                for t in sorted(mech.drug_targets.keys()):
                    st.markdown(f"- `{t}`")
            else:
                st.markdown("_No targets identified._")

        with col_b:
            st.markdown("#### Mechanisms of action")
            if mech.mechanisms_of_action:
                moa_rows = [
                    {
                        "Mechanism": moa.mechanism_of_action,
                        "Action type": moa.action_type or "",
                        "Targets": ", ".join(moa.target_symbols) if moa.target_symbols else "—",
                    }
                    for moa in mech.mechanisms_of_action
                ]
                st.dataframe(moa_rows, hide_index=True, width="stretch")
            else:
                st.markdown("_No mechanisms of action recorded._")

        if mech.candidates:
            st.markdown("#### Repurposing candidates from mechanism")
            for c in mech.candidates:
                with st.container(border=True):
                    st.markdown(
                        f"**{c.target_symbol} ({c.action_type}) → {c.disease_name}**"
                    )
                    if c.disease_description:
                        st.caption(c.disease_description)
                    if c.target_function:
                        st.caption(f"_Target function:_ {c.target_function}")


# ----- Clinical Trials tab -----
with tab_trials:
    if not selected_disease:
        st.info("_No investigated diseases — run an analysis first._")
    elif selected_disease not in findings_by_disease:
        st.info("_Selected disease has no findings._")
    else:
        finding = findings_by_disease[selected_disease]
        ct = finding.clinical_trials

        st.markdown(f"### Clinical trials — {selected_disease}")
        st.caption(f"Source: {finding.source}")

        if ct is None:
            st.markdown("_No clinical trials data available._")
        else:
            if ct.summary:
                st.write(ct.summary)

            search_total = ct.search.total_count if ct.search else 0
            by_status = ct.search.by_status if ct.search else {}
            a1, a2, a3 = st.columns(3)
            a1.metric("Total trials", search_total)
            a2.metric("Recruiting", by_status.get("RECRUITING", 0))
            a3.metric("Active (not recruiting)", by_status.get("ACTIVE_NOT_RECRUITING", 0))

            if by_status:
                st.markdown("#### Status breakdown")
                sorted_statuses = dict(sorted(by_status.items(), key=lambda kv: -kv[1]))
                st.bar_chart(sorted_statuses, horizontal=True)

            if ct.completed and ct.completed.trials:
                st.markdown(
                    f"#### Completed trials ({ct.completed.total_count} total)"
                )
                completed_rows = [
                    {
                        "NCT": f"https://clinicaltrials.gov/study/{t.nct_id}",
                        "Title": t.title,
                        "Phase": t.phase or "Unknown",
                        "Status": t.overall_status,
                    }
                    for t in ct.completed.trials[:25]
                ]
                st.dataframe(
                    completed_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "NCT": st.column_config.LinkColumn("NCT", display_text=r"NCT\d+"),
                    },
                )

            if ct.terminated and ct.terminated.trials:
                st.markdown(f"#### Terminated trials ({ct.terminated.total_count})")
                for t in ct.terminated.trials[:15]:
                    with st.container(border=True):
                        st.markdown(
                            f"[{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id})"
                            f" — {t.title or '_no title_'}"
                        )
                        category = _classify_stop_reason(t.why_stopped)
                        bits = [t.phase or "Unknown phase", category]
                        if t.why_stopped:
                            bits.append(f"_{t.why_stopped}_")
                        st.caption(" · ".join(bits))

            if ct.landscape and ct.landscape.competitors:
                st.markdown(
                    f"#### Competitive landscape ({len(ct.landscape.competitors)})"
                )
                comp_rows = [
                    {
                        "Drug": c.drug_name,
                        "Sponsor": c.sponsor,
                        "Max phase": c.max_phase,
                        "Trials": c.trial_count,
                    }
                    for c in ct.landscape.competitors[:25]
                ]
                st.dataframe(comp_rows, hide_index=True, width="stretch")


# ----- Literature tab -----
with tab_lit:
    if not selected_disease:
        st.info("_No investigated diseases — run an analysis first._")
    elif selected_disease not in findings_by_disease:
        st.info("_Selected disease has no findings._")
    else:
        finding = findings_by_disease[selected_disease]

        st.markdown(f"### Literature — {selected_disease}")
        st.caption(f"Source: {finding.source}")

        if finding.literature is None or finding.literature.evidence_summary is None:
            st.markdown("_No evidence summary available._")
        else:
            lit = finding.literature.evidence_summary
            l1, l2 = st.columns(2)
            l1.metric("Evidence strength", lit.strength.title())
            l2.metric("Study count", lit.study_count)

            if lit.summary:
                st.markdown("#### Summary")
                st.write(lit.summary)

            if lit.key_findings:
                st.markdown("#### Key findings")
                for f in lit.key_findings:
                    st.markdown(f"- {f}")

            if lit.supporting_pmids:
                st.markdown(f"#### Supporting PMIDs ({len(lit.supporting_pmids)})")
                pmid_links = " · ".join(
                    f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)"
                    for p in lit.supporting_pmids
                )
                st.markdown(pmid_links)
