"""Format a SupervisorOutput as a Markdown report."""

from datetime import datetime

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.agents.clinical_trials.clinical_trials_output import ClinicalTrialsOutput
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


def _fmt_literature(lit: LiteratureOutput) -> str:
    lines: list[str] = []

    if lit.evidence_summary:
        es = lit.evidence_summary
        lines.append(f"**Evidence strength:** {es.strength}")
        lines.append(f"**Study count:** {es.study_count}")
        if es.study_types:
            lines.append(f"**Study types:** {', '.join(es.study_types)}")
        lines.append(f"**Adverse effects reported:** {'Yes' if es.has_adverse_effects else 'No'}")
        if es.summary:
            lines.append(f"\n{es.summary}")
        if es.key_findings:
            lines.append("\n**Key findings:**")
            for finding in es.key_findings:
                lines.append(f"- {finding}")
        if es.supporting_pmids:
            pmid_links = ", ".join(
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                for pmid in es.supporting_pmids
            )
            lines.append(f"\n**Supporting PMIDs:** {pmid_links}")
    else:
        lines.append("_No evidence summary available._")

    return "\n".join(lines)


def _fmt_clinical_trials(ct: ClinicalTrialsOutput) -> str:
    lines: list[str] = []

    if ct.summary:
        lines.append(ct.summary)

    if ct.whitespace:
        ws = ct.whitespace
        counts = []
        if ws.exact_match_count is not None:
            counts.append(f"{ws.exact_match_count} exact matches")
        if ws.drug_only_trials is not None:
            counts.append(f"{ws.drug_only_trials} drug-only")
        if ws.indication_only_trials is not None:
            counts.append(f"{ws.indication_only_trials} indication-only")
        if counts:
            lines.append(f"\n**Trial counts:** {', '.join(counts)}")
        if ws.is_whitespace:
            lines.append("**Whitespace detected** — limited competition in this indication.")
        if ws.indication_drugs:
            drug_names = ", ".join(d.drug_name for d in ws.indication_drugs if d.drug_name)
            if drug_names:
                lines.append(f"**Competing drugs:** {drug_names}")

    if ct.landscape and ct.landscape.competitors:
        lines.append(f"\n**Competitive landscape ({len(ct.landscape.competitors)} competitors):**")
        for comp in ct.landscape.competitors[:10]:
            lines.append(f"- {comp.drug_name} ({comp.sponsor}) — {comp.max_phase}, {comp.trial_count} trial(s)")

    if ct.trials:
        lines.append(f"\n**Active / completed trials ({len(ct.trials)}):**")
        for trial in ct.trials[:10]:
            phase = trial.phase or "Unknown phase"
            status = trial.overall_status or ""
            lines.append(f"- [{trial.nct_id}](https://clinicaltrials.gov/study/{trial.nct_id}) — {trial.title} ({phase}{', ' + status if status else ''})")

    if ct.terminated:
        lines.append(f"\n**Terminated / withdrawn trials ({len(ct.terminated)}):**")
        for t in ct.terminated[:10]:
            reason = f" — *{t.why_stopped}*" if t.why_stopped else ""
            category = f" [{t.stop_category}]" if t.stop_category else ""
            title = f" {t.title}" if t.title else ""
            lines.append(f"- [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id}){title} ({t.phase}){category}{reason}")

    if not lines:
        lines.append("_No clinical trials data available._")

    return "\n".join(lines)


def _fmt_mechanism(mech: MechanismOutput) -> str:
    lines: list[str] = []

    if mech.summary:
        lines.append(mech.summary)

    if mech.drug_targets:
        targets = ", ".join(sorted(mech.drug_targets.keys()))
        lines.append(f"\n**Molecular targets:** {targets}")

    if mech.mechanisms_of_action:
        lines.append("\n**Mechanisms of action:**")
        for moa in mech.mechanisms_of_action:
            syms = ", ".join(moa.target_symbols) if moa.target_symbols else "—"
            lines.append(f"- {moa.mechanism_of_action} ({moa.action_type}) → {syms}")

    if mech.shaped_associations:
        lines.append("\n**Mechanistic associations:**")
        for sa in mech.shaped_associations:
            shape_label = {
                "hypothesis": "Repurposing hypothesis",
                "contraindication": "Contraindication signal",
                "confirms_known": "Confirms known use",
                "neutral": "Neutral",
            }.get(sa.shape, sa.shape)
            lines.append(f"- **{sa.target_symbol} / {sa.disease_name}** [{shape_label}]: {sa.rationale}")

    if mech.pathways:
        lines.append("\n**Reactome pathways (per target):**")
        for symbol, pathway_list in sorted(mech.pathways.items()):
            if pathway_list:
                names = ", ".join(p.pathway_name for p in pathway_list[:5] if p.pathway_name)
                lines.append(f"- **{symbol}:** {names}")

    return "\n".join(lines)


def format_report(output: SupervisorOutput) -> str:
    """Render a SupervisorOutput as a Markdown string."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    drug = output.drug_name or "Unknown drug"

    lines: list[str] = [
        f"# IndicationScout Report: {drug}",
        f"_Generated {now}_",
        "",
        "---",
        "",
    ]

    # Summary
    lines += [
        "## Summary",
        "",
        output.summary if output.summary else "_No summary produced._",
        "",
        "---",
        "",
    ]

    # Candidate diseases
    lines += ["## Candidate Diseases", ""]
    if output.candidates:
        for c in output.candidates:
            lines.append(f"- {c}")
    else:
        lines.append("_No candidates surfaced._")
    lines += ["", "---", ""]

    # Mechanism
    lines += ["## Mechanistic Analysis", ""]
    if output.mechanism:
        lines.append(_fmt_mechanism(output.mechanism))
    else:
        lines.append("_Mechanism analysis not run._")
    lines += ["", "---", ""]

    # Per-candidate findings
    lines += ["## Candidate Findings", ""]
    if output.findings:
        for finding in output.findings:
            lines += [f"### {finding.disease}", ""]

            if finding.literature:
                lines += ["#### Literature", "", _fmt_literature(finding.literature), ""]

            if finding.clinical_trials:
                lines += ["#### Clinical Trials", "", _fmt_clinical_trials(finding.clinical_trials), ""]

            lines.append("---")
            lines.append("")
    else:
        lines.append("_No candidate findings produced._")

    return "\n".join(lines)
