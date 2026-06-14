"""Format a SupervisorOutput as a Markdown report."""

import re
from datetime import datetime

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput


def _title_case_disease(name: str) -> str:
    """Capitalize the first letter of each whitespace-separated word, leaving the
    rest of each word untouched so acronyms (e.g. NSCLC) and possessives (e.g.
    Alzheimer's) are preserved."""
    return " ".join(w[:1].upper() + w[1:] if w else w for w in name.split(" "))


def _fmt_literature(lit: LiteratureOutput) -> str:
    lines: list[str] = []

    if lit.evidence_summary:
        es = lit.evidence_summary
        strength_line = f"**Evidence strength:** {es.strength}"
        if es.direction != "none":
            strength_line += f", {es.direction}"
        lines.append(strength_line)
        lines.append(f"**Relevant studies:** {es.study_count}")
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
        if es.contradicting_pmids:
            pmid_links = ", ".join(
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                for pmid in es.contradicting_pmids
            )
            lines.append(f"\n**Contradicting PMIDs:** {pmid_links}")
    else:
        lines.append("_No evidence summary available._")

    return "\n".join(lines)


# Relationships where the trial/literature artifacts are contaminated by an approved subtype
# the registry pulls in (e.g. PAH under a "Hypertension" search) and the subtype cannot be
# cleanly separated. The example trial tables are suppressed for these — a curated-looking list
# is misleading when most of it is the approved subtype's trials.
_CONTAMINATED_RELATIONSHIPS = frozenset({"broader_distinct", "broader_overlapping"})


def _fmt_clinical_trials(
    ct: ClinicalTrialsOutput,
    indication: str = "",
    approval_relationship: str = "",
) -> str:
    lines: list[str] = []
    suppress_trial_tables = approval_relationship in _CONTAMINATED_RELATIONSHIPS

    if ct.summary:
        lines.append(ct.summary)

    if ct.contaminated_nct_ids:
        n = len(ct.contaminated_nct_ids)
        lines.append(
            f"\n_{n} trial(s) excluded as a different indication: "
            f"{', '.join(ct.contaminated_nct_ids)}._"
        )

    if ct.approval:
        a = ct.approval
        if a.label_found:
            if a.is_approved:
                target = a.matched_indication or "this indication"
                lines.append(f"**FDA approval:** Approved ({target})")
            else:
                lines.append(
                    "**FDA approval:** Not found on FDA label for this indication"
                )
        else:
            names = ", ".join(a.drug_names_checked) if a.drug_names_checked else "drug"
            lines.append(
                f"**FDA approval:** No FDA label found for {names} — status undetermined"
            )

    if ct.search:
        s = ct.search
        lines.append(
            f"\n**Trial activity:** {s.total_count} total trial(s) for this pair"
        )
        if s.total_count == 0:
            lines.append(
                "- _Whitespace: no trials found for this drug × indication pair._"
            )

    # Rendered example trials skip contamination (trials the CT agent judged a different
    # indication pulled in by the recall-first search). The total_count headers stay verbatim
    # per the broader_distinct rule — only the listed examples are filtered.
    contaminated = set(ct.contaminated_nct_ids)

    if suppress_trial_tables:
        # Show the verbatim total but no example list — the artifact is dominated by the
        # approved subtype's trials and cannot be cleanly filtered to this indication.
        if ct.completed:
            lines.append(
                f"\n**Completed trials ({ct.completed.total_count} total):** "
                "not listed — trial record contaminated by approved subtype; "
                "see the demotion note in the summary."
            )
        if ct.terminated and ct.terminated.total_count:
            lines.append(
                f"\n**Terminated trials ({ct.terminated.total_count}):** "
                "not listed — trial record contaminated by approved subtype."
            )
        if not lines:
            lines.append("_No clinical trials data available._")
        return "\n".join(lines)

    if ct.completed:
        c = ct.completed
        n_excluded = sum(1 for t in c.trials if t.nct_id in contaminated)
        excl_note = (
            f", {n_excluded} excluded as a different indication"
            if n_excluded
            else ""
        )
        lines.append(f"\n**Completed trials ({c.total_count} total{excl_note}):**")
        shown = [t for t in c.trials if t.nct_id not in contaminated]
        for trial in shown[:10]:
            phase = trial.phase or "Unknown phase"
            status = trial.overall_status or ""
            lines.append(
                f"- [{trial.nct_id}](https://clinicaltrials.gov/study/{trial.nct_id}) — {trial.title} ({phase}{', ' + status if status else ''})"
            )

    if ct.terminated:
        term = ct.terminated
        if term.total_count:
            n_excluded = sum(1 for t in term.trials if t.nct_id in contaminated)
            excl_note = (
                f", {n_excluded} excluded as a different indication"
                if n_excluded
                else ""
            )
            lines.append(
                f"\n**Terminated trials ({term.total_count}{excl_note}):**"
            )
            shown = [t for t in term.trials if t.nct_id not in contaminated]
            for t in shown[:10]:
                reason = f" — *{t.why_stopped}*" if t.why_stopped else ""
                title = f" {t.title}" if t.title else ""
                phase = t.phase or "Unknown phase"
                classified = _classify_stop_reason(t.why_stopped)
                category = f" [{classified}]" if classified != t.why_stopped else ""
                lines.append(
                    f"- [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id}){title} ({phase}){category}{reason}"
                )

    if not lines:
        lines.append("_No clinical trials data available._")

    return "\n".join(lines)


def _title_case_known_diseases(text: str, disease_names: list[str]) -> str:
    """Replace every case-insensitive occurrence of each known disease name in ``text``
    with its title-cased form. Longest names first so multi-word names aren't shadowed
    by their substrings (e.g. "non-small cell lung cancer" before "lung cancer")."""
    if not text or not disease_names:
        return text
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in disease_names:
        if not name:
            continue
        key = name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    unique_names.sort(key=len, reverse=True)
    for name in unique_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub(_title_case_disease(name), text)
    return text


_BLURB_TABLE_FIELDS: list[tuple[str, str]] = [
    ("stage", "Stage"),
    ("literature", "Literature"),
    ("blocker", "Constraint"),
    ("active_programs", "Active programs"),
    ("key_risk", "Key risk"),
    ("verdict", "Assessment"),
]


def _escape_table_cell(value: str) -> str:
    """Escape characters that would break a markdown table cell.

    Pipes need backslash-escaping; embedded newlines collapse to a space (the
    LLM is instructed to keep field values on one line, but defensive).
    """
    return value.replace("|", r"\|").replace("\n", " ").strip()


def _render_blurb(blurb: CandidateBlurb) -> list[str]:
    """Render a CandidateBlurb as markdown lines.

    Layout per candidate:
        1. A 2-column markdown table with one row per non-empty structured field
           (Stage, Blocker, Active programs, Key risk, Verdict). The header row
           is intentionally blank so the table reads as a label/value pair list.
        2. A single bold-prefix `**Watch:** ...` line, only if `watch` is set.
        3. The 2-sentence italicized prose, only if set.

    If every structured field AND the prose are empty, returns an empty list
    (caller falls through to the unchanged ranked line).
    """
    table_rows: list[tuple[str, str]] = []
    for attr, label in _BLURB_TABLE_FIELDS:
        value = getattr(blurb, attr, "").strip()
        if value:
            table_rows.append((label, _escape_table_cell(value)))
    watch = blurb.watch.strip()
    prose = blurb.prose.strip()
    if not table_rows and not watch and not prose:
        return []

    out: list[str] = []
    if table_rows:
        # Empty header row → CommonMark still renders a valid table; the visible
        # output looks like a label/value pair list with column alignment.
        out.append("|  |  |")
        out.append("|---|---|")
        for label, value in table_rows:
            out.append(f"| **{label}** | {value} |")
    if watch:
        if out:
            out.append("")
        out.append(f"**Watch:** {_escape_table_cell(watch)}")
    if prose:
        if out:
            out.append("")
        out.append(f"_{prose}_")
    return out


def _splice_blurbs_into_summary(summary: str, findings: list[CandidateFindings]) -> str:
    """Replace each ranked summary line's structured tail with the matching blurb.

    The supervisor's summary string is a ranked list of the form
    `N. <disease> — literature: ..., trials: ...`. For each line
    that matches a finding with a populated CandidateBlurb, the structured tail
    (everything from the em-dash onward) is stripped and the blurb is rendered
    underneath as: structured fields (only non-empty ones), then the 2-sentence
    prose. Lines that don't match any finding (e.g. the trailing "Closed signals:"
    line) and lines without an em-dash are passed through unchanged. Disease
    matching is case-insensitive on the disease name only.
    """
    blurb_by_disease: dict[str, CandidateBlurb] = {}
    for f in findings:
        if f.blurb is None:
            continue
        rendered = _render_blurb(f.blurb)
        if not rendered:
            continue
        blurb_by_disease[f.disease.lower().strip()] = f.blurb
    if not blurb_by_disease:
        return summary

    # Group "rank": the rank number ("1", "2", ...). Group "head": disease portion
    # (before em-dash). Any leading whitespace the LLM emitted is dropped — the rank
    # line is rebuilt flush-left so it lines up with the field/prose lines rendered
    # below it (and so CommonMark doesn't treat 4+ leading spaces as a code block).
    rank_line = re.compile(r"^\s*(?P<rank>\d+)\.\s+(?P<head>.+?)\s+—\s+.+$")
    footer_line = re.compile(
        r"^\s*(?:Demoted\s+—|Closed\s+signals\s*:|Evidence\s+gate\s+exclusions\s*:)",
        re.IGNORECASE,
    )
    out_lines: list[str] = []
    footer_separator_emitted = False
    for line in summary.splitlines():
        if not footer_separator_emitted and footer_line.match(line):
            # Separator between the blurb stack and the footer block so the
            # demoted/closed/exclusions lines don't sit flush against the last
            # blurb's prose.
            out_lines.append("---")
            out_lines.append("")
            footer_separator_emitted = True
        m = rank_line.match(line)
        if m is None:
            out_lines.append(line)
            continue
        head = m.group("head")
        head_lower = head.lower()
        match_key: str | None = None
        # Longest-match avoids "lung cancer" stealing a match meant for
        # "non-small cell lung cancer".
        for key in sorted(blurb_by_disease.keys(), key=len, reverse=True):
            if key and key in head_lower:
                match_key = key
                break
        if match_key is None:
            out_lines.append(line)
            continue
        blurb = blurb_by_disease.pop(match_key)
        rank = m.group("rank")
        # Markdown collapses adjacent non-blank lines into one paragraph, so emit
        # a blank line between the ranked disease line and the blurb block (and a
        # trailing blank so the next ranked entry doesn't fold back into this one).
        out_lines.append(f"{rank}. {_title_case_disease(head)}")
        out_lines.append("")
        out_lines.extend(_render_blurb(blurb))
        out_lines.append("")
    return "\n".join(out_lines)


def format_report(output: SupervisorOutput) -> str:
    """Render a SupervisorOutput as a Markdown string."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    drug = output.drug_name or "Unknown drug"

    lines: list[str] = [
        f"# IndicationScout Report: {_title_case_disease(drug)}",
        f"_Generated {now}_",
        "",
        "_Not for clinical use; for research purposes only_",
        "",
        "---",
        "",
    ]

    # Summary — render the supervisor's prose intro, then each top-disease as a ranked
    # entry with its blurb pulled from disease_findings. Title-case any known disease
    # names that appear inside the LLM-generated prose.
    known_diseases = [f.disease for f in output.disease_findings] + list(
        output.candidate_diseases or []
    )
    findings_by_canonical = {
        f.disease.lower().strip(): f for f in output.disease_findings
    }

    lines += ["## Summary", ""]
    if output.summary:
        spliced = _splice_blurbs_into_summary(output.summary, output.disease_findings)
        lines.append(_title_case_known_diseases(spliced, known_diseases))
        lines.append("")
    elif output.top_diseases:
        for rank, disease in enumerate(output.top_diseases, start=1):
            finding = findings_by_canonical.get(disease.lower().strip())
            lines.append(f"{rank}. {_title_case_disease(disease)}")
            lines.append("")
            if finding is not None and finding.blurb is not None:
                lines.extend(_render_blurb(finding.blurb))
            lines.append("")
    else:
        lines.append("_No summary produced._")
        lines.append("")

    lines += [
        "_Note: trial counts in this summary reflect ClinicalTrials.gov only and may "
        "undercount activity registered in ex-US registries (e.g. jRCT, ChiCTR, "
        "EU-CTR, ANZCTR). Studies cited in the literature section may reference "
        "trials in those registries that are not represented in the trial counts above._",
        "",
        "---",
        "",
    ]

    # Candidate diseases
    lines += ["## Diseases Considered", ""]
    if output.candidate_diseases:
        lines.append(
            "_Note: not every disease listed here is investigated in depth. "
            "Only diseases with a section under **Findings by Disease** below have "
            "literature and clinical-trial evidence pulled for this run._"
        )
        lines.append("")
        for c in output.candidate_diseases:
            lines.append(f"- {_title_case_disease(c)}")
    else:
        lines.append("_No candidates surfaced._")
    lines += ["", "---", ""]

    # Per-disease findings
    lines += ["## Findings by Disease", ""]
    if output.disease_findings:
        for finding in output.disease_findings:
            lines += [
                f"## {_title_case_disease(finding.disease)} _(source: {finding.source})_",
                "",
            ]

            if finding.literature:
                lines += [
                    f"### Literature — {_title_case_disease(finding.disease)}",
                    "",
                    _fmt_literature(finding.literature),
                    "",
                ]

            if finding.clinical_trials:
                rel = finding.blurb.approval_relationship if finding.blurb else ""
                lines += [
                    "### Clinical Trials",
                    "",
                    _fmt_clinical_trials(
                        finding.clinical_trials, finding.disease, rel
                    ),
                    "",
                ]

            lines.append("---")
            lines.append("")
    else:
        lines.append("_No candidate findings produced._")

    return "\n".join(lines)
