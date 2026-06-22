"""Format a SupervisorOutput as a Markdown report."""

import re
from datetime import datetime

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.services.dev_stage import dev_stage_phrase

# Max example trials enumerated per scope (completed / terminated). When the relevant
# (non-contaminated) list exceeds this, the body shows the first N and discloses the
# truncation so the rendered count can't read as the full list.
_TRIAL_RENDER_CAP = 10


def _trial_count_clause(total_on_record: int, n_fetched: int, n_shown: int) -> str:
    """Build a reconciling clause for a completed/terminated trial header.

    The header already states `total_on_record` (the raw API pair total — pre-filter). The OLD
    formatter then appended ", N hidden as a different indication" where N was contamination
    counted WITHIN the fetched (≤50) slice — a DIFFERENT population from total_on_record, so the
    two read as if they subtracted ("64 total, 45 hidden" → looks like 19 visible, but 5 show).

    This clause keeps every number against the SAME population it belongs to and only states what
    is actually true of the fetched slice:
      - all `total_on_record` trials were fetched (n_fetched == total_on_record): say how many of
        them are relevant vs hidden-as-a-different-indication;
      - only a slice was fetched (n_fetched < total_on_record): say "showing N relevant of the
        first M fetched" so the rendered list count is never read against the full total.
    Returns "" when there is nothing to disclose (everything fetched, nothing hidden).
    """
    n_hidden = n_fetched - n_shown
    if n_fetched >= total_on_record:
        # The full population was fetched, so hidden+shown reconcile against total_on_record.
        if n_hidden:
            return (
                f" {n_shown} relevant; {n_hidden} hidden as a different indication"
                f" (of {total_on_record})."
            )
        return ""
    # Only a slice was fetched; never imply the shown count subtracts from total_on_record.
    if n_hidden:
        return (
            f" showing {n_shown} relevant of the first {n_fetched} fetched"
            f" ({n_hidden} of those fetched hidden as a different indication)."
        )
    return f" showing {n_shown} of the first {n_fetched} fetched."


def _title_case_disease(name: str) -> str:
    """Capitalize the first letter of each whitespace-separated word, leaving the
    rest of each word untouched so acronyms (e.g. NSCLC) and possessives (e.g.
    Alzheimer's) are preserved."""
    return " ".join(w[:1].upper() + w[1:] if w else w for w in name.split(" "))


def _fmt_literature(lit: LiteratureOutput) -> str:
    lines: list[str] = []

    if lit.evidence_summary:
        es = lit.evidence_summary
        # class_level = the disease-relevant RCTs are for OTHER drugs in the class, not this
        # one (the combined synthesize call). Make the line honest so the section never reads as
        # direct drug evidence — strength/direction are forced to "none" for class_level by the
        # deterministic cap in synthesize, so this line is the only meaningful rendering.
        if es.evidence_basis == "class_level":
            strength_line = (
                "**Evidence strength:** class-level signal "
                "(no direct evidence for this drug)"
            )
        elif es.evidence_basis == "approved":
            strength_line = (
                "**Evidence strength:** evidence is for an already-approved "
                "sub-indication (not repurposing)"
            )
        else:
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
        # Context (non-efficacy) PMIDs: relevant PK/safety/mechanism studies that are cited as
        # context but carry no efficacy direction — shown so a reader does not see them as dropped.
        if es.neutral_pmids:
            pmid_links = ", ".join(
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                for pmid in es.neutral_pmids
            )
            lines.append(f"\n**Context (non-efficacy) PMIDs:** {pmid_links}")
    else:
        lines.append("_No evidence summary available._")

    return "\n".join(lines)


def _fmt_clinical_trials(
    ct: ClinicalTrialsOutput,
    indication: str = "",
    approval_relationship: str = "none",
) -> str:
    lines: list[str] = []

    # Authoritative development-stage line — the SINGLE source of the phase-tier judgment
    # (from judge_dev_stage). The CT sub-agent prose describes trials but must NOT judge the
    # tier (see clinical_trials.txt), so the tier is stated here once and cannot contradict it.
    stage_line = dev_stage_phrase(ct.signals) if ct.signals else None
    if stage_line:
        lines.append(f"**Development stage:** {stage_line}")

    if ct.summary:
        lines.append(ct.summary)

    if ct.contaminated_nct_ids:
        n = len(ct.contaminated_nct_ids)
        lines.append(
            f"\n_{n} trial(s) excluded as a different indication: "
            f"{', '.join(ct.contaminated_nct_ids)}._"
        )
        # Surface WHY they were excluded (the gate's 1-2 sentence justification — e.g. "the drug is
        # approved for CML, so CML trials are the approved sub-indication, not repurposing of the
        # broader leukemia candidate"). Only shown when something was actually excluded, so the
        # reader understands a thin/"no usable" signal is because approved-indication evidence was
        # set aside, not because none exists.
        if ct.relevance_reasoning.strip():
            lines.append(f"\n_Why: {ct.relevance_reasoning.strip()}_")

    if ct.approval:
        a = ct.approval
        if a.label_found:
            if a.is_approved:
                target = a.matched_indication or "this indication"
                lines.append(f"**FDA approval:** Approved ({target})")
            elif approval_relationship == "contaminated":
                # The verbatim candidate term isn't itself on the label, but the candidate
                # OVERLAPS the drug's approved indications — either an approved sub-indication
                # sits inside this broader candidate (umbrella, e.g. NAFLD contains approved
                # MASH) or the candidate is a sibling that shares a search term with an
                # approval (e.g. systemic hypertension vs approved PAH). A bare "Not found"
                # reads as a self-contradiction against the literature block's "already-
                # approved" verdict, so we surface the overlap without asserting a specific
                # parent/child relationship (the label alone does not tell us which it is).
                lines.append(
                    "**FDA approval:** This exact term is not itself on the FDA label, but "
                    "the candidate overlaps the drug's approved indications"
                )
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

    # Rendered example trials skip contamination — the per-trial relevance gate (including the
    # approval-aware TEST 1) already tagged approved-sub-indication / different-indication trials
    # as contaminated_nct_ids, so the filtered `shown` list below is clean for THIS candidate even
    # when the candidate is approval-relationship "contaminated". (We no longer blanket-suppress the
    # whole table for contaminated candidates — that discarded the trials the gate had already
    # cleanly isolated. The total_count header stays verbatim; only contaminated examples are
    # filtered, and `_trial_count_clause` discloses the gap.)
    contaminated = set(ct.contaminated_nct_ids)

    if ct.completed:
        c = ct.completed
        shown = [t for t in c.trials if t.nct_id not in contaminated]
        lines.append(
            f"\n**Completed trials ({c.total_count} total on record):**"
            f"{_trial_count_clause(c.total_count, len(c.trials), len(shown))}"
        )
        for trial in shown[:_TRIAL_RENDER_CAP]:
            phase = trial.phase or "Unknown phase"
            status = trial.overall_status or ""
            lines.append(
                f"- [{trial.nct_id}](https://clinicaltrials.gov/study/{trial.nct_id}) — {trial.title} ({phase}{', ' + status if status else ''})"
            )
        if len(shown) > _TRIAL_RENDER_CAP:
            lines.append(
                f"- _…and {len(shown) - _TRIAL_RENDER_CAP} more relevant completed "
                f"trial(s) not listed (showing first {_TRIAL_RENDER_CAP} of "
                f"{len(shown)})._"
            )

    # NOTE: a separate "Active / ongoing trials" registry list was removed. It rendered active
    # trials from a status-only filter (_is_active over the search slice), which diverged from
    # the card's authoritative `active_programs` (the LLM judgment, which includes unknown-status
    # pivotal trials and is relevance/pivotal-curated) — e.g. a card could cite an unknown-status
    # Phase 3 the status-filtered list dropped. Two independent definitions of "active" produced
    # card-vs-section mismatches. Active trials are already surfaced authoritatively via the
    # card's active_programs and the CT prose; the redundant second list is gone.

    if ct.terminated:
        term = ct.terminated
        if term.total_count:
            shown = [t for t in term.trials if t.nct_id not in contaminated]
            lines.append(
                f"\n**Terminated trials ({term.total_count} total on record):**"
                f"{_trial_count_clause(term.total_count, len(term.trials), len(shown))}"
            )
            for t in shown[:_TRIAL_RENDER_CAP]:
                reason = f" — *{t.why_stopped}*" if t.why_stopped else ""
                title = f" {t.title}" if t.title else ""
                phase = t.phase or "Unknown phase"
                classified = _classify_stop_reason(t.why_stopped)
                category = f" [{classified}]" if classified != t.why_stopped else ""
                lines.append(
                    f"- [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id}){title} ({phase}){category}{reason}"
                )
            if len(shown) > _TRIAL_RENDER_CAP:
                lines.append(
                    f"- _…and {len(shown) - _TRIAL_RENDER_CAP} more relevant terminated "
                    f"trial(s) not listed (showing first {_TRIAL_RENDER_CAP} of "
                    f"{len(shown)})._"
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
            # Each disease nests UNDER "## Findings by Disease", so it is an H3 and its
            # Literature/Clinical Trials subsections are H4 — keeps the heading hierarchy
            # nested for document navigation/TOC rather than flat at H2/H3.
            lines += [
                f"### {_title_case_disease(finding.disease)} _(source: {finding.source})_",
                "",
            ]

            if finding.literature:
                lines += [
                    f"#### Literature — {_title_case_disease(finding.disease)}",
                    "",
                    _fmt_literature(finding.literature),
                    "",
                ]

            if finding.clinical_trials:
                lines += [
                    "#### Clinical Trials",
                    "",
                    _fmt_clinical_trials(
                        finding.clinical_trials,
                        finding.disease,
                        finding.approval_relationship,
                    ),
                    "",
                ]

            lines.append("---")
            lines.append("")
    else:
        lines.append("_No candidate findings produced._")

    return "\n".join(lines)
