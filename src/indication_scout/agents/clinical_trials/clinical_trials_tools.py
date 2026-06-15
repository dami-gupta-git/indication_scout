import logging
from datetime import date

from langchain_core.tools import tool

from indication_scout.agents._trial_formatting import (
    _classify_stop_reason,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    FinalizeClinicalTrialsArtifact,
)
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.fda import FDAClient
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.services.approval_check import (
    extract_approved_from_labels,
    get_approved_indications,
)
from indication_scout.services.disease_helper import resolve_mesh_id

_settings = get_settings()

logger = logging.getLogger(__name__)


def _scrub_post_cutoff_outcome(trial: Trial, cutoff: date) -> tuple[Trial, bool]:
    """Strip outcome fields a 2020 holdout couldn't have known in 2020.

    Returns (trial_or_scrubbed_copy, was_scrubbed).

    A trial that started before the cutoff is correctly included in a
    holdout — but if it completed or terminated AFTER the cutoff, its
    `overall_status`, `why_stopped`, and `completion_date` reflect a
    future the holdout shouldn't see. We replace those with UNKNOWN /
    None so the trial appears as "still in progress at the cutoff."

    Trials with a completion_date before the cutoff are returned
    unchanged. A trial with a terminal status (COMPLETED/TERMINATED)
    but NO completion_date is scrubbed conservatively — we can't prove
    it completed before the cutoff, so we don't surface its outcome.
    Non-terminal trials with no completion_date are returned unchanged.

    The caller decides what to do with `was_scrubbed=True` trials — e.g.
    `get_completed` and `get_terminated` drop them entirely (a trial
    that wasn't yet completed/terminated at the cutoff doesn't belong
    in those scopes), while `search_trials` keeps them with the
    scrubbed status.
    """
    _TERMINAL_STATUSES = {"COMPLETED", "TERMINATED"}
    completion_iso = trial.completion_date
    if not completion_iso:
        # No completion date. If the trial reports a terminal status, we
        # can't prove that outcome predates the cutoff — scrub it. A
        # non-terminal trial has no future outcome to hide, so keep it.
        if trial.overall_status in _TERMINAL_STATUSES:
            scrubbed = trial.model_copy(
                update={
                    "overall_status": "UNKNOWN",
                    "why_stopped": None,
                    "completion_date": None,
                }
            )
            return scrubbed, True
        return trial, False
    cutoff_iso = cutoff.isoformat()
    # CT.gov dates are "YYYY-MM-DD" or "YYYY-MM"; both compare
    # lexicographically against an ISO cutoff prefix.
    if completion_iso < cutoff_iso:
        return trial, False
    scrubbed = trial.model_copy(
        update={
            "overall_status": "UNKNOWN",
            "why_stopped": None,
            "completion_date": None,
        }
    )
    return scrubbed, True


def build_clinical_trials_tools(
    date_before: date | None = None,
    assigned_indication: str | None = None,
) -> list:

    # Closure-scoped snapshot of the NCTs rendered to the agent for the relevance
    # classification, keyed by (drug, indication). get_completed/get_terminated
    # populate it; finalize_analysis reads it to enforce that every shown trial got
    # a verdict. Keyed (not flat) so a re-call of the same pair rewrites the same
    # deterministic set and a different indication can't contaminate the check.
    # Keyed on INDICATION only, not (drug, indication): the agent investigates one indication
    # per instance, but may query the trial tools under several drug-name VARIANTS of that pair
    # (e.g. "bupropion" and the combo alias "naltrexone bupropion"). Keying on the stable
    # indication keeps a re-call rewriting the same set, tolerates drug-name variants, and still
    # separates distinct indications (the cross-pair-reuse guard).
    shown_by_indication: dict[str, set[str]] = {}

    # The indication this agent instance was launched to investigate. The trial tools
    # reject a call for any OTHER indication so a drifting LLM (e.g. querying "smoking
    # cessation" while assigned "nicotine dependence" — a distinct MeSH descriptor) is
    # nudged back instead of accumulating a second key and crashing at finalize. None
    # disables the check (callers that don't pin an indication).
    _assigned = (assigned_indication or "").lower().strip()

    def _indication_mismatch(indication: str) -> str | None:
        """REJECTED message when `indication` differs from the assigned one, else None."""
        if not _assigned:
            return None
        if (indication or "").lower().strip() == _assigned:
            return None
        return (
            f"REJECTED: this agent investigates only '{assigned_indication}'. "
            f"You called the tool with '{indication}'. Re-call with "
            f"'{assigned_indication}' as the indication; do not query other "
            "indications."
        )

    @tool(response_format="content_and_artifact")
    async def search_trials(
        drug: str, indication: str
    ) -> tuple[str, SearchTrialsResult]:
        """All-status trials for a drug × indication pair.

        Returns total count for the pair, per-status counts (recruiting,
        active, withdrawn), and the top 50 trials by enrollment. The TERMINATED
        and COMPLETED counts live on get_terminated and get_completed
        respectively — call those for those scopes.

        Whitespace verdict: total_count == 0 means no trials of this drug in
        this indication.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, SearchTrialsResult()
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.debug(
                "search_trials: could not resolve MeSH id for indication '%s'; "
                "returning empty SearchTrialsResult",
                indication,
            )
            return (
                f"Search for {drug} × {indication}: MeSH unresolved, skipped.",
                SearchTrialsResult(),
            )
        mesh_id, mesh_term = resolved

        async with ClinicalTrialsClient() as client:
            result = await client.search_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )

        # Holdout scrubber: when date_before is set, strip outcome fields
        # for trials that completed/terminated AFTER the cutoff so the
        # supervisor doesn't see future trial outcomes. search_trials
        # keeps these trials (they were in progress at the cutoff) but
        # rewrites their status to UNKNOWN.
        scrubbed_n = 0
        if date_before is not None:
            new_trials = []
            for t in result.trials:
                t2, was_scrubbed = _scrub_post_cutoff_outcome(t, date_before)
                if was_scrubbed:
                    scrubbed_n += 1
                new_trials.append(t2)
            result.trials = new_trials

        shown = len(result.trials)
        cap_note = "; top 50 shown" if shown < result.total_count else ""
        scrub_note = (
            f"; scrubbed post-cutoff outcomes from {scrubbed_n} trial(s) "
            f"(status set to UNKNOWN)"
            if scrubbed_n
            else ""
        )
        # Per-status breakdown from the artifact's by_status (see
        # agent_data_contracts.md). Rendered from the SearchTrialsResult field
        # so header and artifact stay consistent.
        bs = result.by_status
        status_breakdown = (
            f" (recruiting={bs.get('RECRUITING', 0)}, "
            f"active={bs.get('ACTIVE_NOT_RECRUITING', 0)}, "
            f"withdrawn={bs.get('WITHDRAWN', 0)}, "
            f"unknown={bs.get('UNKNOWN', 0)})"
        )
        header = (
            f"Search for {drug} × {indication}: {result.total_count} trials"
            f"{status_breakdown}{cap_note}{scrub_note}\n"
            f"Resolved query MeSH: {mesh_term} ({mesh_id}) — compare each trial's mesh "
            f"column against this descriptor to judge relevance vs contamination."
        )
        phase_dist = _phase_distribution(result.trials)
        table = _format_trial_table(
            result.trials,
            columns=("nct_id", "phase", "status", "mesh", "title"),
            cap=_settings.clinical_trials_cap,
        )
        content = (
            f"{header}\n"
            f"Phase distribution (shown): {phase_dist}\n"
            f"Trials shown (top {_settings.clinical_trials_cap} by enrollment):\n"
            f"{table}"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def get_completed(
        drug: str, indication: str
    ) -> tuple[str, CompletedTrialsResult]:
        """COMPLETED trials for a drug × indication pair.

        Returns total completed, Phase 3 count, and the top 50 completed
        trials by enrollment. A completed Phase 3 trial that did not lead
        to subsequent regulatory progression is a strong signal that the
        primary endpoint was not met.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, CompletedTrialsResult()
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.debug(
                "get_completed: could not resolve MeSH id for indication '%s'; "
                "returning empty CompletedTrialsResult",
                indication,
            )
            return (
                f"Completed for {drug} × {indication}: MeSH unresolved, skipped.",
                CompletedTrialsResult(),
            )
        _mesh_id, mesh_term = resolved

        async with ClinicalTrialsClient() as client:
            result = await client.get_completed_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )

        # Holdout drop: a trial that completed AFTER the cutoff was not
        # completed at the cutoff, so it does not belong in this scope.
        # Drop it (search_trials will still surface it as in-progress).
        scrub_dropped = 0
        if date_before is not None:
            kept = []
            for t in result.trials:
                _, was_scrubbed = _scrub_post_cutoff_outcome(t, date_before)
                if was_scrubbed:
                    scrub_dropped += 1
                else:
                    kept.append(t)
            result.trials = kept
            result.total_count = max(0, result.total_count - scrub_dropped)

        # Record the classification set for this indication (un-capped — every shown
        # trial must get a verdict at finalize). Keyed on normalized indication so drug-name
        # variants AND case/whitespace variants accumulate into one set rather than splitting.
        shown_by_indication.setdefault(indication.lower().strip(), set()).update(
            t.nct_id for t in result.trials if t.nct_id
        )

        scrub_note = (
            f"; dropped {scrub_dropped} post-cutoff completion(s) "
            f"(not yet completed at cutoff)"
            if scrub_dropped
            else ""
        )
        header = (
            f"Completed for {drug} × {indication}: {result.total_count} total"
            f"{scrub_note}\n"
            f"Judge relevance from the drugs (is {drug} the studied drug?), title, and "
            f"summary — is the disease THIS indication, not a distinct subtype?"
        )
        phase_dist = _phase_distribution(result.trials)
        table = _format_trial_table(
            result.trials,
            columns=("nct_id", "phase", "interventions", "title", "brief_summary"),
            cap=len(result.trials),
        )
        content = (
            f"{header}\n"
            f"Phase distribution (shown): {phase_dist}\n"
            f"Trials shown (all {len(result.trials)} — classify EVERY one):\n"
            f"{table}"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def get_terminated(
        drug: str, indication: str
    ) -> tuple[str, TerminatedTrialsResult]:
        """TERMINATED trials for a drug × indication pair.

        Returns total terminated and the top 50 terminated trials by enrollment.
        Each Trial carries `why_stopped` text. A safety/efficacy stop on
        this exact pair is direct evidence the hypothesis was tested and
        stopped early; business/enrollment stops are sponsor decisions and
        neutral on drug performance.

        The stop-category counts in the content string are computed from
        the trials shown, not the full population, and may undercount when
        more than 50 terminations exist for the pair.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, TerminatedTrialsResult()
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.debug(
                "get_terminated: could not resolve MeSH id for indication '%s'; "
                "returning empty TerminatedTrialsResult",
                indication,
            )
            return (
                f"Terminated for {drug} × {indication}: MeSH unresolved, skipped.",
                TerminatedTrialsResult(),
            )
        _mesh_id, mesh_term = resolved

        async with ClinicalTrialsClient() as client:
            result = await client.get_terminated_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )

        # Holdout drop: a trial that terminated AFTER the cutoff was not
        # terminated at the cutoff. Drop from this scope (search_trials
        # still surfaces it with UNKNOWN status).
        scrub_dropped = 0
        if date_before is not None:
            kept = []
            for t in result.trials:
                _, was_scrubbed = _scrub_post_cutoff_outcome(t, date_before)
                if was_scrubbed:
                    scrub_dropped += 1
                else:
                    kept.append(t)
            result.trials = kept
            result.total_count = max(0, result.total_count - scrub_dropped)

        shown = len(result.trials)
        safety_efficacy = sum(
            1
            for t in result.trials
            if _classify_stop_reason(t.why_stopped) in {"safety", "efficacy"}
        )
        scrub_note = (
            f"; dropped {scrub_dropped} post-cutoff termination(s) "
            f"(not yet terminated at cutoff)"
            if scrub_dropped
            else ""
        )

        # Record the classification set for this indication (un-capped — every shown
        # trial must get a verdict at finalize). Keyed on normalized indication so drug-name
        # variants AND case/whitespace variants accumulate into one set rather than splitting.
        shown_by_indication.setdefault(indication.lower().strip(), set()).update(
            t.nct_id for t in result.trials if t.nct_id
        )

        header = (
            f"Terminated for {drug} × {indication}: {result.total_count} total "
            f"({safety_efficacy} safety/efficacy in shown set){scrub_note}\n"
            f"Judge relevance from the drugs (is {drug} the studied drug?), title, and "
            f"summary — is the disease THIS indication, not a distinct subtype?"
        )
        phase_dist = _phase_distribution(result.trials)
        table = _format_trial_table(
            result.trials,
            columns=(
                "nct_id",
                "phase",
                "interventions",
                "stop_reason",
                "title",
                "brief_summary",
            ),
            cap=len(result.trials),
            include_why_stopped=True,
            stop_classifier=_classify_stop_reason,
        )
        content = (
            f"{header}\n"
            f"Phase distribution (shown): {phase_dist}\n"
            f"Trials shown (all {shown} — classify EVERY one):\n"
            f"{table}"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def get_landscape(indication: str) -> tuple[str, IndicationLandscape]:
        """Get the competitive landscape for an indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by phase then enrollment,
        plus phase distribution and recent starts. Use to understand how crowded the space is.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, IndicationLandscape()
        # Holdout skip: the landscape aggregates per-trial overall_status and
        # phase across all competitors for the indication. Those aggregates
        # would leak post-cutoff trial outcomes (e.g. a competitor that
        # advanced to Phase 3 in 2024 would show as Phase 3 in a 2020
        # holdout). Rather than scrub per-trial inside the aggregator, just
        # disable landscape entirely under date_before — the supervisor's
        # core reasoning runs off search/completed/terminated.
        if date_before is not None:
            return (
                f"Landscape for {indication}: skipped under date_before "
                f"holdout ({date_before.isoformat()}) — landscape aggregates "
                f"trial state across all competitors and cannot be reconstructed "
                f"as-of-cutoff without per-competitor scrubbing.",
                IndicationLandscape(),
            )

        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.debug(
                "get_landscape: could not resolve MeSH id for indication '%s'; "
                "returning empty IndicationLandscape",
                indication,
            )
            return (
                f"Landscape for {indication}: MeSH unresolved, skipped.",
                IndicationLandscape(),
            )
        _mesh_id, mesh_term = resolved

        async with ClinicalTrialsClient() as client:
            landscape = await client.get_landscape(
                mesh_term,
                date_before=date_before,
                top_n=10,
            )
        return (
            f"Landscape for {indication}: {len(landscape.competitors)} competitors",
            landscape,
        )

    @tool(response_format="content_and_artifact")
    async def check_fda_approval(
        drug: str, indication: str
    ) -> tuple[str, ApprovalCheck]:
        """Check whether the drug is FDA-approved for this indication.

        Resolves all known trade/generic names for the drug via ChEMBL, then checks current FDA
        labels for any label whose approved indications cover the given indication. Use this
        whenever the completed scope contains any trial — it is the only tool that can tell you
        whether a completed trial led to approval.

        When is_approved is True, the drug IS approved for this indication — this is NOT a
        repurposing opportunity. When False, the indication was not found on FDA labels (which
        does not distinguish trial failure from approval pending from approval outside the US).
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, ApprovalCheck()
        # Holdout path: when date_before is set, the live openFDA labels would
        # leak today's approvals (e.g. semaglutide's 2025 MASH approval into a
        # 2020 holdout). Use the hardcoded approvals table instead. Drugs not
        # in the table return is_approved=False with label_found=False so the
        # supervisor knows approval reasoning was disabled, not that the drug
        # is unapproved.
        if date_before is not None:
            approved_set = await get_approved_indications(
                drug_name=drug,
                candidate_diseases=[indication],
                as_of=date_before,
            )
            is_approved = indication in approved_set
            result = ApprovalCheck(
                is_approved=is_approved,
                label_found=True,
                matched_indication=indication if is_approved else None,
                drug_names_checked=[drug],
            )
            content = (
                f"FDA approval check for {drug} × {indication} "
                f"(as of {date_before.isoformat()}): "
                f"{'APPROVED' if is_approved else 'not approved per hardcoded table'}"
            )
            return content, result

        try:
            chembl_id = await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
        except DataSourceError:
            logger.warning(
                "check_fda_approval: could not resolve '%s' to ChEMBL id", drug
            )
            return (
                f"FDA approval check for {drug} × {indication}: drug not resolved.",
                ApprovalCheck(),
            )

        drug_names = await get_all_drug_names(chembl_id, DEFAULT_CACHE_DIR)
        if not drug_names:
            logger.warning(
                "check_fda_approval: no drug names for ChEMBL id '%s'", chembl_id
            )
            return (
                f"FDA approval check for {drug} × {indication}: no drug names.",
                ApprovalCheck(),
            )

        async with FDAClient(cache_dir=DEFAULT_CACHE_DIR) as client:
            label_texts = await client.get_all_label_indications(drug_names)

        if not label_texts:
            logger.warning(
                "check_fda_approval: no FDA labels found for any of %s", drug_names
            )
            return (
                f"FDA approval check for {drug} × {indication}: "
                f"no FDA label found (checked {len(drug_names)} drug names)",
                ApprovalCheck(
                    is_approved=False,
                    label_found=False,
                    drug_names_checked=drug_names,
                ),
            )

        approved = await extract_approved_from_labels(
            label_texts, [indication], DEFAULT_CACHE_DIR
        )
        approved_lower = {d.lower().strip() for d in approved}
        is_approved = indication.lower().strip() in approved_lower
        matched = indication if is_approved else None

        result = ApprovalCheck(
            is_approved=is_approved,
            label_found=True,
            matched_indication=matched,
            drug_names_checked=drug_names,
        )
        content = (
            f"FDA approval check for {drug} × {indication}: "
            f"{'APPROVED' if is_approved else 'not on FDA label'} "
            f"(checked {len(drug_names)} drug names)"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def finalize_analysis(
        verdicts: list[dict],
        relevance_reasoning: str,
    ) -> tuple[str, FinalizeClinicalTrialsArtifact | str]:
        """Signal that the analysis is complete.

        Call this as the very last step. Pass:
        - verdicts: one entry PER completed/terminated trial shown to you, each a dict
          {"nct": "<NCT id>", "verdict": "relevant" | "contaminated"}. You MUST classify
          EVERY shown trial — omit none. "relevant" = studies this drug for THIS exact
          indication (a narrower subtype rolls up). "contaminated" = a DISTINCT disease
          (e.g. pulmonary vs systemic hypertension) or a different drug's trial.
        - relevance_reasoning: 1-2 sentences justifying the split.

        Do NOT write a prose summary — the trial-section prose is authored separately after
        the development stage is resolved. This terminates the agent loop.

        Rejected (re-call to fix) when: any shown trial is missing a verdict; or a verdict
        names an NCT that was not shown.
        """
        # The supervisor builds a fresh agent (fresh, empty shown_by_indication) per
        # analyze_clinical_trials call, so this instance investigates ONE indication —
        # shown_by_indication holds AT MOST one key (drug-name variants of the same pair
        # accumulate into it). More than one key means a tools instance was reused across
        # indications: the relevance gate would tag one pair's trials contaminated under
        # another (the cross-pair leak). That is a wiring error the LLM can't fix by
        # retrying, so fail loudly rather than silently union an accumulated set into a verdict.
        if len(shown_by_indication) > 1:
            raise DataSourceError(
                "clinical_trials",
                "finalize_analysis: shown_by_indication holds "
                f"{len(shown_by_indication)} indications {sorted(shown_by_indication)} — "
                "the tools instance was reused across indications. Build a fresh "
                "clinical-trials agent per analyze_clinical_trials call.",
            )
        # Empty when no trials were shown (e.g. MeSH unresolved) → empty verdicts is valid.
        shown: set[str] = next(iter(shown_by_indication.values()), set())

        verdict_ncts = {v.get("nct") for v in verdicts if v.get("nct")}
        missing = shown - verdict_ncts
        unknown = verdict_ncts - shown
        if missing:
            return (
                f"REJECTED: missing verdicts for {len(missing)} shown trial(s): "
                f"{', '.join(sorted(missing))}. Classify EVERY shown trial "
                f"(relevant or contaminated) and re-call.",
                "",
            )
        if unknown:
            return (
                f"REJECTED: verdicts name {len(unknown)} trial(s) that were not shown: "
                f"{', '.join(sorted(unknown))}. Only classify shown trials and re-call.",
                "",
            )

        relevant_ncts = [
            v["nct"]
            for v in verdicts
            if v.get("verdict") == "relevant" and v.get("nct")
        ]
        contaminated_ncts = [
            v["nct"]
            for v in verdicts
            if v.get("verdict") == "contaminated" and v.get("nct")
        ]
        artifact = FinalizeClinicalTrialsArtifact(
            relevant_ncts=relevant_ncts,
            contaminated_ncts=contaminated_ncts,
            relevance_reasoning=relevance_reasoning or "",
        )
        return "Analysis complete.", artifact

    return [
        search_trials,
        get_completed,
        get_terminated,
        get_landscape,
        check_fda_approval,
        finalize_analysis,
    ]
