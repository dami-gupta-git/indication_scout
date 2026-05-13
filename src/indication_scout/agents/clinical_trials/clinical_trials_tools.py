import logging
from datetime import date

from langchain_core.tools import tool

from indication_scout.agents._trial_formatting import (
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.config import get_settings
from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    NEGATION_PREFIXES,
    STOP_KEYWORDS,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
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


def _classify_stop_reason(why_stopped: str | None) -> str:
    """Keyword-based stop classification of a CT.gov why_stopped string.

    Returns one of: safety, efficacy, business, enrollment, unknown — or, when
    no keyword matches, the original why_stopped text verbatim.
    Has a 20-char negation lookback so phrasings like "no safety concerns"
    don't classify as safety.
    """
    if not why_stopped:
        return "unknown"
    lower = why_stopped.lower()
    for keyword, category in STOP_KEYWORDS.items():
        if keyword in lower:
            idx = lower.index(keyword)
            prefix = lower[max(0, idx - 20) : idx]
            if any(neg in prefix for neg in NEGATION_PREFIXES):
                neg_end = max(
                    prefix.rfind(neg) + len(neg)
                    for neg in NEGATION_PREFIXES
                    if neg in prefix
                )
                between = prefix[neg_end:]
                if not any(sep in between for sep in (",", "-", ".", ";")):
                    continue
            return category
    return why_stopped


def _scrub_post_cutoff_outcome(trial: Trial, cutoff: date) -> tuple[Trial, bool]:
    """Strip outcome fields a 2020 holdout couldn't have known in 2020.

    Returns (trial_or_scrubbed_copy, was_scrubbed).

    A trial that started before the cutoff is correctly included in a
    holdout — but if it completed or terminated AFTER the cutoff, its
    `overall_status`, `why_stopped`, and `completion_date` reflect a
    future the holdout shouldn't see. We replace those with UNKNOWN /
    None so the trial appears as "still in progress at the cutoff."

    Trials with no `completion_date` (or with a completion_date before
    the cutoff) are returned unchanged.

    The caller decides what to do with `was_scrubbed=True` trials — e.g.
    `get_completed` and `get_terminated` drop them entirely (a trial
    that wasn't yet completed/terminated at the cutoff doesn't belong
    in those scopes), while `search_trials` keeps them with the
    scrubbed status.
    """
    completion_iso = trial.completion_date
    if not completion_iso:
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
) -> list:

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
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.warning(
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
        header = (
            f"Search for {drug} × {indication}: {result.total_count} trials"
            f"{cap_note}{scrub_note}"
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
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.warning(
                "get_completed: could not resolve MeSH id for indication '%s'; "
                "returning empty CompletedTrialsResult",
                indication,
            )
            return (
                f"Completed for {drug} × {indication}: MeSH unresolved, skipped.",
                CompletedTrialsResult(),
            )
        mesh_id, mesh_term = resolved

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

        shown = len(result.trials)
        cap_note = "; top 50 shown" if shown < result.total_count else ""
        scrub_note = (
            f"; dropped {scrub_dropped} post-cutoff completion(s) "
            f"(not yet completed at cutoff)"
            if scrub_dropped
            else ""
        )
        header = (
            f"Completed for {drug} × {indication}: {result.total_count} total"
            f"{cap_note}{scrub_note}"
        )
        phase_dist = _phase_distribution(result.trials)
        table = _format_trial_table(
            result.trials,
            columns=("nct_id", "phase", "refs", "mesh", "title"),
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
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.warning(
                "get_terminated: could not resolve MeSH id for indication '%s'; "
                "returning empty TerminatedTrialsResult",
                indication,
            )
            return (
                f"Terminated for {drug} × {indication}: MeSH unresolved, skipped.",
                TerminatedTrialsResult(),
            )
        mesh_id, mesh_term = resolved

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
        cap_note = (
            f"; top 50 shown (stop-category counts cover the {shown} shown only)"
            if shown < result.total_count
            else ""
        )
        scrub_note = (
            f"; dropped {scrub_dropped} post-cutoff termination(s) "
            f"(not yet terminated at cutoff)"
            if scrub_dropped
            else ""
        )
        header = (
            f"Terminated for {drug} × {indication}: {result.total_count} total "
            f"({safety_efficacy} safety/efficacy in shown set){cap_note}{scrub_note}"
        )
        phase_dist = _phase_distribution(result.trials)
        table = _format_trial_table(
            result.trials,
            columns=("nct_id", "phase", "stop_reason", "refs", "mesh", "title"),
            cap=_settings.clinical_trials_cap,
            include_why_stopped=True,
            stop_classifier=_classify_stop_reason,
        )
        content = (
            f"{header}\n"
            f"Phase distribution (shown): {phase_dist}\n"
            f"Trials shown (top {_settings.clinical_trials_cap} by enrollment):\n"
            f"{table}"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def get_landscape(indication: str) -> tuple[str, IndicationLandscape]:
        """Get the competitive landscape for an indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by phase then enrollment,
        plus phase distribution and recent starts. Use to understand how crowded the space is.
        """
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
            logger.warning(
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
        # Holdout path: when date_before is set, the live openFDA labels would
        # leak today's approvals (e.g. semaglutide's 2025 MASH approval into a
        # 2020 holdout). Use the hardcoded approvals table instead. Drugs not
        # in the table return is_approved=False with label_found=False so the
        # supervisor knows approval reasoning was disabled, not that the drug
        # is unapproved.
        if date_before is not None:
            approved_set = get_approved_indications(
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
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete.

        Call this as the very last step, passing your 2-3 sentence plain-text summary of the
        findings. This terminates the agent loop.

        Empty or whitespace-only summaries are rejected — re-call with a real summary.
        """
        if not summary or not summary.strip():
            return "REJECTED: empty summary. Re-call with the full summary.", ""
        return "Analysis complete.", summary

    return [
        search_trials,
        get_completed,
        get_terminated,
        get_landscape,
        check_fda_approval,
        finalize_analysis,
    ]
