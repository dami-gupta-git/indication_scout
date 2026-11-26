import logging
from datetime import date
from pathlib import Path

from langchain_core.tools import tool

from indication_scout.agents._trial_formatting import (
    _classify_stop_reason,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents._trial_signals import is_non_therapeutic_study
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    FinalizeClinicalTrialsArtifact,
)
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.services.approval_check import (
    ApprovalLabel,
    get_approved_indications,
)
from indication_scout.services.disease_helper import resolve_mesh_id

_settings = get_settings()

logger = logging.getLogger(__name__)


def _scrub_post_cutoff_outcome(trial: Trial, cutoff: date) -> tuple[Trial, bool]:
    """Strip outcome fields a holdout could not have known (status, why_stopped, completion_date).

    Returns (trial_or_scrubbed_copy, was_scrubbed). A trial completed/terminated AFTER the cutoff has those fields
    replaced with UNKNOWN / None so it appears in progress at the cutoff. A terminal-status trial with no
    completion_date is scrubbed conservatively; non-terminal trials are unchanged.

    The caller handles `was_scrubbed=True`: `get_completed` and `get_terminated` drop such trials, `search_trials`
    keeps them with the scrubbed status.
    """
    _TERMINAL_STATUSES = {"COMPLETED", "TERMINATED"}
    completion_iso = trial.completion_date
    if not completion_iso:
        # No completion date: scrub a terminal status (the outcome may postdate the cutoff); keep non-terminal trials.
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
    # CT.gov dates are "YYYY-MM-DD" or "YYYY-MM"; both compare lexicographically with an ISO cutoff prefix.
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
    cache_dir: Path = DEFAULT_CACHE_DIR,
    approval_label: ApprovalLabel | None = None,
) -> list:
    """Build the trials tools for one drug × indication pair.

    `approval_label` is the candidate step's FDA verdict for the pair (get_fda_approved_disease_mapping). check_fda_approval
    reports it as-is in a live run and raises if it is missing; holdout runs (`date_before` set) ignore it.
    """

    # The indication this agent was launched to investigate. Calls for any OTHER indication are rejected so a
    # drifting LLM (e.g. "smoking cessation" vs assigned "nicotine dependence") is nudged back instead of crashing
    # at finalize. None disables the check.
    _assigned = (assigned_indication or "").lower().strip()

    def _drop_non_therapeutic(
        trials: list, scope: str, drug: str, indication: str
    ) -> list:
        """Remove trials whose every typed intervention is non-therapeutic, logging what went.

        Applied at fetch, before the classification set is recorded. The check is disease-independent, so doing it
        once here keeps the verdict identical across candidates.
        """
        kept = [t for t in trials if not is_non_therapeutic_study(t)]
        dropped = [t.nct_id for t in trials if is_non_therapeutic_study(t)]
        if dropped:
            logger.debug(
                "%s: dropped %d non-therapeutic trial(s) for %s x %s (all interventions are "
                "diagnostic/device/tracer/procedure): %s",
                scope,
                len(dropped),
                drug,
                indication,
                ", ".join(dropped),
            )
        return kept

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
    ) -> tuple[str, SearchTrialsResult | None]:
        """All-status trials for a drug × indication pair.

        Returns total count for the pair, per-status counts (recruiting,
        active, withdrawn), the top 50 trials by enrollment, and every ongoing trial. The TERMINATED
        and COMPLETED counts live on get_terminated and get_completed
        respectively — call those for those scopes.

        Whitespace verdict: total_count == 0 means no trials of this drug in
        this indication.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, None
        resolved = await resolve_mesh_id(indication)
        if resolved is None:
            logger.debug(
                "search_trials: could not resolve MeSH id for indication '%s'; "
                "returning empty SearchTrialsResult",
                indication,
            )
            return (
                f"Search for {drug} × {indication}: MeSH unresolved, skipped.",
                SearchTrialsResult(resolution_status="unresolved"),
            )
        mesh_id, mesh_term, route = resolved

        async with ClinicalTrialsClient(cache_dir=cache_dir) as client:
            result = await client.search_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )
        result.resolution_route = route
        result.resolution_term = mesh_term
        result.trials = _drop_non_therapeutic(
            result.trials, "search_trials", drug, indication
        )

        # Holdout scrubber: trials that completed/terminated AFTER the cutoff stay in the results with UNKNOWN status.
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
        cap_note = (
            "; enrollment top 50 plus every ongoing trial shown"
            if shown < result.total_count
            else ""
        )
        scrub_note = (
            f"; scrubbed post-cutoff outcomes from {scrubbed_n} trial(s) "
            f"(status set to UNKNOWN)"
            if scrubbed_n
            else ""
        )
        # Per-status breakdown from the artifact's by_status, so header and artifact stay consistent.
        bs = result.by_status
        status_breakdown = (
            f" (recruiting={bs.get('RECRUITING', 0)}, "
            f"active={bs.get('ACTIVE_NOT_RECRUITING', 0)}, "
            f"withdrawn={bs.get('WITHDRAWN', 0)}, "
            f"unknown={bs.get('UNKNOWN', 0)})"
        )
        header = (
            f"Search for {drug} × {indication}: {result.total_count} registry query matches"
            f"{status_breakdown}{cap_note}{scrub_note}\n"
            f"Resolved query MeSH: {mesh_term} ({mesh_id}) — compare each trial's mesh "
            f"column against this descriptor to judge relevance vs contamination."
        )
        phase_dist = _phase_distribution(result.trials)
        # `interventions` + `arms` + `brief_summary` feed the relevance gate's TEST 0 (no extra fetch).
        table = _format_trial_table(
            result.trials,
            columns=(
                "nct_id",
                "phase",
                "status",
                "interventions",
                "arms",
                "mesh",
                "brief_summary",
                "title",
            ),
            cap=len(result.trials),
        )
        content = (
            f"{header}\n"
            f"Phase distribution (shown): {phase_dist}\n"
            f"Trials shown (all {shown} — classify EVERY one):\n"
            f"{table}"
        )
        return content, result

    @tool(response_format="content_and_artifact")
    async def get_completed(
        drug: str, indication: str
    ) -> tuple[str, CompletedTrialsResult | None]:
        """COMPLETED trials for a drug × indication pair.

        Returns total completed, Phase 3 count, and the top 50 completed
        trials by enrollment. A completed Phase 3 trial that did not lead
        to subsequent regulatory progression is a strong signal that the
        primary endpoint was not met.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, None
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
        _mesh_id, mesh_term, _route = resolved

        async with ClinicalTrialsClient(cache_dir=cache_dir) as client:
            result = await client.get_completed_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )
        result.trials = _drop_non_therapeutic(
            result.trials, "get_completed", drug, indication
        )

        # Holdout drop: a trial that completed AFTER the cutoff doesn't belong in this scope (search_trials still
        # shows it as in progress).
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


        scrub_note = (
            f"; dropped {scrub_dropped} post-cutoff completion(s) "
            f"(not yet completed at cutoff)"
            if scrub_dropped
            else ""
        )
        header = (
            f"Completed for {drug} × {indication}: {result.total_count} registry query matches"
            f"{scrub_note}\n"
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
                "arms",
                "title",
                "brief_summary",
            ),
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
    ) -> tuple[str, TerminatedTrialsResult | None]:
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
            return mismatch, None
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
        _mesh_id, mesh_term, _route = resolved

        async with ClinicalTrialsClient(cache_dir=cache_dir) as client:
            result = await client.get_terminated_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )
        result.trials = _drop_non_therapeutic(
            result.trials, "get_terminated", drug, indication
        )

        # Holdout drop: a trial that terminated AFTER the cutoff doesn't belong in this scope (search_trials still
        # shows it with UNKNOWN status).
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


        header = (
            f"Terminated for {drug} × {indication}: {result.total_count} registry query matches "
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
                "arms",
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
    async def get_landscape(indication: str) -> tuple[str, IndicationLandscape | None]:
        """Get the competitive landscape for an indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by phase then enrollment,
        plus phase distribution and recent starts. Use to understand how crowded the space is.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, None
        # Holdout skip: the landscape aggregates per-trial status and phase across competitors, which would leak
        # post-cutoff outcomes (a competitor reaching Phase 3 in 2024 showing as Phase 3 in a 2020 holdout).
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
        _mesh_id, mesh_term, _route = resolved

        async with ClinicalTrialsClient(cache_dir=cache_dir) as client:
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
    ) -> tuple[str, ApprovalCheck | None]:
        """Check whether the drug is FDA-approved for this indication.

        Reports the FDA-label verdict for this drug and indication, judged across all current
        FDA labels for every known name of the drug. Use this whenever the completed scope
        contains any trial — it is the only tool that can tell you whether a completed trial
        led to approval.

        When is_approved is True, the drug IS approved for this indication — this is NOT a
        repurposing opportunity. When False, the indication was not found on FDA labels (which
        does not distinguish trial failure from approval pending from approval outside the US).
        When inconclusive is True, the label could not be checked; draw no approval conclusion.
        """
        mismatch = _indication_mismatch(indication)
        if mismatch is not None:
            return mismatch, None
        # Holdout path: live openFDA labels would leak today's approvals (e.g. semaglutide's 2025 MASH approval into
        # a 2020 holdout), so use the hardcoded approvals table. Drugs not in it return is_approved=False with
        # label_found=False: approval reasoning was disabled, not the drug unapproved.
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

        # Live path: report the verdict the candidate step already reached for this pair (get_fda_approved_disease_mapping),
        # so the trials card and the candidate drop cannot disagree. No table lookup, model call or label fetch here: the
        # mapping already consults the hardcoded approvals table, and a second lookup keyed on the agent's drug name could
        # override an unresolved verdict. A missing label means the caller did not wire the pair's verdict through, which
        # is a bug, not a "not approved".
        if approval_label is None:
            raise DataSourceError(
                "check_fda_approval",
                f"no approval label passed for {drug} × {indication} in a live run",
            )
        if approval_label == "unresolved":
            return (
                f"FDA approval check for {drug} × {indication}: could not be checked "
                f"(label text unavailable or unreadable)",
                ApprovalCheck(
                    is_approved=False,
                    label_found=False,
                    inconclusive=True,
                    drug_names_checked=[drug],
                ),
            )
        is_approved = approval_label == "approved"
        return (
            f"FDA approval check for {drug} × {indication}: "
            f"{'APPROVED' if is_approved else 'not on FDA label'}",
            ApprovalCheck(
                is_approved=is_approved,
                label_found=True,
                matched_indication=indication if is_approved else None,
                drug_names_checked=[drug],
            ),
        )

    @tool(response_format="content_and_artifact")
    async def finalize_analysis() -> tuple[str, FinalizeClinicalTrialsArtifact]:
        """Signal that the analysis is complete. Call this as the very last step, after all the fetch tools.

        Takes no arguments. Each shown trial is classified afterwards by isolated per-trial checks, and the
        trial-section prose is authored separately after the development stage is resolved. This terminates the
        agent loop.
        """
        return "Analysis complete.", FinalizeClinicalTrialsArtifact()

    return [
        search_trials,
        get_completed,
        get_terminated,
        get_landscape,
        check_fda_approval,
        finalize_analysis,
    ]
