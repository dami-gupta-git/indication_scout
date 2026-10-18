import logging
from datetime import date
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

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
    TrialVerdict,
)
from indication_scout.services.approval_check import (
    extract_approved_from_labels,
    get_approved_indications,
)
from indication_scout.services.disease_helper import resolve_mesh_id
from indication_scout.services.trial_target import judge_trials_treat_disease

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
    target_drug: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list:

    # Snapshot of the NCTs shown to the agent for relevance classification, keyed by normalized indication only (not
    # drug): the agent may query under several drug-name variants of one pair, so this keeps a re-call rewriting the
    # same set while separating distinct indications. get_completed/get_terminated populate it; finalize_analysis
    # enforces a verdict for every shown trial.
    shown_by_indication: dict[str, set[str]] = {}

    # The Trial records behind those ids, so the therapeutic-target gate at finalize can read each objective.
    # Keyed on NCT alone; a later fetch rewrites the same entry.
    records_by_nct: dict[str, Trial] = {}

    # NCTs the therapeutic-target gate already rejected and named back to the agent; bounds that to one round
    # before the gate applies the demotion itself.
    gate_rejected_told: set[str] = set()

    # The drug this agent was launched to analyze, pinned by the caller so a model cannot widen the finalize
    # drug-role check by querying under another drug's name.
    _target_drug = (target_drug or "").strip()

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
        mesh_id, mesh_term = resolved

        async with ClinicalTrialsClient(cache_dir=cache_dir) as client:
            result = await client.search_trials(
                drug,
                mesh_term,
                date_before=date_before,
            )
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
        # Record search-pool trials in the classification set too, so the relevance gate forces a verdict on them;
        # otherwise a contaminated active trial leaks into the "Phase 3 active" signal. The table above is un-capped
        # because a trial never shown gets a blind verdict, which reads "relevant".
        shown_by_indication.setdefault(indication.lower().strip(), set()).update(
            t.nct_id for t in result.trials if t.nct_id
        )
        records_by_nct.update({t.nct_id: t for t in result.trials if t.nct_id})
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
        _mesh_id, mesh_term = resolved

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

        # Record the un-capped classification set (every shown trial needs a verdict at finalize), keyed on
        # normalized indication so drug-name and case/whitespace variants accumulate into one set.
        shown_by_indication.setdefault(indication.lower().strip(), set()).update(
            t.nct_id for t in result.trials if t.nct_id
        )
        records_by_nct.update({t.nct_id: t for t in result.trials if t.nct_id})

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
        _mesh_id, mesh_term = resolved

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

        # Record the un-capped classification set (every shown trial needs a verdict at finalize), keyed on
        # normalized indication so drug-name and case/whitespace variants accumulate into one set.
        shown_by_indication.setdefault(indication.lower().strip(), set()).update(
            t.nct_id for t in result.trials if t.nct_id
        )
        records_by_nct.update({t.nct_id: t for t in result.trials if t.nct_id})

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
        _mesh_id, mesh_term = resolved

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
        - verdicts: one entry PER trial shown to you — across the search, completed, AND
          terminated scopes — each a dict with three keys:
            "nct": the trial's NCT id.
            "drug_role": the part THIS DRUG plays in that trial, read off its title,
              interventions and summary — "studied" (the trial evaluates this drug),
              "comparator" (this drug is the control arm the studied agent is measured
              against), "background" (given to all arms as standard care), or "absent".
            "verdict": "relevant" or "contaminated". "relevant" = studies this drug for THIS
              exact indication (a narrower NON-approved subtype rolls up). "contaminated" =
              the condition is an APPROVED indication of this drug (or a narrower form of one)
              — already-approved evidence, not repurposing; OR a DISTINCT disease (e.g.
              pulmonary vs systemic hypertension); OR another drug's trial.
          You MUST classify EVERY shown trial — omit none, and give each exactly one entry.
        - relevance_reasoning: 1-2 sentences justifying the split.

        Do NOT write a prose summary — the trial-section prose is authored separately after
        the development stage is resolved. This terminates the agent loop.

        Rejected (re-call to fix) when: any shown trial is missing an entry; an entry names an
        NCT that was not shown; a trial has more than one entry; an entry is malformed or uses
        a value outside those listed; an entry is "relevant" while drug_role is not
        "studied"; or a trial marked "relevant" did not set out to treat this indication.
        """
        # Each analyze_clinical_trials call builds a fresh agent, so shown_by_indication holds at most one key.
        # More than one means a tools instance was reused across indications and the relevance gate would leak one
        # pair's contamination into another. The LLM can't fix that by retrying, so fail loudly.
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

        # Parse before counting: an entry with a typo'd verdict or role would otherwise drop its trial from both sets.
        parsed: list[TrialVerdict] = []
        malformed: list[str] = []
        for raw in verdicts:
            try:
                parsed.append(TrialVerdict.model_validate(raw))
            except ValidationError as exc:
                nct = (raw or {}).get("nct") or "(no nct)"
                fields = ", ".join(
                    str(e["loc"][0]) for e in exc.errors() if e.get("loc")
                )
                malformed.append(f"{nct} ({fields or 'unparseable'})")
        if malformed:
            return (
                f"REJECTED: {len(malformed)} entr(y/ies) malformed: "
                f"{', '.join(sorted(malformed))}. Each entry needs nct, drug_role "
                f"(studied|comparator|background|absent) and verdict "
                f"(relevant|contaminated). Re-call with all three, spelled exactly.",
                "",
            )

        duplicates = sorted(
            {v.nct for v in parsed if [x.nct for x in parsed].count(v.nct) > 1}
        )
        if duplicates:
            return (
                f"REJECTED: {len(duplicates)} trial(s) have more than one entry: "
                f"{', '.join(duplicates)}. Give each shown trial exactly one entry "
                f"and re-call.",
                "",
            )

        verdict_ncts = {v.nct for v in parsed}
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

        # A trial is this drug's evidence only when the drug is the agent under study (role asked per trial), so a
        # comparator-arm trial cannot count.
        mismatched = sorted(
            v.nct
            for v in parsed
            if v.verdict == "relevant" and v.drug_role != "studied"
        )
        if mismatched:
            roles = {v.nct: v.drug_role for v in parsed}
            named = _target_drug or "this drug"
            listed = ", ".join(f"{n} ({roles[n]})" for n in mismatched)
            return (
                f"REJECTED: {len(mismatched)} trial(s) marked relevant while {named} is not "
                f"the studied agent: {listed}. Only a trial that STUDIES {named} is its own "
                f"evidence; a comparator, background or absent role is contamination. "
                f"Re-call with corrected verdicts.",
                "",
            )

        relevant_ncts = [v.nct for v in parsed if v.verdict == "relevant"]
        contaminated_ncts = [v.nct for v in parsed if v.verdict == "contaminated"]

        # Therapeutic-target gate: the drug's role is checked above; this checks the trial's objective. Asked once per
        # trial in isolation and cached, then applied deterministically because the agent's own relevance call counted
        # a cardiomyopathy trial as muscular-dystrophy development. A rejected trial moves to contaminated for good.
        gated = [records_by_nct[n] for n in relevant_ncts if n in records_by_nct]
        if relevant_ncts:
            treats = (
                await judge_trials_treat_disease(
                    _target_drug or "this drug",
                    _assigned or next(iter(shown_by_indication), ""),
                    gated,
                    DEFAULT_CACHE_DIR,
                )
                if gated
                else {}
            )
            # A relevant NCT with no stored record can't be gated, so it is demoted (fail-closed).
            demoted = [n for n in relevant_ncts if not treats.get(n, False)]
            if demoted:
                logger.debug(
                    "clinical_trials: therapeutic-target gate demoted %d trial(s) for %s x %s: %s",
                    len(demoted),
                    _target_drug or "this drug",
                    _assigned,
                    ", ".join(demoted),
                )
                unreported = [n for n in demoted if n not in gate_rejected_told]
                gate_rejected_told.update(demoted)
                if unreported:
                    named = _assigned or "this indication"
                    return (
                        f"REJECTED: {len(unreported)} trial(s) marked relevant did not set out "
                        f"to TREAT {named} — the objective is a complication, comorbidity, "
                        f"distinct disease, or a pharmacokinetic/safety characterization: "
                        f"{', '.join(sorted(unreported))}. This verdict is final. Re-call with "
                        f"those trial(s) marked contaminated and relevance_reasoning rewritten "
                        f"to match.",
                        "",
                    )
                relevant_ncts = [n for n in relevant_ncts if n not in demoted]
                contaminated_ncts = contaminated_ncts + demoted
        relevant_text = ", ".join(sorted(relevant_ncts)) or "none"
        contaminated_text = ", ".join(sorted(contaminated_ncts)) or "none"
        relevance_reasoning = (
            f"Relevant trials: {relevant_text}. Excluded trials: {contaminated_text}."
        )
        artifact = FinalizeClinicalTrialsArtifact(
            relevant_ncts=relevant_ncts,
            contaminated_ncts=contaminated_ncts,
            relevance_reasoning=relevance_reasoning,
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
