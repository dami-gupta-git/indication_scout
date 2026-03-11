"""
ClinicalTrials.gov REST API v2 client.

Five methods:
  1. get_trial         — Fetch a single trial by NCT ID
  2. search_trials     — Trial Agent: drug + condition → trial records
  3. detect_whitespace — Trial Agent: is this drug-condition pair unexplored?
  4. get_landscape     — Landscape Agent: competitive map for a condition
  5. get_terminated    — Critique Agent: what has failed in this space?
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from indication_scout.constants import (
    CLINICAL_TRIALS_BASE_URL,
    CLINICAL_TRIALS_RECENT_START_YEAR,
    CLINICAL_TRIALS_WHITESPACE_CONDITION_MAX,
    CLINICAL_TRIALS_WHITESPACE_EXACT_MAX,
    CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER,
    CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS,
    STOP_KEYWORDS,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError

from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    ConditionDrug,
    ConditionLandscape,
    Intervention,
    PrimaryOutcome,
    RecentStart,
    TerminatedTrial,
    Trial,
    WhitespaceResult,
)


def _classify_stop_reason(why_stopped: str | None) -> str:
    """Keyword-based stop classification. LLM refinement happens at the agent layer."""
    if not why_stopped:
        return "unknown"
    lower = why_stopped.lower()
    for keyword, category in STOP_KEYWORDS.items():
        if keyword in lower:
            return category
    return "other"


class ClinicalTrialsClient(BaseClient):
    BASE_URL = CLINICAL_TRIALS_BASE_URL
    PAGE_SIZE = 100

    @property
    def _source_name(self) -> str:
        return "clinical_trials"

    # ------------------------------------------------------------------
    # Public: get_trial
    # ------------------------------------------------------------------

    async def get_trial(self, nct_id: str) -> Trial:
        """Fetch a single trial by NCT ID.

        Args:
            nct_id: The NCT identifier (e.g., "NCT04971785")

        Returns:
            Trial object with all available data

        Raises:
            DataSourceError: If the trial is not found
        """
        url = f"{self.BASE_URL}/{nct_id}"
        params = {"format": "json"}
        data = await self._rest_get(url, params)

        if not data:
            raise DataSourceError(self._source_name, f"No trial found for '{nct_id}'")

        return self._parse_trial(data)

    # ------------------------------------------------------------------
    # Public: search_trials
    # ------------------------------------------------------------------

    async def search_trials(
        self,
        drug: str,
        condition: str | None = None,
        date_before: date | None = None,
        phase_filter: str | None = None,
        max_results: int = 200,
    ) -> list[Trial]:
        """Search for trials matching drug and optional condition."""
        trials: list[Trial] = []
        page_token: str | None = None

        while len(trials) < max_results:
            params = self._build_search_params(
                drug=drug,
                condition=condition,
                date_before=date_before,
                phase_filter=phase_filter,
                page_token=page_token,
            )
            data = await self._rest_get(self.BASE_URL, params)
            studies = data.get("studies", [])
            trials.extend(self._parse_trial(s) for s in studies)

            page_token = data.get("nextPageToken")
            if not page_token or len(studies) < self.PAGE_SIZE:
                break

        return trials[:max_results]

    # ------------------------------------------------------------------
    # Public: detect_whitespace
    # ------------------------------------------------------------------

    async def detect_whitespace(
        self,
        drug: str,
        condition: str,
        date_before: date | None = None,
    ) -> WhitespaceResult:
        """Is this drug-condition pair being explored in clinical trials?

        Runs three concurrent queries:
          - Exact match: trials with both drug AND condition (max 50)
          - Drug-only count: total trials with this drug (any condition)
          - Condition-only count: total trials with this condition (any drug)

        If no exact matches (whitespace exists), fetches condition trials and
        populates condition_drugs with other drugs being tested for this condition:
          - Filters to Phase 2+ trials only (excludes noisy Phase 1 data)
          - Ranks by phase (descending) then active status (recruiting preferred)
          - Deduplicates by drug_name, keeping the highest-ranked trial per drug
          - Returns top 50 unique drugs

        This tells the Trial Agent what competitors exist in the condition space
        when the queried drug has no trials there.
        """
        # All three are independent — run concurrently
        exact_task = self.search_trials(
            drug=drug,
            condition=condition,
            date_before=date_before,
            max_results=CLINICAL_TRIALS_WHITESPACE_EXACT_MAX,
        )
        drug_count_task = self._count_trials(
            drug=drug, condition=None, date_before=date_before
        )
        condition_count_task = self._count_trials(
            drug=None, condition=condition, date_before=date_before
        )

        exact_trials, drug_count, condition_count = await asyncio.gather(
            exact_task, drug_count_task, condition_count_task
        )

        # Condition drugs: only populated when whitespace exists
        # Restrict to Phase 2+ for meaningful efficacy signal
        condition_drugs: list[ConditionDrug] = []
        if not exact_trials:
            condition_trials = await self._fetch_all_condition_trials(
                condition,
                date_before=date_before,
                max_results=CLINICAL_TRIALS_WHITESPACE_CONDITION_MAX,
                phase_filter=CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER,
            )

            # Collect drug/biologic trials as candidates
            candidates: list[ConditionDrug] = []
            for t in condition_trials:
                for interv in t.interventions:
                    if interv.intervention_type in ("Drug", "Biological"):
                        candidates.append(
                            ConditionDrug(
                                nct_id=t.nct_id,
                                drug_name=interv.intervention_name,
                                condition=t.conditions[0] if t.conditions else "",
                                phase=t.phase,
                                status=t.overall_status,
                            )
                        )
                        break  # one per trial

            # Rank: most advanced phase first, then by active status
            active_statuses = {
                "RECRUITING",
                "ACTIVE_NOT_RECRUITING",
                "ENROLLING_BY_INVITATION",
            }
            candidates.sort(
                key=lambda cd: (
                    self._phase_rank(cd.phase),
                    cd.status in active_statuses,
                ),
                reverse=True,
            )

            # Deduplicate by drug_name, keeping highest-ranked entry
            seen_drugs: set[str] = set()
            unique_candidates: list[ConditionDrug] = []
            for cd in candidates:
                if cd.drug_name not in seen_drugs:
                    seen_drugs.add(cd.drug_name)
                    unique_candidates.append(cd)

            condition_drugs = unique_candidates[:CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS]

        return WhitespaceResult(
            is_whitespace=len(exact_trials) == 0,
            exact_match_count=len(exact_trials),
            drug_only_trials=drug_count,
            condition_only_trials=condition_count,
            condition_drugs=condition_drugs,
        )

    # ------------------------------------------------------------------
    # Public: get_landscape
    # ------------------------------------------------------------------

    async def get_landscape(
        self,
        condition: str,
        date_before: date | None = None,
        top_n: int = 50,
    ) -> ConditionLandscape:
        """Competitive landscape for a condition — drug/biologic trials, grouped by sponsor + drug.

        Fetches all trials for the condition, then filters client-side to
        intervention_type in ("Drug", "Biological") only. Ranks competitors
        by phase then enrollment. Returns top_n competitors.
        """
        trials = await self._fetch_all_condition_trials(
            condition,
            date_before=date_before,
            phase_filter="(EARLY_PHASE1 OR PHASE1 OR PHASE2 OR PHASE3 OR PHASE4)",
        )

        return self._aggregate_landscape(trials, top_n=top_n)

    # ------------------------------------------------------------------
    # Public: get_terminated
    # ------------------------------------------------------------------

    async def get_terminated(
        self,
        query: str,
        date_before: date | None = None,
        max_results: int = 100,
    ) -> list[TerminatedTrial]:
        """Terminated/withdrawn/suspended trials for a drug, class, or condition."""
        params = self._build_search_params(
            drug=None,
            condition=None,
            date_before=date_before,
            extra_term=query,
            status_filter="TERMINATED,WITHDRAWN,SUSPENDED",
        )
        data = await self._rest_get(self.BASE_URL, params)
        studies = data.get("studies", [])
        return [self._parse_terminated_trial(s) for s in studies[:max_results]]

    # ------------------------------------------------------------------
    # Private: condition-level fetching (no drug filter)
    # ------------------------------------------------------------------

    async def _fetch_all_condition_trials(
        self,
        condition: str,
        date_before: date | None = None,
        max_results: int | None = None,
        phase_filter: str | None = None,
    ) -> list[Trial]:
        """Fetch all trials for a condition.

        If max_results is None, paginates until exhausted.
        """
        trials: list[Trial] = []
        page_token: str | None = None

        while max_results is None or len(trials) < max_results:
            params = self._build_search_params(
                condition=condition,
                date_before=date_before,
                page_token=page_token,
                phase_filter=phase_filter,
            )
            data = await self._rest_get(self.BASE_URL, params)

            studies = data.get("studies", [])
            trials.extend(self._parse_trial(s) for s in studies)

            page_token = data.get("nextPageToken")
            if not page_token or len(studies) < self.PAGE_SIZE:
                break

        return trials[:max_results] if max_results else trials

    # ------------------------------------------------------------------
    # Private: parameter building
    # ------------------------------------------------------------------

    def _build_search_params(
        self,
        *,
        drug: str | None = None,
        condition: str | None = None,
        date_before: date | None = None,
        page_token: str | None = None,
        extra_term: str | None = None,
        status_filter: str | None = None,
        phase_filter: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": self.PAGE_SIZE,
            "countTotal": "true",
        }

        if condition:
            params["query.cond"] = condition
        if drug:
            params["query.intr"] = drug
        if extra_term:
            params["query.term"] = extra_term

        # Temporal holdout: restrict to trials posted before cutoff
        if date_before:
            date_str = date_before.strftime("%Y-%m-%d")
            term = params.get("query.term", "")
            date_filter = f"AREA[StudyFirstPostDate]RANGE[MIN, {date_str}]"
            params["query.term"] = (
                f"{term} {date_filter}".strip() if term else date_filter
            )

        if status_filter:
            params["filter.overallStatus"] = status_filter

        if phase_filter:
            # Phase filter uses AREA[Phase] syntax in query.term
            term = params.get("query.term", "")
            phase_term = f"AREA[Phase]{phase_filter}"
            params["query.term"] = (
                f"{term} {phase_term}".strip() if term else phase_term
            )

        if page_token:
            params["pageToken"] = page_token

        return params

    async def _count_trials(
        self,
        drug: str | None,
        condition: str | None,
        date_before: date | None = None,
    ) -> int:
        """Quick count without fetching full records."""
        params = self._build_search_params(
            drug=drug,
            condition=condition,
            date_before=date_before,
        )
        params["pageSize"] = 1  # we only need the count

        data = await self._rest_get(self.BASE_URL, params)
        return data.get("totalCount", 0)

    # ------------------------------------------------------------------
    # Parsers: v2 API response → Pydantic models
    # ------------------------------------------------------------------

    def _parse_trial(self, study: dict) -> Trial:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        outcomes = proto.get("outcomesModule", {})
        refs = proto.get("referencesModule", {})

        # Interventions
        interventions = [
            Intervention(
                intervention_type=i.get("type", "").replace("_", " ").title(),
                intervention_name=i.get("name", ""),
                description=i.get("description"),
            )
            for i in arms.get("interventions", [])
        ]

        # Primary outcomes
        primary_outcomes = [
            PrimaryOutcome(
                measure=o.get("measure", ""),
                time_frame=o.get("timeFrame"),
            )
            for o in outcomes.get("primaryOutcomes", [])
        ]

        # Phases — v2 returns a list like ["PHASE2", "PHASE3"]
        phases_raw = design.get("phases", [])
        phase = self._normalize_phase(phases_raw)

        # Collaborators
        collabs = [c.get("name", "") for c in sponsor_mod.get("collaborators", [])]

        # PMIDs from references
        pmids = [r["pmid"] for r in refs.get("references", []) if r.get("pmid")]

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count")

        return Trial(
            nct_id=ident.get("nctId", ""),
            title=ident.get("briefTitle", ""),
            brief_summary=desc.get("briefSummary"),
            phase=phase,
            overall_status=status.get("overallStatus", ""),
            why_stopped=status.get("whyStopped"),
            conditions=proto.get("conditionsModule", {}).get("conditions", []),
            interventions=interventions,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
            collaborators=collabs,
            enrollment=enrollment,
            start_date=self._extract_date(status.get("startDateStruct")),
            completion_date=self._extract_date(
                status.get("primaryCompletionDateStruct")
            ),
            study_type=design.get("studyType", ""),
            primary_outcomes=primary_outcomes,
            results_posted=study.get("hasResults", False),
            references=pmids,
        )

    def _parse_terminated_trial(self, study: dict) -> TerminatedTrial:
        """Parse a study into a TerminatedTrial with stop classification."""
        trial = self._parse_trial(study)

        # Extract primary drug name from interventions
        drug_name = None
        for interv in trial.interventions:
            if interv.intervention_type in ("Drug", "Biological"):
                drug_name = interv.intervention_name
                break

        return TerminatedTrial(
            nct_id=trial.nct_id,
            title=trial.title,
            drug_name=drug_name,
            condition=trial.conditions[0] if trial.conditions else None,
            phase=trial.phase,
            why_stopped=trial.why_stopped,
            stop_category=_classify_stop_reason(trial.why_stopped),
            enrollment=trial.enrollment,
            sponsor=trial.sponsor,
            start_date=trial.start_date,
            termination_date=trial.completion_date,
            references=trial.references,
        )

    # ------------------------------------------------------------------
    # Aggregation: landscape builder
    # ------------------------------------------------------------------

    def _aggregate_landscape(
        self, trials: list[Trial], top_n: int = 50
    ) -> ConditionLandscape:
        """Group trials by sponsor + drug into a competitive landscape.

        Filters to Drug/Biological interventions only. Ranks competitors
        by max phase (descending), then total enrollment (descending).
        """
        phase_dist: dict[str, int] = {}
        recent_starts: list[RecentStart] = []
        competitors: dict[str, CompetitorEntry] = {}  # key: "sponsor|drug"

        for t in trials:
            # Only count trials with a Drug or Biological intervention
            drug_name = self._primary_drug_name(t)
            if drug_name == "Unknown":
                continue

            # Phase counts (all drug trials)
            phase_dist[t.phase] = phase_dist.get(t.phase, 0) + 1

            # Recent starts (last 2 years)
            if t.start_date and t.start_date >= CLINICAL_TRIALS_RECENT_START_YEAR:
                recent_starts.append(
                    RecentStart(
                        nct_id=t.nct_id,
                        sponsor=t.sponsor,
                        drug=drug_name,
                        phase=t.phase,
                    )
                )

            # Group by sponsor + drug
            key = f"{t.sponsor}|{drug_name}"

            if key not in competitors:
                competitors[key] = CompetitorEntry(
                    sponsor=t.sponsor,
                    drug_name=drug_name,
                    drug_type=self._primary_drug_type(t),
                    max_phase=t.phase,
                    trial_count=0,
                    statuses=set(),
                    total_enrollment=0,
                    most_recent_start=None,
                )

            entry = competitors[key]
            entry.trial_count += 1
            entry.statuses.add(t.overall_status)
            entry.total_enrollment += t.enrollment or 0

            if self._phase_rank(t.phase) > self._phase_rank(entry.max_phase):
                entry.max_phase = t.phase

            if t.start_date and (
                entry.most_recent_start is None
                or t.start_date > entry.most_recent_start
            ):
                entry.most_recent_start = t.start_date

        # Rank: highest phase first, then largest enrollment
        ranked = sorted(
            competitors.values(),
            key=lambda c: (self._phase_rank(c.max_phase), c.total_enrollment),
            reverse=True,
        )

        return ConditionLandscape(
            total_trial_count=len(trials),
            competitors=ranked[:top_n],
            phase_distribution=phase_dist,
            recent_starts=recent_starts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_phase(phases: list[str]) -> str:
        """Convert v2 phase list like ['PHASE2', 'PHASE3'] → 'Phase 2/Phase 3'."""
        if not phases:
            return "Not Applicable"
        mapping = {
            "EARLY_PHASE1": "Early Phase 1",
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "NA": "Not Applicable",
        }
        normalized = [mapping.get(p, p) for p in phases]
        return "/".join(normalized)

    @staticmethod
    def _extract_date(date_struct: dict | None) -> str | None:
        """Extract date string from v2 date struct like {'date': '2021-03-15'}."""
        if not date_struct:
            return None
        return date_struct.get("date")

    @staticmethod
    def _primary_drug_name(trial: Trial) -> str:
        """Extract the primary drug intervention name, or 'Unknown'."""
        for interv in trial.interventions:
            if interv.intervention_type in ("Drug", "Biological"):
                return interv.intervention_name
        return "Unknown"

    @staticmethod
    def _primary_drug_type(trial: Trial) -> str | None:
        """Extract intervention type of the primary drug."""
        for interv in trial.interventions:
            if interv.intervention_type in ("Drug", "Biological"):
                return interv.intervention_type
        return None

    @staticmethod
    def _phase_rank(phase: str) -> int:
        """Numeric rank for phase comparison. Higher = later stage."""
        ranks = {
            "Not Applicable": 0,
            "Early Phase 1": 1,
            "Phase 1": 2,
            "Phase 1/Phase 2": 3,
            "Phase 2": 4,
            "Phase 2/Phase 3": 5,
            "Phase 3": 6,
            "Phase 3/Phase 4": 7,
            "Phase 4": 8,
        }
        return ranks.get(phase, 0)
