"""
ClinicalTrials.gov REST API v2 client.

Five methods:
  1. get_trial         — Fetch a single trial by NCT ID
  2. search_trials     — Trial Agent: drug + indication → trial records
  3. detect_whitespace — Trial Agent: is this drug-indication pair unexplored?
  4. get_landscape     — Landscape Agent: competitive map for an indication
  5. get_terminated    — Critique Agent: what has failed in this space?
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from indication_scout.constants import (
    CLINICAL_TRIALS_BASE_URL,
    CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS,
    CLINICAL_TRIALS_RECENT_START_YEAR,
    CLINICAL_TRIALS_TERMINATED_DRUG_PAGE_SIZE,
    CLINICAL_TRIALS_WHITESPACE_INDICATION_MAX,
    CLINICAL_TRIALS_WHITESPACE_EXACT_MAX,
    CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER,
    CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS,
    NEGATION_PREFIXES,
    STOP_KEYWORDS,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError

from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    IndicationDrug,
    IndicationLandscape,
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
            idx = lower.index(keyword)
            prefix = lower[max(0, idx - 20):idx]
            if any(neg in prefix for neg in NEGATION_PREFIXES):
                neg_end = max(prefix.rfind(neg) + len(neg) for neg in NEGATION_PREFIXES if neg in prefix)
                between = prefix[neg_end:]
                if not any(sep in between for sep in (",", "-", ".", ";")):
                    continue
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
        """
        Fetch a single trial by NCT ID.

        Args:  nct_id: The NCT identifier (e.g., "NCT04971785")
        Returns: Trial object with all available data
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
        indication: str | None = None,
        date_before: date | None = None,
        phase_filter: str | None = None,
        max_results: int = 200,
    ) -> list[Trial]:
        """Search for trials matching drug and optional indication."""
        return await self._paginated_search(
            drug=drug,
            indication=indication,
            date_before=date_before,
            phase_filter=phase_filter,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Public: detect_whitespace
    # ------------------------------------------------------------------

    async def detect_whitespace(
        self,
        drug: str,
        indication: str,
        date_before: date | None = None,
    ) -> WhitespaceResult:
        """Is this drug-indication pair being explored in clinical trials?

        Runs three concurrent queries:
          - Exact match: trials with both drug AND indication (max 50)
          - Drug-only count: total trials with this drug (any indication)
          - Indication-only count: total trials with this indication (any drug)

        If no exact matches (whitespace exists), fetches indication trials and
        populates indication_drugs with other drugs being tested for this indication:
          - Filters to Phase 2+ trials only (excludes noisy Phase 1 data)
          - Ranks by phase (descending) then active status (recruiting preferred)
          - Deduplicates by drug_name, keeping the highest-ranked trial per drug
          - Returns top 50 unique drugs

        This tells the Trial Agent what competitors exist in the indication space
        when the queried drug has no trials there.
        """
        # All three are independent — run concurrently
        exact_task = self.search_trials(
            drug=drug,
            indication=indication,
            date_before=date_before,
            max_results=CLINICAL_TRIALS_WHITESPACE_EXACT_MAX,
        )
        drug_count_task = self._count_trials(
            drug=drug, indication=None, date_before=date_before
        )
        indication_count_task = self._count_trials(
            drug=None, indication=indication, date_before=date_before
        )

        exact_trials, drug_count, indication_count = await asyncio.gather(
            exact_task, drug_count_task, indication_count_task
        )

        # Indication drugs: only populated when whitespace exists
        # Restrict to Phase 2+ for meaningful efficacy signal
        indication_drugs: list[IndicationDrug] = []
        if not exact_trials:
            indication_trials = await self._fetch_all_indication_trials(
                indication,
                date_before=date_before,
                max_results=CLINICAL_TRIALS_WHITESPACE_INDICATION_MAX,
                phase_filter=CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER,
            )

            # Collect drug/biologic trials as candidates
            candidates: list[IndicationDrug] = []
            for t in indication_trials:
                drug_name, _ = self._primary_drug(t)
                if drug_name != "Unknown":
                    candidates.append(IndicationDrug.from_trial(t, drug_name))

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
            unique_candidates: list[IndicationDrug] = []
            for cd in candidates:
                if cd.drug_name not in seen_drugs:
                    seen_drugs.add(cd.drug_name)
                    unique_candidates.append(cd)

            indication_drugs = unique_candidates[:CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS]

        return WhitespaceResult(
            is_whitespace=len(exact_trials) == 0,
            no_data=drug_count == 0 and indication_count == 0,
            exact_match_count=len(exact_trials),
            drug_only_trials=drug_count,
            indication_only_trials=indication_count,
            indication_drugs=indication_drugs,
        )

    # ------------------------------------------------------------------
    # Public: get_landscape
    # ------------------------------------------------------------------

    async def get_landscape(
        self,
        indication: str,
        date_before: date | None = None,
        top_n: int = 50,
    ) -> IndicationLandscape:
        """Competitive landscape for an indication — drug/biologic trials, grouped by sponsor + drug.

        Fetches all trials for the indication, then filters client-side to
        intervention_type in ("Drug", "Biological") only. Ranks competitors
        by phase then enrollment. Returns top_n competitors.
        """
        trials, total_count = await asyncio.gather(
            self._fetch_all_indication_trials(
                indication,
                date_before=date_before,
                phase_filter="(EARLY_PHASE1 OR PHASE1 OR PHASE2 OR PHASE3 OR PHASE4)",
                max_results=CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS,
                sort="EnrollmentCount:desc",
            ),
            self._count_trials(drug=None, indication=indication, date_before=date_before),
        )

        return self._aggregate_landscape(trials, total_count=total_count, top_n=top_n)

    # ------------------------------------------------------------------
    # Public: get_terminated
    # ------------------------------------------------------------------

    async def get_terminated(
        self,
        drug: str,
        indication: str,
        date_before: date | None = None,
        max_results: int = 20,
    ) -> list[TerminatedTrial]:
        """Terminated trials for a drug and indication.

        Runs two concurrent queries:
          - Drug query: safety/efficacy terminations for this drug (any indication).
            Filtered to stop_category in {safety, efficacy} only — enrollment,
            business, and unknown terminations are dropped as noise.
          - Indication query: what has failed in this indication space (any drug),
            up to max_results.
        Returns the union, deduped by nct_id.
        """
        drug_params = self._build_search_params(
            drug=drug,
            date_before=date_before,
            status_filter="TERMINATED",
        )
        drug_params["pageSize"] = CLINICAL_TRIALS_TERMINATED_DRUG_PAGE_SIZE
        indication_params = self._build_search_params(
            indication=indication,
            date_before=date_before,
            status_filter="TERMINATED",
        )
        drug_data, indication_data = await asyncio.gather(
            self._rest_get(self.BASE_URL, drug_params),
            self._rest_get(self.BASE_URL, indication_params),
        )

        # Drug query: only safety/efficacy terminations are meaningful signal
        drug_results = [
            self._parse_terminated_trial(s)
            for s in drug_data.get("studies", [])
        ]
        drug_results = [
            t for t in drug_results
            if t.stop_category in {"safety", "efficacy"}
        ]

        # Indication query: all terminations up to max_results
        indication_results = [
            self._parse_terminated_trial(s)
            for s in indication_data.get("studies", [])
        ][:max_results]

        seen: set[str] = set()
        results: list[TerminatedTrial] = []
        for trial in drug_results + indication_results:
            if trial.nct_id not in seen:
                seen.add(trial.nct_id)
                results.append(trial)
        return results

    # ------------------------------------------------------------------
    # Private: indication-level fetching (no drug filter)
    # ------------------------------------------------------------------

    async def _fetch_all_indication_trials(
        self,
        indication: str,
        date_before: date | None = None,
        max_results: int | None = None,
        phase_filter: str | None = None,
        sort: str | None = None,
    ) -> list[Trial]:
        """Fetch all trials for an indication.

        If max_results is None, paginates until exhausted.
        """
        return await self._paginated_search(
            indication=indication,
            date_before=date_before,
            phase_filter=phase_filter,
            max_results=max_results,
            sort=sort,
        )

    # ------------------------------------------------------------------
    # Private: paginated search
    # ------------------------------------------------------------------

    async def _paginated_search(
        self,
        *,
        drug: str | None = None,
        indication: str | None = None,
        date_before: date | None = None,
        phase_filter: str | None = None,
        max_results: int | None = None,
        sort: str | None = None,
    ) -> list[Trial]:
        """Core pagination loop shared by all trial-fetching methods.

        If max_results is None, paginates until exhausted.
        """
        trials: list[Trial] = []
        page_token: str | None = None

        while max_results is None or len(trials) < max_results:
            params = self._build_search_params(
                drug=drug,
                indication=indication,
                date_before=date_before,
                phase_filter=phase_filter,
                page_token=page_token,
                sort=sort,
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
        indication: str | None = None,
        date_before: date | None = None,
        page_token: str | None = None,
        extra_term: str | None = None,
        status_filter: str | None = None,
        phase_filter: str | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": self.PAGE_SIZE,
            "countTotal": "true",
        }

        if indication:
            params["query.cond"] = (
                indication  # API uses "cond" for indication/indication
            )
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

        if sort:
            params["sort"] = sort

        return params

    async def _count_trials(
        self,
        drug: str | None,
        indication: str | None,
        date_before: date | None = None,
    ) -> int:
        """Quick count without fetching full records."""
        params = self._build_search_params(
            drug=drug,
            indication=indication,
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
            indications=proto.get("conditionsModule", {}).get("conditions", []),
            interventions=interventions,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
            enrollment=enrollment,
            start_date=self._extract_date(status.get("startDateStruct")),
            completion_date=self._extract_date(
                status.get("primaryCompletionDateStruct")
            ),
            primary_outcomes=primary_outcomes,
            references=pmids,
        )

    def _parse_terminated_trial(self, study: dict) -> TerminatedTrial:
        """Parse a study into a TerminatedTrial with stop classification."""
        trial = self._parse_trial(study)

        drug_name, _ = self._primary_drug(trial)
        if drug_name == "Unknown":
            drug_name = None

        return TerminatedTrial(
            nct_id=trial.nct_id,
            title=trial.title,
            drug_name=drug_name,
            indication=trial.indications[0] if trial.indications else None,
            phase=trial.phase,
            why_stopped=trial.why_stopped,
            stop_category=_classify_stop_reason(trial.why_stopped),
            enrollment=trial.enrollment,
            sponsor=trial.sponsor,
            start_date=trial.start_date,
            termination_date=trial.completion_date,
        )

    # ------------------------------------------------------------------
    # Aggregation: landscape builder
    # ------------------------------------------------------------------

    def _aggregate_landscape(
        self, trials: list[Trial], total_count: int, top_n: int = 50
    ) -> IndicationLandscape:
        """Group trials by sponsor + drug into a competitive landscape.

        Filters to Drug/Biological interventions only. Ranks competitors
        by max phase (descending), then total enrollment (descending).
        """
        phase_dist: dict[str, int] = {}
        recent_starts: list[RecentStart] = []
        competitors: dict[str, CompetitorEntry] = {}  # key: "sponsor|drug"

        for t in trials:
            # Only count trials with a Drug or Biological intervention
            drug_name, drug_type = self._primary_drug(t)
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
                    drug_type=drug_type,
                    max_phase=t.phase,
                    trial_count=0,
                    statuses=set(),
                    total_enrollment=0,
                )

            entry = competitors[key]
            entry.trial_count += 1
            entry.statuses.add(t.overall_status)
            entry.total_enrollment += t.enrollment or 0

            if self._phase_rank(t.phase) > self._phase_rank(entry.max_phase):
                entry.max_phase = t.phase

        # Rank: highest phase first, then largest enrollment
        ranked = sorted(
            competitors.values(),
            key=lambda c: (self._phase_rank(c.max_phase), c.total_enrollment),
            reverse=True,
        )

        return IndicationLandscape(
            total_trial_count=total_count,
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
    def _primary_drug(trial: Trial) -> tuple[str, str | None]:
        """Extract the primary Drug/Biological intervention's name and type.

        Returns:
            (drug_name, intervention_type) — defaults to ("Unknown", None)
            if no Drug/Biological intervention is found.
        """
        for interv in trial.interventions:
            if interv.intervention_type in ("Drug", "Biological"):
                return interv.intervention_name, interv.intervention_type
        return "Unknown", None

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
