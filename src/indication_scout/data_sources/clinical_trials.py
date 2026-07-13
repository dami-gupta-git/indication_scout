"""
ClinicalTrials.gov REST API v2 client.

Pair-scoped query methods follow a count + top-50 exemplars pattern:
  1. get_trial               — Fetch a single trial by NCT ID
  2. search_trials           — All-status trials for a drug × indication pair
  3. get_completed_trials    — COMPLETED trials for a drug × indication pair
  4. get_terminated_trials   — TERMINATED trials for a drug × indication pair
  5. get_landscape           — Competitive map for an indication
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path
from typing import Any

from indication_scout.config import get_settings
from indication_scout.constants import (
    CLINICAL_TRIALS_BASE_URL,
    CLINICAL_TRIALS_CACHE_TTL,
    CLINICAL_TRIALS_FETCH_MAX,
    CLINICAL_TRIALS_RECENT_START_YEAR,
    DEFAULT_CACHE_DIR,
    VACCINE_NAME_KEYWORDS,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()


def _mesh_cond(mesh_term: str) -> str:
    """Format a MeSH preferred term as a CT.gov server-side condition filter."""
    # KNOWN BUG (not yet fixed): AREA[ConditionMeshTerm] matches a trial when the
    # term appears in EITHER its direct condition meshes OR its mesh ancestors.
    # So "Hypertension" (D006973) also pulls in Pulmonary Hypertension trials
    # (D006976), whose tree rolls up under Hypertension — exactly the mixing the
    # design intends to avoid. Fix belongs here at the source: after fetching,
    # require the resolved descriptor to appear in the trial's direct
    # mesh_conditions (drop ancestor-only matches), keyed on MeSH ID not term.
    return f'AREA[ConditionMeshTerm]"{mesh_term}"'


from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    Intervention,
    MeshTerm,
    PrimaryOutcome,
    RecentStart,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)


class ClinicalTrialsClient(BaseClient):
    BASE_URL = CLINICAL_TRIALS_BASE_URL
    PAGE_SIZE = 100

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        super().__init__()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

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
    # Public: search_trials (all-status pair query: counts + top-50)
    # ------------------------------------------------------------------

    async def search_trials(
        self,
        drug: str,
        mesh_term: str,
        date_before: date | None = None,
    ) -> SearchTrialsResult:
        """All-status trials for a drug × indication pair.

        Indication is filtered server-side via CT.gov's
        `AREA[ConditionMeshTerm]"<mesh_term>"` syntax (precise descriptor
        match — no ancestor/descendant fuzziness, no free-text noise).
        Drug side stays free-text via `query.intr` so trials whose
        intervention isn't MeSH-tagged are still caught.

        Issues four cheap count calls (total + RECRUITING +
        ACTIVE_NOT_RECRUITING + WITHDRAWN) plus one fetch of up to
        CLINICAL_TRIALS_FETCH_MAX records sorted by enrollment desc. The
        same query.cond filter is used for counts and fetch, so
        `len(trials) <= total_count` always.

        TERMINATED and COMPLETED counts are NOT reported here — those live
        on get_terminated_trials and get_completed_trials respectively to
        avoid surfacing the same number from two places.
        """
        cache_params = {
            "drug": drug,
            "mesh_term": mesh_term,
            "date_before": date_before.isoformat() if date_before else None,
            "fetch_max": CLINICAL_TRIALS_FETCH_MAX,
        }
        cached = cache_get("ct_search", cache_params, self.cache_dir)
        if cached is not None:
            return SearchTrialsResult.model_validate(cached)

        cond = _mesh_cond(mesh_term)
        total_task = self._count_trials_total(
            drug=drug, indication=cond, date_before=date_before
        )
        recruiting_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="RECRUITING",
        )
        active_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="ACTIVE_NOT_RECRUITING",
        )
        withdrawn_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="WITHDRAWN",
        )
        # UNKNOWN: CT.gov auto-assigns when a record hasn't been updated in
        # ~2 years. The trial ran but outcome is unknowable from CT.gov status.
        # Critical for repurposing analysis — these must NOT be confused with
        # "trial never happened."
        unknown_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="UNKNOWN",
        )
        fetch_task = self._paginated_search(
            drug=drug,
            indication=cond,
            date_before=date_before,
            max_results=CLINICAL_TRIALS_FETCH_MAX,
            sort="EnrollmentCount:desc",
        )

        total, recruiting, active, withdrawn, unknown, (trials, _) = (
            await asyncio.gather(
                total_task,
                recruiting_task,
                active_task,
                withdrawn_task,
                unknown_task,
                fetch_task,
            )
        )

        result = SearchTrialsResult(
            total_count=total,
            by_status={
                "RECRUITING": recruiting,
                "ACTIVE_NOT_RECRUITING": active,
                "WITHDRAWN": withdrawn,
                "UNKNOWN": unknown,
            },
            trials=trials,
        )
        cache_set(
            "ct_search",
            cache_params,
            result.model_dump(mode="json"),
            self.cache_dir,
            ttl=CLINICAL_TRIALS_CACHE_TTL,
        )
        return result

    # ------------------------------------------------------------------
    # Public: get_landscape
    # ------------------------------------------------------------------

    async def get_landscape(
        self,
        mesh_term: str,
        date_before: date | None = None,
        top_n: int = 50,
    ) -> IndicationLandscape:
        """Competitive landscape for an indication — drug/biologic trials, grouped by sponsor + drug.

        Indication is filtered server-side via `AREA[ConditionMeshTerm]"<mesh_term>"`
        — same precise descriptor match as the pair-scoped query methods.
        Fetches trials for the indication sorted by most recent start date,
        capped at clinical_trials_landscape_max_trials. Then filters
        client-side to Drug/Biological interventions (vaccines excluded).
        Ranks competitors by max phase, then most recent start date.
        Returns top_n competitors after filtering.

        Total count comes from a single `countTotal` call — exact, no page cap.
        """
        cache_params = {
            "mesh_term": mesh_term,
            "date_before": date_before.isoformat() if date_before else None,
            "top_n": top_n,
            "landscape_max": _settings.clinical_trials_landscape_max_trials,
        }
        cached = cache_get("ct_landscape", cache_params, self.cache_dir)
        if cached is not None:
            return IndicationLandscape.model_validate(cached)

        landscape_max = _settings.clinical_trials_landscape_max_trials
        cond = _mesh_cond(mesh_term)

        fetch_task = self._fetch_all_indication_trials(
            cond,
            date_before=date_before,
            phase_filter="(EARLY_PHASE1 OR PHASE1 OR PHASE2 OR PHASE3 OR PHASE4)",
            max_results=landscape_max,
            sort="StartDate:desc",
        )
        count_task = self._count_trials_total(
            indication=cond,
            date_before=date_before,
        )

        (trials, _), total_count = await asyncio.gather(fetch_task, count_task)

        result = self._aggregate_landscape(trials, total_count=total_count, top_n=top_n)
        cache_set(
            "ct_landscape",
            cache_params,
            result.model_dump(mode="json"),
            self.cache_dir,
            ttl=CLINICAL_TRIALS_CACHE_TTL,
        )
        return result

    # ------------------------------------------------------------------
    # Public: get_terminated_trials (TERMINATED pair query: count + top-50)
    # ------------------------------------------------------------------

    async def get_terminated_trials(
        self,
        drug: str,
        mesh_term: str,
        date_before: date | None = None,
    ) -> TerminatedTrialsResult:
        """TERMINATED trials for a drug × indication pair.

        Indication is filtered server-side via `AREA[ConditionMeshTerm]`.
        One count call (total terminated) plus one fetch of up to
        CLINICAL_TRIALS_FETCH_MAX TERMINATED records sorted by enrollment
        desc. Each returned Trial carries `why_stopped`; stop-category
        classification happens at the tool layer (no separate model field).
        """
        cache_params = {
            "drug": drug,
            "mesh_term": mesh_term,
            "date_before": date_before.isoformat() if date_before else None,
            "fetch_max": CLINICAL_TRIALS_FETCH_MAX,
        }
        cached = cache_get("ct_terminated", cache_params, self.cache_dir)
        if cached is not None:
            return TerminatedTrialsResult.model_validate(cached)

        cond = _mesh_cond(mesh_term)
        total_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="TERMINATED",
        )
        fetch_task = self._paginated_search(
            drug=drug,
            indication=cond,
            date_before=date_before,
            max_results=CLINICAL_TRIALS_FETCH_MAX,
            sort="EnrollmentCount:desc",
            status_filter="TERMINATED",
        )

        total, (trials, _) = await asyncio.gather(total_task, fetch_task)

        await self._augment_references_via_pubmed(trials, date_before=date_before)

        result = TerminatedTrialsResult(total_count=total, trials=trials)
        cache_set(
            "ct_terminated",
            cache_params,
            result.model_dump(mode="json"),
            self.cache_dir,
            ttl=CLINICAL_TRIALS_CACHE_TTL,
        )
        return result

    # ------------------------------------------------------------------
    # Private: PubMed NCT-fallback for missing trial references
    # ------------------------------------------------------------------

    async def _augment_references_via_pubmed(
        self, trials: list[Trial], date_before: date | None = None
    ) -> None:
        """Backfill empty `Trial.references` via PubMed NCT-id search.

        CT.gov's `referencesModule.references` is incomplete — many trials
        have a real published readout in PubMed that CT.gov never linked.
        PubMed's `[si]` (Secondary Source ID) field tag matches NCT numbers
        cited in abstracts, so a query like `NCT01490632[si]` recovers the
        readout paper directly.

        Mutates `trials` in-place. Only fills `references` when it's empty;
        never overwrites a CT.gov-provided PMID list. Failures (PubMed
        errors, timeouts) log a warning and leave references untouched.
        Honors `date_before` for holdout correctness.
        """
        targets = [t for t in trials if not t.references and t.nct_id]
        if not targets:
            return
        async with PubMedClient(cache_dir=self.cache_dir) as pm:
            tasks = [
                pm.search(
                    query=f"{t.nct_id}[si]", max_results=10, date_before=date_before
                )
                for t in targets
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(
                    "PubMed NCT-fallback gather failed: %r — leaving references empty",
                    e,
                )
                return
        for trial, pmids in zip(targets, results):
            if isinstance(pmids, Exception):
                logger.warning(
                    "PubMed NCT-fallback for %s failed: %r — leaving references empty",
                    trial.nct_id,
                    pmids,
                )
                continue
            if pmids:
                trial.references = list(pmids)

    # ------------------------------------------------------------------
    # Public: get_completed_trials (COMPLETED pair query: count + top-50)
    # ------------------------------------------------------------------

    async def get_completed_trials(
        self,
        drug: str,
        mesh_term: str,
        date_before: date | None = None,
    ) -> CompletedTrialsResult:
        """COMPLETED trials for a drug × indication pair.

        Indication is filtered server-side via `AREA[ConditionMeshTerm]`.
        One count call (total completed) plus one fetch of up to
        CLINICAL_TRIALS_FETCH_MAX COMPLETED records sorted by enrollment desc.
        Phase information is read off each returned Trial.
        """
        cache_params = {
            "drug": drug,
            "mesh_term": mesh_term,
            "date_before": date_before.isoformat() if date_before else None,
            "fetch_max": CLINICAL_TRIALS_FETCH_MAX,
        }
        cached = cache_get("ct_completed", cache_params, self.cache_dir)
        if cached is not None:
            return CompletedTrialsResult.model_validate(cached)

        cond = _mesh_cond(mesh_term)
        total_task = self._count_trials_total(
            drug=drug,
            indication=cond,
            date_before=date_before,
            status_filter="COMPLETED",
        )
        fetch_task = self._paginated_search(
            drug=drug,
            indication=cond,
            date_before=date_before,
            max_results=CLINICAL_TRIALS_FETCH_MAX,
            sort="EnrollmentCount:desc",
            status_filter="COMPLETED",
        )

        total, (trials, _) = await asyncio.gather(total_task, fetch_task)

        await self._augment_references_via_pubmed(trials, date_before=date_before)

        result = CompletedTrialsResult(total_count=total, trials=trials)
        cache_set(
            "ct_completed",
            cache_params,
            result.model_dump(mode="json"),
            self.cache_dir,
            ttl=CLINICAL_TRIALS_CACHE_TTL,
        )
        return result

    # ------------------------------------------------------------------
    # Private: indication-level fetching (no drug filter)
    # ------------------------------------------------------------------

    async def _fetch_all_indication_trials(
        self,
        indication: str,
        date_before: date | None = None,
        max_results: int | None = None,
        max_pages: int | None = None,
        phase_filter: str | None = None,
        sort: str | None = None,
    ) -> tuple[list[Trial], bool]:
        """Fetch trials for an indication.

        If max_results is None, paginates until exhausted (or until max_pages
        is reached, if set). Returns (trials, saturated).
        """
        return await self._paginated_search(
            indication=indication,
            date_before=date_before,
            phase_filter=phase_filter,
            max_results=max_results,
            max_pages=max_pages,
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
        status_filter: str | None = None,
        max_results: int | None = None,
        max_pages: int | None = None,
        sort: str | None = None,
    ) -> tuple[list[Trial], bool]:
        """Core pagination loop shared by all trial-fetching methods.

        If max_results is None, paginates until exhausted (or until max_pages
        is reached, if set). Returns (trials, saturated) where saturated is
        True if a cap stopped the walk before CT.gov indicated exhaustion.
        """
        trials: list[Trial] = []
        page_token: str | None = None
        pages_fetched = 0
        saturated = False

        while max_results is None or len(trials) < max_results:
            if max_pages is not None and pages_fetched >= max_pages:
                saturated = True
                break

            params = self._build_search_params(
                drug=drug,
                indication=indication,
                date_before=date_before,
                phase_filter=phase_filter,
                status_filter=status_filter,
                page_token=page_token,
                sort=sort,
            )
            data = await self._rest_get(self.BASE_URL, params)
            studies = data.get("studies", [])
            trials.extend(self._parse_trial(s) for s in studies)
            pages_fetched += 1

            page_token = data.get("nextPageToken")
            if not page_token or len(studies) < self.PAGE_SIZE:
                break

        result = trials[:max_results] if max_results else trials
        return result, saturated

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

        # Temporal holdout: restrict to trials that started before cutoff
        if date_before:
            date_str = date_before.strftime("%Y-%m-%d")
            term = params.get("query.term", "")
            date_filter = f"AREA[StartDate]RANGE[MIN, {date_str}]"
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

    async def _count_trials_total(
        self,
        *,
        drug: str | None = None,
        indication: str | None = None,
        date_before: date | None = None,
        status_filter: str | None = None,
        phase_filter: str | None = None,
    ) -> int:
        """Single-call exact count via CT.gov v2 `countTotal=true&pageSize=1`.

        Used by the new pair-scoped query methods. Free-text condition match
        (no MeSH filtering) — counts are slightly noisier than a MeSH-precise
        walk would be, but the cost is one HTTP call instead of up to 1000
        record fetches, and the noise is acceptable for "how busy is this
        space" framing.
        """
        params = self._build_search_params(
            drug=drug,
            indication=indication,
            date_before=date_before,
            status_filter=status_filter,
            phase_filter=phase_filter,
        )
        params["pageSize"] = 1
        data = await self._rest_get(self.BASE_URL, params)
        return data.get("totalCount", 0)

    # ------------------------------------------------------------------
    # Parsers: v2 API response → Pydantic models
    # ------------------------------------------------------------------

    def _parse_trial(self, study: dict) -> Trial:
        proto = study.get("protocolSection", {})
        derived = study.get("derivedSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        outcomes = proto.get("outcomesModule", {})
        refs = proto.get("referencesModule", {})
        cond_browse = derived.get("conditionBrowseModule", {})

        # MeSH condition terms (derivedSection, not protocolSection)
        mesh_conditions = [
            MeshTerm(id=m.get("id", ""), term=m.get("term", ""))
            for m in cond_browse.get("meshes", [])
        ]

        # MeSH ancestors (broader terms up the MeSH tree)
        mesh_ancestors = [
            MeshTerm(id=m.get("id", ""), term=m.get("term", ""))
            for m in cond_browse.get("ancestors", [])
        ]

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
            mesh_conditions=mesh_conditions,
            mesh_ancestors=mesh_ancestors,
            interventions=interventions,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
            enrollment=enrollment,
            start_date=self._extract_date(status.get("startDateStruct")),
            completion_date=self._extract_date(
                status.get("primaryCompletionDateStruct")
            )
            or self._extract_date(status.get("completionDateStruct")),
            primary_outcomes=primary_outcomes,
            references=pmids,
        )

    # ------------------------------------------------------------------
    # Aggregation: landscape builder
    # ------------------------------------------------------------------

    def _aggregate_landscape(
        self, trials: list[Trial], total_count: int, top_n: int = 50
    ) -> IndicationLandscape:
        """Group trials by sponsor + drug into a competitive landscape.

        Filters to Drug/Biological interventions only; excludes vaccines
        (detected by name keywords — they are not mechanism competitors).
        Ranks by max phase (descending), then most recent start date
        (descending) as tiebreaker. Applies top_n cap after filtering.
        """
        phase_dist: dict[str, int] = {}
        recent_starts: list[RecentStart] = []
        competitors: dict[str, CompetitorEntry] = {}  # key: "sponsor|drug"

        for t in trials:
            # Only count trials with a Drug or Biological intervention
            drug_name, drug_type = self._primary_drug(t)
            if drug_name == "Unknown":
                continue

            # Exclude vaccines — they are not mechanism competitors
            if drug_type == "Biological" and self._is_vaccine(drug_name):
                continue

            # Phase counts (all drug/biologic trials after vaccine filter)
            phase_dist[t.phase] = phase_dist.get(t.phase, 0) + 1

            # Recent starts
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
                    most_recent_start=None,
                )

            entry = competitors[key]
            entry.trial_count += 1
            entry.statuses.add(t.overall_status)
            entry.total_enrollment += t.enrollment or 0

            if self._phase_rank(t.phase) > self._phase_rank(entry.max_phase):
                entry.max_phase = t.phase

            # Track most recent start date for recency tiebreaker
            if t.start_date and (
                entry.most_recent_start is None
                or t.start_date > entry.most_recent_start
            ):
                entry.most_recent_start = t.start_date

        # Rank: highest phase first, then most recent start date (later = better).
        # most_recent_start may be None — treat as earliest possible date so
        # competitors with no start date sort last within a phase tier.
        ranked = sorted(
            competitors.values(),
            key=lambda c: (
                self._phase_rank(c.max_phase),
                c.most_recent_start or "",
            ),
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
    def _is_vaccine(drug_name: str) -> bool:
        """Return True if the drug name matches known vaccine name keywords.

        Used to exclude vaccine biologics from the competitive landscape —
        they are prevention-focused and not mechanism competitors for drugs.
        """
        name_lower = drug_name.lower()
        return any(kw in name_lower for kw in VACCINE_NAME_KEYWORDS)

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
