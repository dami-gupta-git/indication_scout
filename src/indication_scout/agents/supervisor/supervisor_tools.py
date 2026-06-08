"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes one tool: it runs the full sub-agent and
returns its typed output as an artifact plus a short summary string for the LLM. Also a
find_candidates tool that hits Open Targets directly to surface disease candidates for a drug.
"""

import asyncio
import logging
import re
from datetime import date
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.agents.supervisor.candidate_dedup import (
    run_hierarchical_dedup,
)
from indication_scout.config import get_settings
from indication_scout.constants import SUPERVISOR_MIN_PMIDS_NO_TRIALS
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.services.approval_check import (
    get_approved_indications,
    get_fda_approved_disease_mapping,
    list_approved_indications_at,
    list_approved_indications_from_labels,
)
from indication_scout.services.llm import query_llm

logger = logging.getLogger(__name__)

# Fresh-context critic for the supervisor's candidate ranking. Given only the
# blurbs (verdict + evidence + one-line fields), it checks ranking-order
# integrity against the same signals the supervisor prompt defines and returns
# advisory findings — it does not re-rank. Kept terse so it judges the ranking,
# not the prose.
_RANKING_CRITIC_SYSTEM = """\
You audit the ORDER of a ranked list of drug-repurposing candidates. You are given \
the candidates in their current rank order, each with a verdict tag and a few \
one-line fields. Apply these rules:

CLOSED candidates MUST appear AFTER every non-closed candidate. A candidate is \
CLOSED when ANY of these hold:
- its verdict or literature indicates multiple negative randomized trials, or \
authors recommend against further development;
- a safety/efficacy termination of this drug x indication;
- completed Phase 3 of this exact pair with no subsequent regulatory progression \
(unless approved for a related narrower indication).

ADVERSE-SIGNAL candidates (verdict names an adverse/safety signal, or harm reports \
outnumber benefit) rank below clean live candidates but above fully CLOSED ones.

POSITIVE signals (a completed trial with a quantified efficacy readout; strong \
mechanism + active trials + supportive literature with no closing signal) should \
rank ABOVE neutral candidates (active/recruiting trials alone, absence of negative \
signals).

Report ONLY ordering violations — cases where a lower-ranked candidate should \
clearly outrank a higher-ranked one under these rules. Do not rewrite prose or \
question evidence content. If the order is sound, say so.

Output: a short plain-text list of violations, each naming the two candidates and \
the rule, or the single line "Ranking is consistent." if there are none."""


def _log_disease_banner(title: str, diseases: list[str]) -> None:
    """Emit a boxed WARNING-level banner listing diseases, one per line.

    Makes pivotal candidate-list transitions easy to spot in run logs: FDA-dropped, final
    allowlist, mechanism-promoted, and top-N investigation set.
    """
    header = f" {title} (n={len(diseases)}) "
    width = max(len(header) + 4, 60)
    bar = "=" * width
    lines = [bar, header.center(width, "="), bar]
    if diseases:
        for d in diseases:
            lines.append(f"  - {d}")
    else:
        lines.append("  (none)")
    lines.append(bar)
    logger.warning("\n%s", "\n".join(lines))


from indication_scout.agents._trial_formatting import (
    _borda_rank_by_enrollment_and_recency,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.services.retrieval import RetrievalService


def build_supervisor_tools(
    llm,
    svc: RetrievalService,
    db: Session,
    date_before: date | None = None,
) -> tuple[list, "callable", "callable"]:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and reused across calls.

    `date_before` is forwarded to the literature and clinical trials sub-agents so all PubMed and
    ClinicalTrials.gov queries share the same temporal cutoff. The mechanism sub-agent doesn't
    accept it (OpenTargets has no date-filtering API).

    Returns (tools, get_merged_allowlist) where get_merged_allowlist() snapshots the post-merge
    competitor + mechanism disease allowlist (lowercase name → (canonical_name, source)),
    intended to be read after the agent loop has finished.
    """

    # Build sub-agents once at supervisor construction (except literature — see below)
    ct_agent = build_clinical_trials_agent(llm=llm, date_before=date_before)
    mech_agent = build_mechanism_agent(llm=llm)

    # Closure-scoped allowlist — populated by find_candidates and analyze_mechanism, checked by
    # analyze_literature / analyze_clinical_trials.
    # allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[
        str, tuple[str, Literal["competitor", "mechanism", "both"]]
    ] = {}
    # EFO ID → lowercase disease name (key into allowed_diseases). Lets the merge step dedup
    # mechanism candidates against competitor entries by ontology ID even when names differ
    # (e.g. "NSCLC" vs "non-small cell lung cancer").
    allowed_efo_ids: dict[str, str] = {}
    # Raw mechanism candidates as they arrive. analyze_mechanism appends here; find_candidates
    # consumes them in merge_and_dedup() after both seed tools finish. Holding the full list lets
    # the hierarchical LLM pass see the complete competitor + mechanism union in one shot.
    mechanism_candidates_buffer: list = []
    # Drug-level mechanism target list, set by analyze_mechanism and read by merge_and_dedup so
    # the hierarchical LLM pass can reason about the drug's MoA when picking survivors.
    mechanism_targets_for_dedup: list[tuple[str, str]] = []
    # Seed-phase gates. find_candidates and analyze_mechanism run in parallel. analyze_mechanism
    # only buffers raw candidates and sets analyze_mechanism_done; it doesn't touch the allowlist.
    # find_candidates seeds the competitor allowlist, awaits analyze_mechanism_done, then runs
    # merge_and_dedup (centralized exact-match + hierarchical-LLM dedup over the union).
    # find_candidates_done is set inside merge_and_dedup so downstream tools (analyze_literature,
    # analyze_clinical_trials, investigate_top_candidates) only observe the post-dedup allowlist.
    # Both events are set in try/finally so a sub-agent crash doesn't deadlock downstream tools.
    find_candidates_done = asyncio.Event()
    analyze_mechanism_done = asyncio.Event()

    # Drug-level shared store. Populated by sub-agents as they run; surfaced to the supervisor via
    # get_drug_briefing. Keyed by normalized drug name. See supervisor_ideas.md for rationale.
    drug_facts: dict[str, dict] = {}

    # Holdout-only: artifacts produced by investigate_top_candidates. The tool invokes
    # analyze_literature/analyze_clinical_trials directly (not through the LangGraph ReAct loop),
    # so their tool messages don't reach result["messages"]. We stash them here and
    # run_supervisor_agent reads them via get_auto_findings() after the run completes. Keyed by
    # lowercase canonical disease name → {"literature": ..., "clinical_trials": ...}.
    auto_findings: dict[str, dict] = {}

    # Per-disease sub-agent artifacts written by analyze_literature and analyze_clinical_trials as
    # the supervisor runs. Used by finalize_supervisor to enforce the top-N evidence gate (drop
    # blurbs for candidates failing the (0 trials AND <N PMIDs) check). Keyed by lowercase
    # canonical disease name → {"literature": ... | None, "clinical_trials": ... | None}.
    findings_local: dict[str, dict] = {}

    # Ordering gate: finalize_supervisor is rejected until critique_ranking has run this
    # run, so the ranking is always audited before it is committed. The LLM ignores the
    # prompt-level "MANDATORY" instruction on its own, so this enforces it in code.
    critique_state: dict[str, bool] = {"ran": False}

    def _drug_key(drug_name: str) -> str:
        return drug_name.lower().strip()

    def _ensure_drug_entry(drug_name: str) -> dict:
        key = _drug_key(drug_name)
        if key not in drug_facts:
            drug_facts[key] = {
                "drug_name": key,
                "drug_aliases": [],  # ChEMBL trade/generic names
                "approved_indications": [],  # list of indication strings
                "mechanism_targets": [],  # list of (gene, action_type)
                "mechanism_disease_associations": [],  # list of (gene, disease, score)
            }
        return drug_facts[key]

    def _render_briefing(drug_name: str) -> str:
        """Render drug_facts[drug_name] as a markdown briefing."""
        entry = drug_facts.get(_drug_key(drug_name))
        if entry is None:
            return f"DRUG INTAKE: {drug_name}\n- (no facts collected yet)"

        lines = [f"DRUG INTAKE: {entry['drug_name']}"]

        if entry["drug_aliases"]:
            lines.append(f"- Trade/generic names: {', '.join(entry['drug_aliases'])}")
        else:
            lines.append("- Trade/generic names: (not yet resolved)")

        if entry["approved_indications"]:
            lines.append("- FDA-approved indications:")
            for ind in entry["approved_indications"]:
                lines.append(f"  - {ind}")
        else:
            lines.append("- FDA-approved indications: (none discovered in this run)")

        if entry["mechanism_targets"]:
            target_strs = [f"{g} ({a})" for g, a in entry["mechanism_targets"]]
            lines.append(f"- Targets: {', '.join(target_strs)}")
        else:
            lines.append("- Targets: (mechanism agent has not run)")

        if entry["mechanism_disease_associations"]:
            lines.append("- Top mechanism-disease associations:")
            # cap at 10 to keep briefing terse
            for g, d, s in entry["mechanism_disease_associations"][:10]:
                # Hide score when it's the placeholder (not surfaced by
                # MechanismCandidate). Show otherwise.
                score_str = f" (score {s:.2f})" if s > 0 else ""
                lines.append(f"  - {g} → {d}{score_str}")

        return "\n".join(lines)

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Returns the post-dedup candidate list — diseases where competitor drugs (sharing the
        same molecular targets) are being developed, PLUS diseases surfaced by the mechanism
        sub-agent. Waits for analyze_mechanism to finish, then runs merge_and_dedup which
        performs exact-match dedup (ID, name, OT name-resolve) followed by a hierarchical LLM
        pass to collapse super/subtype overlaps. Each line in the content string is tagged
        [competitor], [mechanism], or [both].
        """
        try:
            return await _find_candidates_impl(drug_name)
        finally:
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature
            # / analyze_clinical_trials. They will see an empty allowlist and reject downstream.
            find_candidates_done.set()

    async def _find_candidates_impl(drug_name: str) -> tuple[str, list[str]]:
        drug_name = normalize_drug_name(drug_name)
        chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
        competitors = await svc.get_drug_competitors(chembl_id, date_before=date_before)
        diseases = list(competitors.keys())

        # Drug-level intake: populate the shared store with aliases and any
        # FDA-approved indications discovered during the candidate filter.
        entry = _ensure_drug_entry(drug_name)
        try:
            entry["drug_aliases"] = await get_all_drug_names(chembl_id, svc.cache_dir)
        except Exception as e:
            logger.warning(
                "find_candidates: get_all_drug_names failed for %s: %s", chembl_id, e
            )

        # Seed approved_indications from the drug's own FDA label, independent of any candidate
        # list. Without this, an approved indication absent from OpenTargets competitor diseases
        # (e.g. semaglutide × MASH) never reaches the briefing and the supervisor can't reason
        # about subset/superset relationships against it.
        #
        # When date_before is set, swap the live openFDA path for the hardcoded approvals table —
        # the live path leaks today's approvals into a holdout. Drugs not in the table return []
        # and approval reasoning is silently disabled for that holdout run (see PLAN_date_before.md).
        seed_aliases = entry["drug_aliases"] or [drug_name]
        try:
            if date_before is not None:
                seeded = list_approved_indications_at(drug_name, date_before)
            else:
                async with FDAClient(cache_dir=svc.cache_dir) as fda_client:
                    label_texts = await fda_client.get_all_label_indications(
                        seed_aliases
                    )
                seeded = await list_approved_indications_from_labels(
                    label_texts=label_texts,
                    cache_dir=svc.cache_dir,
                )
            if seeded:
                existing = {
                    ind.lower().strip() for ind in entry["approved_indications"]
                }
                for ind in seeded:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
                # logger.warning(
                #     "[TOOL] find_candidates seeded %d approved indication(s) from %s: %s",
                #     len(seeded),
                #     "hardcoded table" if date_before is not None else "label",
                #     seeded,
                # )
        except Exception as e:
            logger.warning(
                "find_candidates: approved-indication seed failed for %s: %s",
                drug_name,
                e,
            )

        # Drop competitor diseases already approved for this drug. Same swap as above: hardcoded
        # table when date_before is set, live FDA otherwise.
        fda_approved_lower: set[str] = set()
        if diseases:
            if date_before is not None:
                fda_approved = get_approved_indications(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    as_of=date_before,
                )
            else:
                mapping = await get_fda_approved_disease_mapping(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    cache_dir=svc.cache_dir,
                )
                fda_approved = {
                    disease for disease, is_approved in mapping.items() if is_approved
                }
            if fda_approved:
                _log_disease_banner(
                    f"FDA-DROPPED (already approved for {drug_name}, source: "
                    f"{'hardcoded table' if date_before is not None else 'live FDA'})",
                    sorted(fda_approved),
                )
                # Record the approved indications in the shared store. Discovered as a side
                # effect of candidate filtering — even though dropped from the candidate list,
                # the supervisor needs them to reason about subset/superset relationships
                # (e.g. CML approval makes "myeloid leukemia" candidate ambiguous).
                existing = {
                    ind.lower().strip() for ind in entry["approved_indications"]
                }
                for ind in fda_approved:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
            fda_approved_lower = {d.lower().strip() for d in fda_approved}

        diseases = [d for d in diseases if d.lower().strip() not in fda_approved_lower]

        # Cap applies to competitor entries only. Mechanism-promoted entries are appended on top
        # after the merge below — they're already small (capped upstream by
        # MECHANISM_TOP_CANDIDATES) and dropping them here would defeat the purpose of the merge.
        candidate_cap = get_settings().supervisor_candidate_cap
        if len(diseases) > candidate_cap:
            logger.warning(
                "[TOOL] find_candidates capping competitor candidates from %d to %d "
                "(SUPERVISOR_CANDIDATE_CAP)",
                len(diseases),
                candidate_cap,
            )
            diseases = diseases[:candidate_cap]

        logger.warning(f"{diseases=}")
        allowed_diseases.clear()
        allowed_efo_ids.clear()
        for d in diseases:
            allowed_diseases[d.lower().strip()] = (d, "competitor")

        # Pull EFO IDs for the competitor allowlist from the raw OT cache. Used to dedup mechanism
        # candidates against competitor entries by ontology ID. Names that don't resolve to an EFO
        # (e.g. renamed by the LLM merge step) get no entry — dedup falls back to name match.
        async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
            raw = await ot_client.get_drug_competitors(
                chembl_id, date_before=date_before
            )
        for disease_lower in allowed_diseases:
            efo_id = raw["disease_efo_ids"].get(disease_lower)
            if efo_id:
                allowed_efo_ids[efo_id] = disease_lower

        _log_disease_banner(
            f"CANDIDATE ALLOWLIST for {drug_name} ({chembl_id}) — competitor source",
            diseases,
        )

        # Wait for analyze_mechanism to populate mechanism_candidates_buffer. The mechanism
        # sub-agent's finally block always sets this gate, so a crash just yields an empty
        # mechanism contribution rather than deadlocking.
        await analyze_mechanism_done.wait()

        # Run the centralized merge + dedup pipeline. find_candidates_done is set inside
        # merge_and_dedup's finally so downstream readers (analyze_literature,
        # analyze_clinical_trials, investigate_top_candidates) only see the post-dedup allowlist.
        await merge_and_dedup(drug_name)

        # Snapshot the post-dedup allowlist in dict insertion order: competitor entries first (OT
        # ranked order), then mechanism-only entries (analyze_mechanism's order); hierarchical
        # dedup may have removed entries from either. "both" entries stay in their competitor slot.
        merged: list[tuple[str, str]] = [
            (canonical, source) for (canonical, source) in allowed_diseases.values()
        ]
        n_competitor = sum(1 for _, s in merged if s == "competitor")
        n_both = sum(1 for _, s in merged if s == "both")
        n_mechanism = sum(1 for _, s in merged if s == "mechanism")

        lines = [
            f"Found {len(merged)} candidate diseases for {drug_name} ({chembl_id}) — "
            f"{n_competitor} competitor, {n_both} both, {n_mechanism} mechanism-only:"
        ]
        for i, (canonical, source) in enumerate(merged, start=1):
            lines.append(f"  {i}. {canonical} [{source}]")
        content = "\n".join(lines)

        merged_names = [canonical for canonical, _ in merged]
        return content, merged_names

    async def merge_and_dedup(drug_name: str) -> None:
        """Merge buffered mechanism candidates into the competitor-seeded allowlist.

        Pipeline (single chokepoint for all candidate-disease deduplication):
          1. Exact ID match — drop mechanism candidate if disease_id already in allowed_efo_ids;
             upgrade matched competitor entry's source to "both".
          2. Exact name match — drop if lowercased name already in allowed_diseases; upgrade to
             "both".
          3. OT name-resolve — resolve unresolved mechanism candidate names to EFO IDs; retry
             step 1 against allowed_efo_ids.
          4. Hierarchical LLM pass — over the full merged list, identify super/subtype overlaps
             the exact-match passes can't catch (UC ⊂ IBD, T2DM ⊂ DM); pick one survivor each.

        Sets find_candidates_done before returning so downstream readers (analyze_literature,
        analyze_clinical_trials, investigate_top_candidates) observe the post-dedup allowlist. The
        find_candidates wrapper's try/finally still sets the gate on any exception path.
        """
        try:
            await _merge_and_dedup_impl(drug_name)
        finally:
            find_candidates_done.set()

    async def _merge_and_dedup_impl(drug_name: str) -> None:
        # Steps 1-3: exact-match dedup against the competitor-seeded allowlist.
        promoted: list[str] = []
        if mechanism_candidates_buffer:
            async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
                for candidate in mechanism_candidates_buffer:
                    key = candidate.disease_name.lower().strip()
                    if not key:
                        continue

                    existing_key: str | None = None
                    if candidate.disease_id and candidate.disease_id in allowed_efo_ids:
                        existing_key = allowed_efo_ids[candidate.disease_id]
                    elif key in allowed_diseases:
                        existing_key = key
                    else:
                        resolved_id = await ot_client.resolve_disease_id(
                            candidate.disease_name
                        )
                        if resolved_id and resolved_id in allowed_efo_ids:
                            existing_key = allowed_efo_ids[resolved_id]

                    if existing_key is not None:
                        existing_name, source = allowed_diseases[existing_key]
                        if source == "competitor":
                            allowed_diseases[existing_key] = (existing_name, "both")
                        if (
                            candidate.disease_id
                            and candidate.disease_id not in allowed_efo_ids
                        ):
                            allowed_efo_ids[candidate.disease_id] = existing_key
                    else:
                        allowed_diseases[key] = (candidate.disease_name, "mechanism")
                        if candidate.disease_id:
                            allowed_efo_ids[candidate.disease_id] = key
                        promoted.append(candidate.disease_name)

        if promoted:
            _log_disease_banner(
                f"MECHANISM-PROMOTED candidates added to allowlist for {drug_name}",
                promoted,
            )

        # Step 4: hierarchical LLM pass — temporarily disabled while we revisit
        # the case that motivated it. The dedup was collapsing actionable
        # subtype candidates (e.g. PCOS, gestational diabetes) into broad
        # parents (metabolic disease) for broadly-acting drugs like metformin.
        # Keep all candidates from the exact-match dedup until we decide on a
        # hardcoded equivalence-group approach.
        # if len(allowed_diseases) < 2:
        #     return
        #
        # # Build the (name, source, efo_id) tuple list in insertion order. For each
        # # row we need the EFO ID, which is stored inverted in allowed_efo_ids.
        # name_to_efo: dict[str, str] = {}
        # for efo_id, lc_name in allowed_efo_ids.items():
        #     if lc_name in allowed_diseases:
        #         name_to_efo[lc_name] = efo_id
        #
        # candidate_tuples: list[tuple[str, str, str | None]] = []
        # for lc_name, (canonical, source) in allowed_diseases.items():
        #     candidate_tuples.append((canonical, source, name_to_efo.get(lc_name)))
        #
        # decisions = await run_hierarchical_dedup(
        #     drug_name=drug_name,
        #     mechanism_targets=list(mechanism_targets_for_dedup),
        #     candidates=candidate_tuples,
        # )
        # if not decisions.decisions:
        #     return
        #
        # # Apply decisions: remove dropped entries from allowed_diseases and
        # # allowed_efo_ids. A name appearing in multiple decisions is removed once.
        # canonical_to_lc: dict[str, str] = {
        #     canonical: lc for lc, (canonical, _) in allowed_diseases.items()
        # }
        # removed_canonicals: list[str] = []
        # removed_lc: set[str] = set()
        # for decision in decisions.decisions:
        #     for dropped_name in decision.dropped:
        #         lc = canonical_to_lc.get(dropped_name)
        #         if lc is None or lc in removed_lc:
        #             continue
        #         removed_lc.add(lc)
        #         removed_canonicals.append(dropped_name)
        #         logger.warning(
        #             "[DEDUP] %r → %r (%s)",
        #             dropped_name,
        #             decision.survivor,
        #             decision.reason or "no reason given",
        #         )
        #         allowed_diseases.pop(lc, None)
        #         # Invert lookup: drop any allowed_efo_ids row that points at this lc.
        #         for efo_id in [e for e, v in allowed_efo_ids.items() if v == lc]:
        #             allowed_efo_ids.pop(efo_id, None)
        #
        # if removed_canonicals:
        #     _log_disease_banner(
        #         f"HIERARCHICAL DEDUP removed candidates for {drug_name}",
        #         removed_canonicals,
        #     )
        #     survivors = [canonical for canonical, _ in allowed_diseases.values()]
        #     _log_disease_banner(
        #         f"CANDIDATE ALLOWLIST for {drug_name} after hierarchical dedup",
        #         survivors,
        #     )

    def _reject(disease_name: str, tool_label: str, empty_output):
        valid = sorted(allowed_diseases.keys())
        msg = (
            f"REJECTED: '{disease_name}' is not in the allowed candidate list. "
            f"You must call {tool_label} only with a disease_name returned VERBATIM by "
            f"find_candidates or added from mechanism associations. "
            f"Do not reword, substitute synonyms, or introduce diseases from training knowledge. "
            f"Valid candidates: {valid}"
        )
        # logger.warning("[TOOL] %s REJECTED disease=%r", tool_label, disease_name)
        return msg, empty_output

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks abstracts, and produces a
        structured evidence summary with strength rating (none / weak / moderate / strong).
        """
        # Wait for both seed tools to finish populating the allowlist. Without this, parallel
        # tool calls can hit analyze_literature before find_candidates / analyze_mechanism have
        # merged their candidates, causing legitimate diseases to be rejected.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        # Build a fresh agent per call so the closure-scoped store dict in literature_tools is not
        # shared across disease invocations.
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_literature", LiteratureOutput())

        # logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

        lit_agent = build_literature_agent(
            llm=llm, svc=svc, db=db, date_before=date_before
        )
        logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

        output = await run_literature_agent(lit_agent, drug_name, disease_name)
        strength = (
            output.evidence_summary.strength if output.evidence_summary else "no data"
        )
        header = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}."
        )
        # Build supervisor-facing summary deterministically from EvidenceSummary so adverse-signal
        # language reaches the supervisor verbatim with no LLM rewrite in between.
        synth_summary = (
            output.evidence_summary.summary if output.evidence_summary else ""
        )
        key_findings = (
            output.evidence_summary.key_findings if output.evidence_summary else []
        )
        parts = [header]
        if synth_summary:
            parts.append(synth_summary)
        if key_findings:
            parts.append(
                "Key findings:\n" + "\n".join(f"- {f}" for f in key_findings)
            )
        summary = "\n\n".join(parts)
        # Write-through for the top-N evidence gate in finalize_supervisor.
        slot = findings_local.setdefault(
            disease_name.lower().strip(), {"literature": None, "clinical_trials": None}
        )
        slot["literature"] = output
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_clinical_trials(
        drug_name: str, disease_name: str
    ) -> tuple[str, ClinicalTrialsOutput]:
        """Run a full clinical trials analysis for a drug-disease pair.

        Checks ClinicalTrials.gov for existing trials, competitive landscape, and terminated
        trials (safety/efficacy red flags).
        """
        # Wait for both seed tools to finish populating the allowlist (see analyze_literature).
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(
                disease_name, "analyze_clinical_trials", ClinicalTrialsOutput()
            )

        # logger.warning("[TOOL] analyze_clinical_trials(drug=%r, disease=%r)", drug_name, disease_name)
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)

        # Drug-level write-through: when the FDA check matches the candidate
        # against an approved indication, capture it in the supervisor's
        # briefing so subsequent reasoning sees the approval status. Trial
        # data still flows through to the summary below — the sub-agent
        # always investigates fully now (no short-circuits).
        approval = output.approval
        if approval is not None and approval.is_approved:
            entry = _ensure_drug_entry(drug_name)
            matched = approval.matched_indication or disease_name
            existing = {ind.lower().strip() for ind in entry["approved_indications"]}
            if matched.lower().strip() not in existing:
                entry["approved_indications"].append(matched)

        # Normal path: counts come from the new exact-count tools (countTotal
        # API). Each scope owns its own count; no cross-scope summing.
        search = output.search
        completed = output.completed
        terminated = output.terminated

        n_total = search.total_count if search else 0
        n_recruiting = search.by_status.get("RECRUITING", 0) if search else 0
        n_active_not_recruiting = (
            search.by_status.get("ACTIVE_NOT_RECRUITING", 0) if search else 0
        )
        n_withdrawn = search.by_status.get("WITHDRAWN", 0) if search else 0
        n_completed = completed.total_count if completed else 0
        n_terminated = terminated.total_count if terminated else 0
        # safety/efficacy classification is computed from the top-50 shown
        # terminated trials; if total_count > len(trials) this is a floor.
        n_safety_efficacy_shown = (
            sum(
                1
                for t in terminated.trials
                if _classify_stop_reason(t.why_stopped) in {"safety", "efficacy"}
            )
            if terminated
            else 0
        )
        header = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_total} total ({n_recruiting} recruiting, "
            f"{n_active_not_recruiting} active, {n_withdrawn} withdrawn). "
            f"{n_completed} completed. "
            f"{n_terminated} terminated "
            f"({n_safety_efficacy_shown} safety/efficacy in shown set)."
        )

        completed_trials = completed.trials if completed else []
        terminated_trials = terminated.trials if terminated else []

        completed_phase_dist = _phase_distribution(completed_trials)
        terminated_phase_dist = _phase_distribution(terminated_trials)

        completed_top = _borda_rank_by_enrollment_and_recency(completed_trials, k=10)
        terminated_top = _borda_rank_by_enrollment_and_recency(terminated_trials, k=10)

        completed_table = _format_trial_table(
            completed_top,
            columns=(
                "nct_id",
                "phase",
                "start_date",
                "completion_date",
                "refs",
                "mesh",
                "title",
            ),
            cap=10,
        )
        terminated_table = _format_trial_table(
            terminated_top,
            columns=(
                "nct_id",
                "phase",
                "stop_reason",
                "start_date",
                "completion_date",
                "refs",
                "mesh",
                "title",
            ),
            cap=10,
            include_why_stopped=True,
            stop_classifier=_classify_stop_reason,
        )

        structured = (
            f"\n\nPhase distribution (completed): {completed_phase_dist}\n"
            f"Phase distribution (terminated): {terminated_phase_dist}\n\n"
            f"Completed trials (top 10 by enrollment + recency):\n"
            f"{completed_table}\n\n"
            f"Terminated trials (top 10 by enrollment + recency):\n"
            f"{terminated_table}"
        )

        sub_agent_summary = output.summary or ""
        summary = f"{header}{structured}"
        if sub_agent_summary:
            summary = f"{summary}\n\n{sub_agent_summary}"
        # Write-through for the top-N evidence gate in finalize_supervisor.
        slot = findings_local.setdefault(
            disease_name.lower().strip(), {"literature": None, "clinical_trials": None}
        )
        slot["clinical_trials"] = output
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Run the mechanism sub-agent for a drug.

        The mechanism agent returns target-level MoA data and the agent's narrative summary.
        Raw mechanism candidates are buffered for the centralized merge_and_dedup pass that
        find_candidates runs once both seed tools have completed.
        """
        try:
            return await _analyze_mechanism_impl(drug_name)
        finally:
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature
            # / analyze_clinical_trials.
            analyze_mechanism_done.set()

    async def _analyze_mechanism_impl(drug_name: str) -> tuple[str, MechanismOutput]:
        drug_name = normalize_drug_name(drug_name)
        output = await run_mechanism_agent(
            mech_agent, drug_name, date_before=date_before
        )
        logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)

        # Buffer raw mechanism candidates for find_candidates to consume in merge_and_dedup()
        # after both seed tools finish. Centralizing the merge lets the full competitor +
        # mechanism union pass through a single hierarchical-dedup pass.
        mechanism_candidates_buffer.clear()
        raw_received: list[str] = []
        for candidate in output.candidates:
            if not (candidate.disease_name or "").strip():
                continue
            mechanism_candidates_buffer.append(candidate)
            raw_received.append(candidate.disease_name)

        if raw_received:
            _log_disease_banner(
                f"MECHANISM-RAW candidates received for {drug_name}",
                raw_received,
            )

        # Drug-level write-through: populate mechanism_targets and mechanism_disease_associations
        # in the shared store. Captured per-MoA so the briefing can show
        # "ABL1 (INHIBITOR), KIT (INHIBITOR)".
        entry = _ensure_drug_entry(drug_name)
        target_pairs: list[tuple[str, str]] = []
        seen_target_pairs: set[tuple[str, str]] = set()
        for moa in output.mechanisms_of_action:
            for sym in moa.target_symbols:
                pair = (sym, moa.action_type or "UNKNOWN")
                if pair not in seen_target_pairs:
                    seen_target_pairs.add(pair)
                    target_pairs.append(pair)
        entry["mechanism_targets"] = target_pairs

        # Mechanism candidates already carry the high-score target→disease associations the agent
        # surfaced. We don't have the raw scores on the candidate model — record the pair without
        # a score for now.
        assocs: list[tuple[str, str, float]] = []
        seen_assoc_pairs: set[tuple[str, str]] = set()
        for cand in output.candidates:
            pair_key = (cand.target_symbol, cand.disease_name)
            if pair_key in seen_assoc_pairs:
                continue
            seen_assoc_pairs.add(pair_key)
            # Score not surfaced on MechanismCandidate; use 0.0 as a placeholder. The supervisor
            # only needs to know "this gene is associated with this disease per OT mechanism
            # evidence" — the briefing renderer hides the score if it's the placeholder.
            assocs.append((cand.target_symbol, cand.disease_name, 0.0))
        entry["mechanism_disease_associations"] = assocs

        # Record the mechanism target list so merge_and_dedup() can pass it to the hierarchical
        # LLM pass as context for survivor selection.
        mechanism_targets_for_dedup.clear()
        mechanism_targets_for_dedup.extend(target_pairs)

        n_targets = len(output.drug_targets)
        header = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets, "
            f"{len(output.candidates)} mechanism candidates "
            f"(merge into allowlist deferred to find_candidates)."
        )
        sub_agent_summary = output.summary or ""
        summary = (
            f"{header}\n\n{sub_agent_summary}".strip() if sub_agent_summary else header
        )
        return summary, output

    @tool
    def get_drug_briefing(drug_name: str) -> str:
        """Return the accumulated drug-level briefing for this drug.

        Read-only view of facts collected by find_candidates, analyze_mechanism,
        and analyze_clinical_trials during this run: ChEMBL aliases, FDA-approved
        indications discovered, mechanism targets, and mechanism disease
        associations. Call this before finalize_supervisor to check whether any
        candidate is related to an approved indication (subset/superset/sibling).
        """
        drug_name = normalize_drug_name(drug_name)
        return _render_briefing(drug_name)

    # Holdout-only tool: bulk-investigate the top-N candidates with no LLM discretion. The probe
    # (scripts/probe_supervisor_t2dm.py) showed the supervisor LLM systematically skips "obvious"
    # candidates like T2DM for semaglutide regardless of prompt instructions — exactly the
    # candidate the holdout is testing, so we remove the LLM's ability to skip by auto-investigating
    # the top-10.
    HOLDOUT_INVESTIGATION_CAP = 10
    # Non-holdout fan-out cap. Mirrors the prompt's "investigate up to a maximum of 6" so the
    # parallel path investigates the same candidate set the serial path would.
    FANOUT_INVESTIGATION_CAP = 6

    @tool(response_format="content_and_artifact")
    async def investigate_top_candidates(
        drug_name: str,
    ) -> tuple[str, list[dict]]:
        """[HOLDOUT MODE ONLY] Auto-investigate the top-10 candidates from the merged allowlist.

        Runs analyze_literature AND analyze_clinical_trials in parallel for the top 10
        candidates by mechanism+competitor strength. Removes the LLM's ability to skip
        "obvious" candidates that holdout-mode evaluations specifically need to recover.

        Call this ONCE, after find_candidates and analyze_mechanism complete. After this
        returns, you may still call analyze_literature / analyze_clinical_trials for
        candidates beyond the top 10 if you want.
        """
        # Wait for both seed tools to finish populating the allowlist.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)

        # Top-N from the merged allowlist. Insertion order preserves find_candidates's competitor
        # ranking, with mechanism-promoted entries appended in analyze_mechanism's order. Holdout
        # uses a wider cap (recover obvious candidates); non-holdout fan-out matches the serial
        # path's per-run investigation ceiling (the prompt's "up to 6") so output parity holds.
        # supervisor_candidate_cap is NOT used here — it only trims the final ranked list, not how
        # many diseases get investigated.
        cap = HOLDOUT_INVESTIGATION_CAP if date_before is not None else FANOUT_INVESTIGATION_CAP
        top_n = list(allowed_diseases.items())[:cap]
        if not top_n:
            return "No candidates in allowlist; nothing to investigate.", []

        canonical_diseases = [canonical for _, (canonical, _) in top_n]
        _log_disease_banner(
            f"INVESTIGATING top candidates for {drug_name} (lit + trials in parallel)",
            canonical_diseases,
        )

        # Fan out: analyze_literature + analyze_clinical_trials in parallel. Pass a ToolCall-shaped
        # dict (not a plain args dict) so .ainvoke() returns a ToolMessage with .artifact
        # populated. A plain dict input returns just the content string and loses the typed artifact.
        async def _invest(disease: str) -> tuple[str, dict]:
            disease_slug = disease.lower().replace(" ", "_")
            lit_call = analyze_literature.ainvoke(
                {
                    "name": "analyze_literature",
                    "args": {"drug_name": drug_name, "disease_name": disease},
                    "id": f"auto_lit_{disease_slug}",
                    "type": "tool_call",
                }
            )
            ct_call = analyze_clinical_trials.ainvoke(
                {
                    "name": "analyze_clinical_trials",
                    "args": {"drug_name": drug_name, "disease_name": disease},
                    "id": f"auto_ct_{disease_slug}",
                    "type": "tool_call",
                }
            )
            lit_msg, ct_msg = await asyncio.gather(lit_call, ct_call)

            lit_artifact = lit_msg.artifact
            ct_artifact = ct_msg.artifact
            # Stash artifacts in the closure so run_supervisor_agent can merge them into the
            # SupervisorOutput. The LangGraph ReAct loop doesn't see these tool messages because
            # they were invoked directly, not through the agent.
            auto_findings[disease.lower().strip()] = {
                "literature": lit_artifact,
                "clinical_trials": ct_artifact,
            }
            strength = (
                lit_artifact.evidence_summary.strength
                if lit_artifact and lit_artifact.evidence_summary
                else "no data"
            )
            n_pmids = len(lit_artifact.pmids) if lit_artifact else 0
            n_total = (
                ct_artifact.search.total_count
                if ct_artifact and ct_artifact.search
                else 0
            )
            n_completed = (
                ct_artifact.completed.total_count
                if ct_artifact and ct_artifact.completed
                else 0
            )
            n_terminated = (
                ct_artifact.terminated.total_count
                if ct_artifact and ct_artifact.terminated
                else 0
            )
            return disease, {
                "disease": disease,
                "literature_strength": strength,
                "literature_pmids": n_pmids,
                "trials_total": n_total,
                "trials_completed": n_completed,
                "trials_terminated": n_terminated,
            }

        results = await asyncio.gather(*(_invest(d) for d in canonical_diseases))
        artifacts = [a for _, a in results]

        # One-line-per-disease compact summary the LLM can rank against.
        lines = [
            f"Auto-investigated {len(artifacts)} top candidates " f"for {drug_name}:"
        ]
        for a in artifacts:
            lines.append(
                f"  - {a['disease']}: literature {a['literature_strength']}, "
                f"{a['literature_pmids']} PMIDs; trials {a['trials_total']} total, "
                f"{a['trials_completed']} completed, {a['trials_terminated']} terminated"
            )
        return "\n".join(lines), artifacts

    def _reconstruct_holdout_summary() -> str:
        """Build the holdout summary deterministically from findings_local.

        Holdout mode renders a structured fact list — no LLM prose. Each investigated candidate
        becomes one ranked line with literature strength, PMID count, and trial counts pulled
        from the typed sub-agent artifacts. Candidates with zero trials and no usable literature
        signal (same gate finalize_supervisor applies to blurbs) drop into a single
        "Evidence gate exclusions:" footer line.
        """
        strength_rank = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}

        ranked: list[tuple] = []
        excluded: list[str] = []
        for disease_lower, slot in findings_local.items():
            allow = allowed_diseases.get(disease_lower)
            if allow is None:
                continue
            canonical = allow[0]
            lit = slot.get("literature")
            ct = slot.get("clinical_trials")
            n_pmids = len(lit.pmids) if lit else 0
            lit_strength = (
                lit.evidence_summary.strength
                if lit and lit.evidence_summary
                else "none"
            )
            lit_study_count = (
                lit.evidence_summary.study_count
                if lit and lit.evidence_summary
                else None
            )
            total_trials = (
                ct.search.total_count
                if (ct is not None and ct.search is not None)
                else 0
            )
            completed_trials = (
                ct.completed.total_count
                if (ct is not None and ct.completed is not None)
                else 0
            )
            terminated_trials = (
                ct.terminated.total_count
                if (ct is not None and ct.terminated is not None)
                else 0
            )

            no_lit_signal = (
                lit_strength == "none"
                or lit_study_count == 0
                or (
                    lit_strength is None
                    and lit_study_count is None
                    and n_pmids < SUPERVISOR_MIN_PMIDS_NO_TRIALS
                )
            )
            if total_trials == 0 and no_lit_signal:
                excluded.append(canonical)
                continue

            ranked.append(
                (
                    strength_rank.get(lit_strength or "none", 0),
                    total_trials,
                    n_pmids,
                    canonical.lower(),
                    canonical,
                    lit_strength or "none",
                    n_pmids,
                    total_trials,
                    completed_trials,
                    terminated_trials,
                )
            )

        ranked.sort(
            key=lambda r: (-r[0], -r[1], -r[2], r[3]),
        )

        lines: list[str] = []
        for i, row in enumerate(ranked, start=1):
            _, _, _, _, canonical, strength, n_pmids, total, completed, terminated = row
            lines.append(
                f"{i}. {canonical} — literature: {strength}, {n_pmids} PMIDs; "
                f"trials: {total} total, {completed} completed, "
                f"{terminated} terminated."
            )
        if excluded:
            excluded.sort(key=lambda s: s.lower())
            lines.append("")
            lines.append("Evidence gate exclusions: " + ", ".join(excluded) + ".")
        return "\n".join(lines)

    @tool
    async def critique_ranking(blurbs: list[dict] | None = None) -> str:
        """Audit your candidate ranking ORDER before finalizing.

        Call this AFTER you have drafted your ranked blurbs but BEFORE
        finalize_supervisor. Pass the same `blurbs` list (in your intended rank
        order) you plan to finalize with. A separate reviewer checks the order
        against the RANKING SIGNALS — specifically that CLOSED and adverse-signal
        candidates do not outrank live ones — and returns any ordering
        violations it finds. It does NOT re-rank for you; reorder the blurbs
        yourself to address the findings, then finalize. If it reports the
        ranking is consistent, finalize as-is.

        Arguments:
        - blurbs: your draft ranked blurbs, same shape as finalize_supervisor.
        """
        critique_state["ran"] = True
        items = blurbs or []
        if not items:
            return "No blurbs provided — nothing to critique."
        lines: list[str] = []
        for i, item in enumerate(items, start=1):
            disease = (item.get("disease") or "").strip() or "(unnamed)"
            verdict = (item.get("verdict") or "").strip()
            literature = (item.get("literature") or "").strip()
            blocker = (item.get("blocker") or "").strip()
            lines.append(
                f"{i}. {disease} | verdict: {verdict or '—'} | "
                f"literature: {literature or '—'} | blocker: {blocker or '—'}"
            )
        prompt = "Current ranking (top to bottom):\n" + "\n".join(lines)
        critique = await query_llm(prompt, system=_RANKING_CRITIC_SYSTEM)
        result = critique.strip()
        logger.info(
            "[TOOL] critique_ranking IN (%d candidates):\n%s", len(items), "\n".join(lines)
        )
        logger.info("[TOOL] critique_ranking OUT:\n%s", result)
        return result

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(
        summary: str, blurbs: list[dict] | None = None
    ) -> tuple[str, dict]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step. This terminates the agent loop.

        Arguments:
        - summary: your ranked structured fact list of investigated candidates
          (see WRITING THE SUMMARY in the system prompt).
        - blurbs: a list of structured per-candidate entries for the TOP 5
          ranked candidates in your summary, in rank order. Each entry is a
          dict with these keys:
            - disease: <verbatim candidate name from find_candidates or
              analyze_mechanism>
            - stage: where the pair sits in development (single line)
            - literature: one-line summary of the literature evidence base —
              strength + shape, e.g. "Strong, 5 RCTs / meta-analyses",
              "Weak, case reports only", "None"
            - blocker: what is currently holding the program back, or empty
              if nothing is
            - active_programs: short summary of what is still moving
            - key_risk: the single biggest risk to the hypothesis
            - verdict: short interpretive tag (e.g. "Live but bottlenecked")
            - watch: next concrete data readout or trial worth watching
              (NCT id and/or expected timing); empty if none on record
            - prose: exactly 2 sentences of interpretive synthesis
          Each blurb must synthesize ONLY the literature and clinical_trials
          sub-agent summaries you saw for that disease this run — do not
          include mechanism content. Pass an empty list if no candidates were
          investigated. Diseases not in the allowlist are dropped. An entry
          with empty disease, or empty in BOTH prose AND every structured
          field, is dropped.
        """
        # Ordering gate: refuse to finalize until the ranking has been audited by
        # critique_ranking this run. Returns a non-terminal instruction so the agent
        # loop calls critique_ranking and then retries finalize. (Reject path returns
        # an empty artifact dict so the content_and_artifact contract holds.)
        if not critique_state["ran"]:
            logger.warning(
                "[TOOL] finalize_supervisor rejected — critique_ranking not called yet"
            )
            return (
                "Cannot finalize yet: you must call critique_ranking exactly once with "
                "your draft blurbs (in rank order) BEFORE finalize_supervisor. Call "
                "critique_ranking now, address any ordering violations it reports, then "
                "call finalize_supervisor again.",
                {},
            )
        logger.info("[TOOL] finalize_supervisor called with %d blurbs", len(blurbs or []))
        validated: list[dict] = []
        structured_keys = (
            "stage",
            "literature",
            "blocker",
            "active_programs",
            "key_risk",
            "verdict",
            "watch",
        )
        for item in blurbs or []:
            disease = (item.get("disease") or "").strip()
            if not disease:
                continue
            fields = {k: (item.get(k) or "").strip() for k in structured_keys}
            prose = (item.get("prose") or "").strip()
            if not prose and not any(fields.values()):
                continue
            disease_key = disease.lower().strip()
            if disease_key not in allowed_diseases:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r "
                    "(not in allowlist)",
                    disease,
                )
                continue
            # Top-N evidence gate: drop candidates where 0 trials AND synthesize indicates no
            # usable literature signal — strength="none" OR study_count==0 (OR both).
            # Strong/moderate/weak literature with at least one relevant abstract and 0 trials is
            # a legitimate repurposing signal and is kept. PMID-count fallback applies only when
            # synthesize didn't run. Investigated-but-filtered candidates still appear in
            # disease_findings (per-disease section); they're only removed from top-N + blurbs.
            slot = findings_local.get(disease_key) or {}
            lit = slot.get("literature")
            ct = slot.get("clinical_trials")
            n_pmids = len(lit.pmids) if lit else 0
            n_trials = (
                ct.search.total_count if (ct is not None and ct.search is not None)
                else 0
            )
            lit_strength = (
                lit.evidence_summary.strength
                if lit and lit.evidence_summary
                else None
            )
            lit_study_count = (
                lit.evidence_summary.study_count
                if lit and lit.evidence_summary
                else None
            )
            no_lit_signal = (
                lit_strength == "none"
                or lit_study_count == 0
                or (
                    lit_strength is None
                    and lit_study_count is None
                    and n_pmids < SUPERVISOR_MIN_PMIDS_NO_TRIALS
                )
            )
            if n_trials == 0 and no_lit_signal:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r "
                    "(evidence gate: %d trials, %d PMIDs, strength=%s, "
                    "study_count=%s)",
                    disease,
                    n_trials,
                    n_pmids,
                    lit_strength,
                    lit_study_count,
                )
                continue
            entry = {"disease": disease, "prose": prose, **fields}
            validated.append(entry)

        if date_before is not None:
            # Holdout mode: ignore the LLM's summary string and rebuild it deterministically from
            # the typed artifacts in findings_local. The LLM drifts to narrative prose under the
            # # APPROVAL RELATIONSHIPS instructions; reconstruction enforces the structured
            # fact-list contract documented in supervisor_holdout.txt.
            filtered_summary = _reconstruct_holdout_summary()
            artifact = {"summary": filtered_summary, "blurbs": []}
            return "Supervisor analysis complete.", artifact

        # Filter the LLM-written summary to drop ranked lines whose disease didn't pass the
        # evidence gate (not in validated). Non-ranked lines (e.g. trailing "Closed signals:")
        # pass through unchanged. Surviving lines are renumbered to stay contiguous. Uses the same
        # regex and longest-substring match strategy as the report formatter's
        # _splice_blurbs_into_summary so disease matching is consistent.
        validated_diseases = {e["disease"].lower().strip() for e in validated}
        rank_line = re.compile(r"^\s*(?P<rank>\d+)\.\s+(?P<head>.+?)\s+—\s+(?P<tail>.+)$")
        # Normalize the heading line so the report uses "signals" instead of
        # "candidates" / "opportunities" / "indications" regardless of what the LLM wrote.
        heading_line = re.compile(
            r"^\s*Ranked\s+repurposing\s+"
            r"(?:candidates|opportunities|indications|signals)"
            r"(?P<rest>\s+for\s+.+?:?\s*)$",
            re.IGNORECASE,
        )
        filtered_lines: list[str] = []
        next_rank = 1
        for line in summary.splitlines():
            heading_match = heading_line.match(line)
            if heading_match is not None:
                rest = heading_match.group("rest").rstrip()
                if not rest.endswith(":"):
                    rest = rest + ":"
                filtered_lines.append(f"Ranked repurposing signals{rest}")
                continue
            m = rank_line.match(line)
            if m is None:
                filtered_lines.append(line)
                continue
            head_lower = m.group("head").lower()
            keep = any(
                d and d in head_lower
                for d in sorted(validated_diseases, key=len, reverse=True)
            )
            if not keep:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping summary line for "
                    "disease=%r (failed evidence gate)",
                    m.group("head"),
                )
                continue
            filtered_lines.append(
                f"{next_rank}. {m.group('head')} — {m.group('tail')}"
            )
            next_rank += 1
        filtered_summary = "\n".join(filtered_lines)

        artifact = {"summary": filtered_summary, "blurbs": validated}
        return "Supervisor analysis complete.", artifact

    def get_merged_allowlist() -> (
        dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]
    ):
        """Snapshot the post-merge competitor + mechanism disease allowlist.

        Returns a copy keyed by lowercase disease name → (canonical_name, source). Sources are
        "competitor", "mechanism", or "both" depending on which sub-agent surfaced the disease.
        """
        return dict(allowed_diseases)

    def get_auto_findings() -> dict[str, dict]:
        """Snapshot artifacts produced by investigate_top_candidates (holdout-only).

        Returns {lowercase_canonical_disease: {"literature": LiteratureOutput,
        "clinical_trials": ClinicalTrialsOutput}}. Empty in non-holdout runs.
        """
        return dict(auto_findings)

    tools = [
        find_candidates,
        analyze_mechanism,
        analyze_literature,
        analyze_clinical_trials,
        get_drug_briefing,
        critique_ranking,
        finalize_supervisor,
    ]
    if date_before is not None or get_settings().supervisor_fanout:
        # Insert investigate_top_candidates before finalize so the LLM can see it
        # after seed-phase tools but before terminating. Enabled in holdout mode
        # (recover "obvious" candidates) and when supervisor_fanout is set (parallel
        # fan-out for speed; see config.Settings.supervisor_fanout).
        tools.insert(-1, investigate_top_candidates)
    return tools, get_merged_allowlist, get_auto_findings
