"""Mechanism tools.

Uses content_and_artifact so typed Python objects are preserved on msg.artifact. Tools share inter-call
data via a closure-scoped store dict. No InjectedState, no LangGraph state machinery.
"""

import logging

from langchain_core.tools import tool

from indication_scout.config import get_settings
from indication_scout.constants import MECHANISM_SIGNAL_KEYS
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import Association, MechanismOfAction

logger = logging.getLogger(__name__)

_settings = get_settings()


def build_mechanism_tools() -> list:
    """Build tools that share data via a closure-scoped store dict.

    get_drug must be called first — it populates the store with the symbol→target_id map that
    get_target_associations uses to resolve target symbols.
    """

    store: dict = {}

    @tool(response_format="content_and_artifact")
    async def get_drug(drug_name: str) -> tuple[str, list[MechanismOfAction]]:
        """Fetch drug data including mechanisms of action.

        Returns the list of MechanismOfAction entries for the drug — each entry has an action_type (e.g.
        INHIBITOR, AGONIST), a mechanism string, and the targets it applies to. Also populates the
        internal target map so subsequent tools can resolve target symbols. Call this first.
        """
        try:
            async with OpenTargetsClient() as client:
                chembl_id = await resolve_drug_name(drug_name, client.cache_dir)
                drug = await client.get_drug(chembl_id)
        except DataSourceError as e:
            # Resolution failed (unknown name, salt with no parent, etc.) — return an empty MoA list so
            # the agent's "no mechanisms" short-circuit fires instead of crashing the agent loop.
            logger.info("get_drug: could not resolve %r: %s", drug_name, e)
            store["target_ids"] = {}
            return (
                f"No drug found for '{drug_name}' — our tools could not resolve "
                f"this name to a ChEMBL entry.",
                [],
            )

        store["target_ids"] = {
            t.target_symbol: t.target_id
            for t in drug.targets
            if t.target_symbol and t.target_id
        }
        moas = drug.mechanisms_of_action

        summary = (
            ", ".join(f"{m.action_type} ({m.mechanism_of_action})" for m in moas)
            or "none"
        )
        return (
            f"Found {len(moas)} mechanism(s) for {drug_name} ({chembl_id}): {summary}",
            moas,
        )

    @tool(response_format="content_and_artifact")
    async def get_target_associations(
        target_symbol: str,
    ) -> tuple[str, dict[str, list[Association]]]:
        """Fetch disease associations for a target, ranked by overall score.

        Returns a dict keyed by target symbol mapping to the top disease associations (capped,
        signal-filtered) with per-datatype evidence scores (genetic_association, literature, etc.). Call
        get_drug first so the target ID is available.
        """
        target_ids: dict[str, str] = store.get("target_ids", {})
        target_id = target_ids.get(target_symbol)
        if not target_id:
            return (
                f"Target '{target_symbol}' not found in store — call get_drug first",
                {},
            )

        async with OpenTargetsClient() as client:
            associations = await client.get_target_data_associations(target_id)

        filtered = [
            a
            for a in associations
            if any(
                a.datatype_scores.get(k, 0) >= _settings.mechanism_signal_threshold
                for k in MECHANISM_SIGNAL_KEYS
            )
        ]
        top = sorted(filtered, key=lambda a: a.overall_score or 0, reverse=True)[
            : _settings.mechanism_associations_cap
        ]

        fetched = store.setdefault("fetched_associations", {})
        for a in top:
            fetched[a.disease_id] = a

        rows = "\n".join(
            (
                f"  {a.disease_id} | {a.disease_name} | overall={a.overall_score:.3f} | "
                f"{a.datatype_scores}"
            )
            for a in top
        )
        return (
            f"{len(top)} signal-filtered associations for {target_symbol} "
            f"(of {len(associations)} total):\n{rows}",
            {target_symbol: top},
        )

    @tool(response_format="content_and_artifact", return_direct=True)
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete.

        Call this as the very last step, passing your 3-4 sentence plain-text summary of the mechanistic
        findings. This terminates the agent loop.
        """
        return "Analysis complete.", summary

    return [get_drug, get_target_associations, finalize_analysis]
