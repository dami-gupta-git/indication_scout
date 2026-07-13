import logging
from datetime import date
from pathlib import Path

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

SALT_SUFFIXES = [
    " hydrochloride",
    " hydrobromide",
    " sulfate",
    " succinate",
    " chloride",
    " dimesylate",
    " tartrate",
    " citrate",
    " tosylate",
    " mesylate",
    " saccharate",
    " hemihydrate",
    " maleate",
    " phosphate",
    " malate",
    " esylate",
    " anhydrous",
]


def normalize_drug_name(name: str) -> str:
    name_lower = name.lower()
    for suffix in SALT_SUFFIXES:
        if name_lower.endswith(suffix):
            return name_lower[: -len(suffix)].strip()
    return name_lower


class DrugIntake(BaseModel):
    """Drug-level facts seeded before the per-pair sub-agents run.

    Shared by find_candidates (supervisor path) and run_pair_analysis (CLI pair path). Holds the
    ChEMBL id plus the facts the clinical-trials / literature sub-agents read: aliases,
    first_approval, and the drug's own FDA-approved indications. No competitor diseases, no
    approved-DROP — that stays in find_candidates.
    """

    chembl_id: str = ""
    aliases: list[str] = []
    first_approval: int | None = (
        None  # year first approved anywhere; None if unresolved
    )
    approved_indications: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


async def seed_drug_intake(
    drug_name: str, cache_dir: Path, date_before: date | None = None
) -> DrugIntake:
    """Resolve a drug and gather its drug-level facts (aliases, first_approval, approved indications).

    Closure-free extraction of the drug-level seeding formerly inlined in find_candidates. Resolves the ChEMBL id itself.
    Each external call is guarded — a failure logs and leaves the corresponding field at its empty/None default (no
    fabricated values, per project rules).

    When `date_before` is set, the approved-indication path swaps live openFDA for the hardcoded approvals table (the live
    path leaks approvals past the cutoff); mirrors the swap find_candidates performed inline.
    """
    # Deferred imports: these modules pull in data-source clients that read settings at import time. Keeping them local
    # avoids import-time settings access in this otherwise-pure helper module.
    from indication_scout.data_sources.chembl import (
        ChEMBLClient,
        get_all_drug_names,
        resolve_drug_name,
    )
    from indication_scout.data_sources.fda import FDAClient
    from indication_scout.services.approval_check import (
        list_approved_indications_at,
        list_approved_indications_from_labels,
    )

    drug_name = normalize_drug_name(drug_name)
    # Raises DataSourceError if the drug is unknown — caller relies on this as a fail-fast existence check.
    chembl_id = await resolve_drug_name(drug_name, cache_dir)

    aliases: list[str] = []
    try:
        aliases = await get_all_drug_names(chembl_id, cache_dir)
    except Exception as e:
        logger.warning(
            "seed_drug_intake: get_all_drug_names failed for %s: %s", chembl_id, e
        )

    first_approval: int | None = None
    try:
        async with ChEMBLClient(cache_dir=cache_dir) as chembl_client:
            molecule = await chembl_client.get_molecule(chembl_id)
        first_approval = molecule.first_approval
    except Exception as e:
        logger.warning(
            "seed_drug_intake: get_molecule(first_approval) failed for %s: %s",
            chembl_id,
            e,
        )

    # Seed approved_indications from the drug's own FDA label, independent of any candidate list.
    seed_aliases = aliases or [drug_name]
    approved_indications: list[str] = []
    try:
        if date_before is not None:
            approved_indications = list_approved_indications_at(drug_name, date_before)
        else:
            async with FDAClient(cache_dir=cache_dir) as fda_client:
                label_texts = await fda_client.get_all_label_indications(seed_aliases)
            approved_indications = await list_approved_indications_from_labels(
                label_texts=label_texts,
                cache_dir=cache_dir,
            )
    except Exception as e:
        logger.warning(
            "seed_drug_intake: approved-indication seed failed for %s: %s", drug_name, e
        )

    return DrugIntake(
        chembl_id=chembl_id,
        aliases=aliases,
        first_approval=first_approval,
        approved_indications=approved_indications,
    )
