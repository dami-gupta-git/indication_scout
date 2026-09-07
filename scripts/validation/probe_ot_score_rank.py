"""Probe: rank the merged (competitor + mechanism) candidate list by OT association score.

For each drug we build the SAME candidate universe the supervisor would see —
competitor diseases (sibling-drug ranking, via `get_drug_competitors`) UNION
mechanism diseases (OT target-disease associations over the drug's targets) — then
score EVERY candidate on ONE comparable axis: the Open Targets `overall_score` for
that (target, disease) pair, taken as the max across the drug's targets. We rank the
merged list by that score and report, per known approved indication, where it lands.

This answers: "how good is the OT association rank?" for the full merged set, with
competitor and mechanism candidates on the same scale (option B — no invented
cross-source weighting). Diagnostic only; writes nothing into the pipeline.

OT associations have no date filter, so `overall_score` reflects today's evidence
(it can include post-cutoff literature). The competitor list IS cutoff-aware via
`date_before`. Writes a CSV (drug name in every row) with rank, score, source, the
target-indication flag, and the approval date when the disease is a known approved
indication (matched against drug_approvals.json — curated ground-truth, not fabricated).

Run:
    CONSTANTS_FILE=.env.constants .venv/bin/python scripts/validation/probe_ot_score_rank.py
"""

import asyncio
import csv
import logging
from datetime import date
from pathlib import Path

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.services.approval_check import _load_drug_approvals_table
from indication_scout.services.llm import query_small_llm
from indication_scout.services.retrieval import RetrievalService

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("probe_ot_score_rank")

# CSV is capped to the top-N ranked candidates per drug (the slice an investigation cap
# would actually reach). The full ranked list still prints to stdout.
TOP_N = 10

# My own repurposing judgment per (drug, exact ranked disease string) for the top-N.
# strong = well-supported repurposing rationale; plausible = defensible but speculative;
# weak = thin/indirect; not_repurposing = monogenic syndrome / on-label / artifact, not a
# repurposing opportunity. This is an Opus judgment, NOT ground truth — the approved_date
# column carries the factual approval status separately.
JUDGMENT: dict[tuple[str, str], str] = {
    # metformin — PDE/AMPK-mitochondrial associations dominate; none are repurposing targets.
    ("metformin", "mitochondrial complex i deficiency"): "not_repurposing",
    ("metformin", "leigh syndrome"): "not_repurposing",
    ("metformin", "leber hereditary optic neuropathy"): "not_repurposing",
    ("metformin", "melas syndrome"): "not_repurposing",
    (
        "metformin",
        "microphthalmia with linear skin defects syndrome",
    ): "not_repurposing",
    # imatinib — KIT/PDGFR/ABL. GIST/CML/ALL on-target oncology; rare KIT/PDGFR syndromes mixed in.
    ("imatinib", "gastrointestinal stromal tumor"): "strong",
    ("imatinib", "chronic myelogenous leukemia"): "strong",
    ("imatinib", "piebaldism"): "not_repurposing",
    ("imatinib", "cutaneous mastocytosis"): "plausible",
    (
        "imatinib",
        "congenital heart defects and skeletal malformations syndrome",
    ): "not_repurposing",
    ("imatinib", "bilateral striopallidodentate calcinosis"): "not_repurposing",
    ("imatinib", "basal ganglia calcification"): "not_repurposing",
    ("imatinib", "myofibromatosis"): "plausible",
    (
        "imatinib",
        "skeletal overgrowth-craniofacial dysmorphism-hyperelastic skin-white matter lesions syndrome",
    ): "not_repurposing",
    ("imatinib", "acute lymphoblastic leukemia"): "strong",
    # sildenafil — PDE5/NO-cGMP vasodilation. PAH strong; many cardiovascular extensions plausible.
    ("sildenafil", "coronary artery disease"): "weak",
    ("sildenafil", "erectile dysfunction"): "not_repurposing",
    ("sildenafil", "pulmonary arterial hypertension"): "strong",
    ("sildenafil", "benign prostatic hyperplasia"): "plausible",
    ("sildenafil", "pulmonary hypertension"): "strong",
    ("sildenafil", "stroke"): "weak",
    ("sildenafil", "impotence"): "not_repurposing",
    ("sildenafil", "hypertension"): "weak",
    ("sildenafil", "cardiovascular disease"): "weak",
    ("sildenafil", "heart failure"): "plausible",
    # baricitinib — JAK1/2. Myeloproliferative cluster plausible; RA on-label; alopecia/UC strong.
    ("baricitinib", "polycythemia vera"): "plausible",
    ("baricitinib", "primary myelofibrosis"): "plausible",
    ("baricitinib", "myelofibrosis"): "plausible",
    ("baricitinib", "myeloproliferative disorder"): "plausible",
    ("baricitinib", "neoplasm"): "not_repurposing",
    ("baricitinib", "splenomegaly"): "not_repurposing",
    ("baricitinib", "rheumatoid arthritis"): "strong",
    ("baricitinib", "autoinflammation"): "plausible",
    ("baricitinib", "acute myeloid leukemia"): "weak",
    ("baricitinib", "ulcerative colitis"): "strong",
    # duloxetine — SNRI. Pain/psych extensions strong-to-plausible; respiratory rows are artifacts.
    ("duloxetine", "obsessive-compulsive disorder"): "plausible",
    ("duloxetine", "major depressive disorder"): "not_repurposing",
    ("duloxetine", "depressive disorder"): "not_repurposing",
    ("duloxetine", "attention deficit hyperactivity disorder"): "weak",
    ("duloxetine", "fibromyalgia"): "strong",
    ("duloxetine", "panic disorder"): "plausible",
    ("duloxetine", "obesity"): "weak",
    ("duloxetine", "nasal congestion"): "not_repurposing",
    ("duloxetine", "seasonal allergic rhinitis"): "not_repurposing",
    ("duloxetine", "common cold"): "not_repurposing",
}

# Rich, well-studied drugs with multiple known/plausible repurposing indications and
# good OT target coverage. cutoff = a date before the repurposing approval so the
# competitor list is built from pre-approval state. (target indication, cutoff)
DRUGS: dict[str, tuple[list[str], date]] = {
    "metformin": (
        ["polycystic ovary syndrome", "gestational diabetes"],
        date(2010, 1, 1),
    ),
    "imatinib": (
        ["gastrointestinal stromal tumor", "hypereosinophilic syndrome"],
        date(2001, 1, 1),
    ),
    "sildenafil": (["pulmonary arterial hypertension"], date(2005, 1, 1)),
    "baricitinib": (["alopecia areata", "rheumatoid arthritis"], date(2018, 1, 1)),
    "duloxetine": (["fibromyalgia", "diabetic neuropathy"], date(2004, 1, 1)),
}


async def _ot_scores_by_efo(
    client: OpenTargetsClient, chembl_id: str
) -> tuple[dict[str, float], dict[str, str]]:
    """Map EFO id -> max OT overall_score across the drug's targets, plus EFO -> name.

    The score axis both sources are placed on. A disease's score is the strongest
    (target, disease) association among all of the drug's targets.
    """
    drug = await client.get_drug(chembl_id)
    score_by_efo: dict[str, float] = {}
    name_by_efo: dict[str, str] = {}
    for t in drug.targets:
        if not t.target_id:
            continue
        associations = await client.get_target_data_associations(t.target_id)
        for a in associations:
            if not a.disease_id or a.overall_score is None:
                continue
            prev = score_by_efo.get(a.disease_id)
            if prev is None or a.overall_score > prev:
                score_by_efo[a.disease_id] = a.overall_score
            if a.disease_name:
                name_by_efo[a.disease_id] = a.disease_name
    return score_by_efo, name_by_efo


async def _build_merged(
    svc: RetrievalService, client: OpenTargetsClient, chembl_id: str, cutoff: date
) -> tuple[list[dict], dict[str, float]]:
    """Build the merged candidate list with per-candidate (efo, name, source, score).

    competitor candidates come from get_drug_competitors (cutoff-aware); mechanism
    candidates are every EFO with an OT association score. A candidate in BOTH is
    tagged 'both'. Score is the OT overall_score (None if the disease has no
    association on any of the drug's targets — kept, ranked last).
    """
    score_by_efo, name_by_efo = await _ot_scores_by_efo(client, chembl_id)

    # Competitor side: get_drug_competitors returns canonical name -> set(drugs).
    # Resolve each to an EFO via the raw competitor cache so we can look up its score.
    raw = await client.get_drug_competitors(chembl_id, date_before=cutoff)
    comp_efo_by_name: dict[str, str] = raw["disease_efo_ids"]  # canonical lower -> efo
    comp_names = set(
        (await svc.get_drug_competitors(chembl_id, date_before=cutoff)).keys()
    )

    rows: list[dict] = []
    seen_efo: set[str] = set()

    # Competitor candidates first.
    for name in sorted(comp_names):
        efo = comp_efo_by_name.get(name.lower().strip())
        score = score_by_efo.get(efo) if efo else None
        if efo:
            seen_efo.add(efo)
        rows.append({"name": name, "efo": efo, "source": "competitor", "score": score})

    # Mechanism-only candidates: EFOs with a score that weren't competitor matches.
    for efo, score in score_by_efo.items():
        if efo in seen_efo:
            # Upgrade the competitor row to 'both'.
            for r in rows:
                if r["efo"] == efo:
                    r["source"] = "both"
                    if r["score"] is None:
                        r["score"] = score
            continue
        rows.append(
            {
                "name": name_by_efo.get(efo, efo),
                "efo": efo,
                "source": "mechanism",
                "score": score,
            }
        )

    return rows, score_by_efo


async def _match(target: str, names: list[str]) -> int | None:
    """LLM-resolve the target indication to an index in `names`, or None."""
    if not names:
        return None
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(names))
    prompt = (
        "You map a target disease to a list of candidate disease names.\n"
        f'Target: "{target}"\n\n'
        "Candidates:\n"
        f"{numbered}\n\n"
        "Return ONLY the integer index of the candidate that best refers to the "
        "target indication. Count a match when the candidate is the SAME disease, an "
        "abbreviation, or a synonym. Do NOT match to a broader parent that drops the "
        "target's specificity. If no candidate fits, return -1. Return only the number."
    )
    resp = (await query_small_llm(prompt)).strip()
    try:
        idx = int(resp.split()[0])
    except (ValueError, IndexError):
        return None
    return idx if 0 <= idx < len(names) else None


def _ranked(rows: list[dict]) -> list[dict]:
    """Rank by OT overall_score desc; unscored (None) sink to the bottom, name-tiebreak."""
    return sorted(
        rows,
        key=lambda r: (-(r["score"] if r["score"] is not None else -1.0), r["name"]),
    )


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent.parent
OUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "holdout_validation"
    / "probe"
    / "probe_ot_score_rank.csv"
)


def _load_approvals() -> dict[str, list[dict]]:
    """drug -> list of {disease, approved}. Curated ground-truth, not fabricated."""
    return _load_drug_approvals_table()


async def _approved_index(approvals: list[dict], names: list[str]) -> dict[int, str]:
    """Map ranked-candidate index -> approval date for each approved indication.

    One LLM call per (approved disease) resolving it to the candidate index it matches
    (disease equivalence). Grounded entirely in drug_approvals.json — no invented labels.
    """
    out: dict[int, str] = {}
    for entry in approvals:
        idx = await _match(entry["disease"], names)
        if idx is not None:
            out[idx] = entry["approved"]
    return out


def _write_csv(per_drug: list[dict]) -> None:
    """One row per ranked candidate; drug name in every row; approved date if known."""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "drug",
                "chembl_id",
                "cutoff",
                "rank",
                "score",
                "source",
                "disease",
                "is_target_indication",
                "approved_date",
                "repurpose_judgment",
            ]
        )
        for d in per_drug:
            for i, r in enumerate(d["ranked"][:TOP_N], start=1):
                score = f"{r['score']:.3f}" if r["score"] is not None else ""
                judgment = JUDGMENT.get((d["drug"], r["name"]), "")
                w.writerow(
                    [
                        d["drug"],
                        d["chembl_id"],
                        d["cutoff"],
                        i,
                        score,
                        r["source"],
                        r["name"],
                        "yes" if i - 1 in d["target_idx"] else "",
                        d["approved_idx"].get(i - 1, ""),
                        judgment,
                    ]
                )
    print(f"\nwrote {OUT_CSV}")


async def main() -> None:
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    approvals = _load_approvals()
    summary: list[tuple[str, str, str, int | None, int, str | None]] = []
    per_drug: list[dict] = []

    async with OpenTargetsClient(cache_dir=svc.cache_dir) as client:
        for drug, (targets, cutoff) in DRUGS.items():
            chembl_id = await resolve_drug_name(drug, svc.cache_dir)
            if not chembl_id:
                logger.warning("%s: could not resolve to a ChEMBL id, skipping", drug)
                continue
            rows, _ = await _build_merged(svc, client, chembl_id, cutoff)
            ranked = _ranked(rows)
            names = [r["name"] for r in ranked]
            target_idx: set[int] = set()

            n_comp = sum(1 for r in rows if r["source"] == "competitor")
            n_both = sum(1 for r in rows if r["source"] == "both")
            n_mech = sum(1 for r in rows if r["source"] == "mechanism")
            print(
                f"\n=== {drug} ({chembl_id}) cutoff={cutoff.isoformat()} — "
                f"{len(rows)} candidates: {n_comp} competitor, {n_both} both, "
                f"{n_mech} mechanism-only ==="
            )
            print(f"  full ranked list ({len(ranked)}) by OT overall_score:")
            for i, r in enumerate(ranked, start=1):
                s = f"{r['score']:.3f}" if r["score"] is not None else "  —  "
                print(f"   {i:>2}. {s} | {r['source']:<10} | {r['name']}")

            for target in targets:
                idx = await _match(target, names)
                if idx is None:
                    print(f"  TARGET '{target}': not found in merged candidates")
                    summary.append((drug, target, "NOT FOUND", None, len(rows), None))
                    continue
                r = ranked[idx]
                target_idx.add(idx)
                s = f"{r['score']:.3f}" if r["score"] is not None else "None"
                print(
                    f"  TARGET '{target}': rank #{idx + 1}/{len(rows)} | "
                    f"score={s} | source={r['source']} | matched='{r['name']}'"
                )
                summary.append((drug, target, r["source"], idx + 1, len(rows), s))

            approved_idx = await _approved_index(approvals.get(drug, []), names[:TOP_N])
            per_drug.append(
                {
                    "drug": drug,
                    "chembl_id": chembl_id,
                    "cutoff": cutoff.isoformat(),
                    "n": len(rows),
                    "n_comp": n_comp,
                    "n_both": n_both,
                    "n_mech": n_mech,
                    "ranked": ranked,
                    "target_idx": target_idx,
                    "approved_idx": approved_idx,
                }
            )

    print("\n\n=== SUMMARY: target indication rank in OT-score-ranked merged list ===")
    print(
        f"{'drug':<12} {'target':<32} {'source':<11} {'rank':>8} {'n':>5} {'score':>8}"
    )
    for drug, target, source, rank, n, score in summary:
        rank_s = f"{rank}" if rank is not None else "—"
        score_s = score if score is not None else "—"
        print(
            f"{drug:<12} {target[:32]:<32} {source:<11} {rank_s:>8} {n:>5} {score_s:>8}"
        )

    _write_csv(per_drug)


if __name__ == "__main__":
    asyncio.run(main())
