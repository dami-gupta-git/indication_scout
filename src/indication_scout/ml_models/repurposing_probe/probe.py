"""
Drug-repurposing approval classifier — feature-distribution probe.

Mirrors the lift + partial-coefficient pattern used in probes 4 and 6 of
SESSION_FINDINGS.md. The goal is NOT to ship a model — it is to decide
whether the cached data has signal, or whether (as in the success_classifier
probe) one feature dominates by leakage.

Inputs (all from _scout_cache/ + pgvector):
  - Labels:    fda_drug_disease_approval/*.json  -> (drug, disease, verdict)
  - Diseases:  mesh_resolver/*.json              -> disease -> (mesh_id, mesh_term)
  - Competitors: competitors_merged/*.json       -> chembl_id -> {disease: [drugs]}
  - Trials:    ct_completed/*.json, ct_terminated/*.json keyed by (drug, mesh_term, date_before)
  - Expand:    expand_search_terms/*.json        -> (chembl_id, disease) -> [terms]
  - ATC:       atc_description/*.json            -> chembl-derived; here we look up via drug name
  - Literature volume: pubmed_abstracts table (pgvector) — count of abstracts
                whose title/abstract mentions the drug AND whose mesh_terms
                contain the disease's MeSH descriptor.

Outputs:
  - features.csv (one row per (drug, disease))
  - probe1_distributions.txt — per-class means (pos vs neg)
  - probe2_lifts.txt         — P(label=1 | feature present) and lift over baseline
  - probe3_lr.txt            — leave-one-drug-out LR coefficients + AUC
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("repurposing_probe")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE = PROJECT_ROOT / "_scout_cache"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def load_labels() -> pd.DataFrame:
    """Build (drug, disease, label) rows from fda_drug_disease_approval cache."""
    rows = []
    for f in sorted(glob(str(CACHE / "fda_drug_disease_approval" / "*.json"))):
        d = json.load(open(f))
        drug = _norm(d.get("drug_name", ""))
        for disease, verdict in d.get("entries", {}).items():
            v = verdict.get("verdict")
            if v is None:
                continue
            rows.append(
                {"drug": drug, "disease": _norm(disease), "label": int(bool(v))}
            )
    df = pd.DataFrame(rows).drop_duplicates(["drug", "disease"])
    log.info(
        "labels: n=%d  pos=%d  neg=%d", len(df), df.label.sum(), (df.label == 0).sum()
    )
    return df


def load_disease_to_mesh() -> dict[str, tuple[str, str]]:
    """disease_name -> (mesh_id, mesh_term)."""
    out = {}
    for f in sorted(glob(str(CACHE / "mesh_resolver" / "*.json"))):
        d = json.load(open(f))
        params = d.get("params", {})
        name = _norm(params.get("indication", ""))
        data = d.get("data")
        if not name or not data:
            continue
        if isinstance(data, list) and len(data) >= 2:
            out[name] = (data[0], data[1])
    log.info("mesh_resolver entries: %d", len(out))
    return out


def load_drug_to_chembl() -> dict[str, str]:
    """drug name (any synonym) -> chembl_id."""
    out = {}
    for f in sorted(glob(str(CACHE / "chembl_id_to_names" / "*.json"))):
        d = json.load(open(f))
        chembl = d.get("chembl_id")
        for n in d.get("names", []) or []:
            out[_norm(n)] = chembl
    return out


def load_chembl_to_primary_drug() -> dict[str, str]:
    """chembl_id -> first (primary) drug name listed in the synonyms cache."""
    out: dict[str, str] = {}
    for f in sorted(glob(str(CACHE / "chembl_id_to_names" / "*.json"))):
        d = json.load(open(f))
        chembl = d.get("chembl_id")
        names = d.get("names") or []
        if chembl and names:
            out[chembl] = _norm(names[0])
    return out


def load_competitors(drug_to_chembl: dict[str, str]) -> dict[tuple[str, str], int]:
    """Returns (drug, disease) -> n_competitors (excluding the drug itself)."""
    chembl_to_disease_drugs: dict[str, dict[str, list[str]]] = {}
    for f in sorted(glob(str(CACHE / "competitors_merged" / "*.json"))):
        d = json.load(open(f))
        chembl = d.get("params", {}).get("chembl_id")
        if not chembl:
            continue
        chembl_to_disease_drugs.setdefault(chembl, {}).update(
            {_norm(k): [_norm(x) for x in v] for k, v in d.get("data", {}).items()}
        )

    out: dict[tuple[str, str], int] = {}
    for drug, chembl in drug_to_chembl.items():
        if chembl not in chembl_to_disease_drugs:
            continue
        for disease, comps in chembl_to_disease_drugs[chembl].items():
            n = sum(1 for c in comps if c != drug)
            out[(drug, disease)] = max(out.get((drug, disease), 0), n)
    log.info("competitor pairs: %d", len(out))
    return out


def load_ct_counts() -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """(drug, mesh_term) -> n_completed / n_terminated (summed over date_before keys)."""
    completed: dict[tuple[str, str], int] = {}
    terminated: dict[tuple[str, str], int] = {}
    for f in sorted(glob(str(CACHE / "ct_completed" / "*.json"))):
        d = json.load(open(f))
        p = d.get("params", {})
        key = (_norm(p.get("drug", "")), _norm(p.get("mesh_term", "")))
        completed[key] = max(
            completed.get(key, 0), d.get("data", {}).get("total_count", 0)
        )
    for f in sorted(glob(str(CACHE / "ct_terminated" / "*.json"))):
        d = json.load(open(f))
        p = d.get("params", {})
        key = (_norm(p.get("drug", "")), _norm(p.get("mesh_term", "")))
        terminated[key] = max(
            terminated.get(key, 0), d.get("data", {}).get("total_count", 0)
        )
    log.info(
        "ct_completed keys=%d  ct_terminated keys=%d", len(completed), len(terminated)
    )
    return completed, terminated


def load_expand_terms(
    drug_to_chembl: dict[str, str], fda_drugs: set[str]
) -> dict[tuple[str, str], int]:
    """(drug, disease) -> n expanded search terms generated by the agent.

    Inverts drug_to_chembl preferring an FDA-canonical drug name when several
    synonyms collide on the same chembl_id (so 'dasatinib' wins over
    'dasatinib anhydrous').
    """
    chembl_to_drug: dict[str, str] = {}
    for name, chembl in drug_to_chembl.items():
        if not chembl:
            continue
        cur = chembl_to_drug.get(chembl)
        # Prefer a name that appears in the fda label set; fall back to first seen.
        if cur is None or (name in fda_drugs and cur not in fda_drugs):
            chembl_to_drug[chembl] = name

    out: dict[tuple[str, str], int] = {}
    for f in sorted(glob(str(CACHE / "expand_search_terms" / "*.json"))):
        d = json.load(open(f))
        p = d.get("params", {})
        chembl = p.get("chembl_id")
        disease = _norm(p.get("disease_name", ""))
        if chembl not in chembl_to_drug:
            continue
        drug = chembl_to_drug[chembl]
        out[(drug, disease)] = len(d.get("data", []) or [])
    log.info("expand_terms pairs: %d  (joined to fda drug names)", len(out))
    return out


def load_atc_per_drug() -> dict[str, dict[str, str]]:
    """drug -> {level1, level2, level3, level4} ATC codes (first match)."""
    out: dict[str, dict[str, str]] = {}
    for f in sorted(glob(str(CACHE / "atc_description" / "*.json"))):
        d = json.load(open(f))
        data = d.get("data", {})
        drug = _norm(data.get("who_name", ""))
        if not drug:
            continue
        out.setdefault(
            drug,
            {
                "atc_level1": data.get("level1", ""),
                "atc_level2": data.get("level2", ""),
                "atc_level3": data.get("level3", ""),
            },
        )
    log.info("ATC drugs: %d", len(out))
    return out


def load_pubmed_volume(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """For each (drug, mesh_term) count pubmed_abstracts that mention the drug and have the mesh term.

    Uses ILIKE on title+abstract for the drug name and ANY-array on mesh_terms for the descriptor term.
    """
    import os
    from sqlalchemy import create_engine, text

    load_dotenv()
    if "DATABASE_URL" not in os.environ:
        log.warning("DATABASE_URL not set; skipping literature feature")
        return {}

    eng = create_engine(os.environ["DATABASE_URL"])
    out: dict[tuple[str, str], int] = {}

    cache_path = OUT / "pubmed_volume_cache.json"
    cached: dict[str, int] = {}
    if cache_path.exists():
        cached = json.load(open(cache_path))
        log.info("pubmed cache: %d entries on disk", len(cached))

    targets = sorted(set(pairs))
    log.info("pubmed lookups: %d (drug, mesh_term) pairs", len(targets))

    sql = text("""
        SELECT count(*) FROM pubmed_abstracts
        WHERE :mesh = ANY(mesh_terms)
          AND (title ILIKE :pat OR coalesce(abstract,'') ILIKE :pat)
    """)
    with eng.connect() as c:
        for i, (drug, mesh_term) in enumerate(targets):
            if i and i % 200 == 0:
                log.info("  pubmed %d / %d", i, len(targets))
            ck = f"{drug}|{mesh_term}"
            if ck in cached:
                n = cached[ck]
            else:
                n = (
                    c.execute(sql, {"mesh": mesh_term, "pat": f"%{drug}%"}).scalar()
                    or 0
                )
                cached[ck] = int(n)
            out[(drug, _norm(mesh_term))] = int(n)

    json.dump(cached, open(cache_path, "w"))
    log.info(
        "pubmed volume entries: %d  (nonzero=%d)  cache_size=%d",
        len(out),
        sum(1 for v in out.values() if v > 0),
        len(cached),
    )
    return out


def build_features(
    labels: pd.DataFrame,
    disease_to_mesh: dict[str, tuple[str, str]],
    drug_to_chembl: dict[str, str],
    competitors: dict[tuple[str, str], int],
    ct_completed: dict[tuple[str, str], int],
    ct_terminated: dict[tuple[str, str], int],
    expand_terms: dict[tuple[str, str], int],
    atc: dict[str, dict[str, str]],
    pubmed_vol: dict[tuple[str, str], int],
) -> pd.DataFrame:
    rows = []
    for _, r in labels.iterrows():
        drug = r["drug"]
        disease = r["disease"]
        mesh = disease_to_mesh.get(disease)
        mesh_term = _norm(mesh[1]) if mesh else ""
        n_comp = competitors.get((drug, disease), 0)
        n_done = ct_completed.get((drug, mesh_term), 0)
        n_term = ct_terminated.get((drug, mesh_term), 0)
        n_exp = expand_terms.get((drug, disease), 0)
        n_lit = pubmed_vol.get((drug, mesh_term), 0)
        chembl = drug_to_chembl.get(drug, "")
        atc_d = atc.get(drug, {})
        rows.append(
            {
                "drug": drug,
                "disease": disease,
                "mesh_id": mesh[0] if mesh else "",
                "mesh_term": mesh_term,
                "chembl_id": chembl,
                "label": r["label"],
                "n_competitors": n_comp,
                "has_competitor": int(n_comp > 0),
                "n_ct_completed": n_done,
                "n_ct_terminated": n_term,
                "n_ct_total": n_done + n_term,
                "term_to_total_ratio": (
                    (n_term / (n_done + n_term)) if (n_done + n_term) else 0.0
                ),
                "n_expand_terms": n_exp,
                "has_expand": int(n_exp > 0),
                "n_pubmed": n_lit,
                "log_n_pubmed": np.log1p(n_lit),
                "has_pubmed": int(n_lit > 0),
                "atc_level1": atc_d.get("atc_level1", ""),
                "atc_level2": atc_d.get("atc_level2", ""),
            }
        )
    df = pd.DataFrame(rows)
    log.info("feature matrix: rows=%d  cols=%d", len(df), len(df.columns))
    return df


def probe_distributions(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    g = df.groupby("label")[feats].mean().T
    g.columns = ["neg_mean", "pos_mean"]
    g["diff"] = g["pos_mean"] - g["neg_mean"]
    g["ratio_pos_over_neg"] = g["pos_mean"] / g["neg_mean"].replace(0, np.nan)
    return g.sort_values("diff", ascending=False)


def probe_lifts(df: pd.DataFrame, indicator_feats: list[str]) -> pd.DataFrame:
    base = df.label.mean()
    rows = []
    for f in indicator_feats:
        present = df[df[f] > 0]
        if not len(present):
            continue
        p = present.label.mean()
        rows.append(
            {
                "feature": f,
                "n_present": len(present),
                "P(pos|present)": p,
                "lift_over_baseline": p / base if base else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("lift_over_baseline", ascending=False)


def probe_lr(df: pd.DataFrame, feats: list[str], group_col: str = "drug") -> dict:
    X = df[feats].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df[group_col].values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Leave-one-drug-out CV
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y), dtype=float)
    valid = np.zeros(len(y), dtype=bool)
    for tr, te in logo.split(X_std, y, groups):
        if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        m = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        m.fit(X_std[tr], y[tr])
        preds[te] = m.predict_proba(X_std[te])[:, 1]
        valid[te] = True

    # Final fit on all data for coefficient inspection
    m_full = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    m_full.fit(X_std, y)

    coefs = pd.DataFrame({"feature": feats, "coef_std": m_full.coef_[0]}).sort_values(
        "coef_std", ascending=False
    )

    if valid.sum() and y[valid].sum() and (y[valid] == 0).sum():
        roc = roc_auc_score(y[valid], preds[valid])
        pr = average_precision_score(y[valid], preds[valid])
    else:
        roc = float("nan")
        pr = float("nan")

    return {
        "n": int(len(y)),
        "p": int(len(feats)),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "n_groups": int(len(set(groups))),
        "roc_auc_logo": float(roc),
        "pr_auc_logo": float(pr),
        "baseline_pr": float(y.mean()),
        "coefs": coefs,
    }


def main():
    labels = load_labels()
    disease_to_mesh = load_disease_to_mesh()
    drug_to_chembl = load_drug_to_chembl()
    competitors = load_competitors(drug_to_chembl)
    ct_completed, ct_terminated = load_ct_counts()
    expand_terms = load_expand_terms(drug_to_chembl, set(labels["drug"].unique()))
    atc = load_atc_per_drug()

    pairs_needed: list[tuple[str, str]] = []
    for _, r in labels.iterrows():
        mesh = disease_to_mesh.get(r["disease"])
        if mesh:
            pairs_needed.append((r["drug"], mesh[1]))
    pubmed_vol = load_pubmed_volume(pairs_needed)

    df = build_features(
        labels,
        disease_to_mesh,
        drug_to_chembl,
        competitors,
        ct_completed,
        ct_terminated,
        expand_terms,
        atc,
        pubmed_vol,
    )
    df.to_csv(OUT / "features.csv", index=False)
    log.info("wrote features.csv (n=%d, pos=%d)", len(df), df.label.sum())

    # Coverage stats — how many rows actually have each feature populated?
    cov = pd.DataFrame(
        {
            "n_rows": [len(df)] * 6,
            "feature": [
                "mesh_resolved",
                "n_competitors>0",
                "n_ct_total>0",
                "n_expand>0",
                "n_pubmed>0",
                "atc",
            ],
            "n_with_feature": [
                int((df.mesh_term != "").sum()),
                int((df.n_competitors > 0).sum()),
                int((df.n_ct_total > 0).sum()),
                int((df.n_expand_terms > 0).sum()),
                int((df.n_pubmed > 0).sum()),
                int((df.atc_level1 != "").sum()),
            ],
        }
    )
    cov["coverage"] = cov["n_with_feature"] / cov["n_rows"]
    print("\n=== COVERAGE ===")
    print(cov.to_string(index=False))
    cov.to_csv(OUT / "coverage.csv", index=False)

    numeric_feats = [
        "n_competitors",
        "has_competitor",
        "n_ct_completed",
        "n_ct_terminated",
        "n_ct_total",
        "term_to_total_ratio",
        "n_expand_terms",
        "has_expand",
        "n_pubmed",
        "log_n_pubmed",
        "has_pubmed",
    ]

    # Probe 1: per-class means
    print("\n=== PROBE 1: per-class means (pos vs neg) ===")
    dist = probe_distributions(df, numeric_feats)
    print(dist.to_string())
    dist.to_csv(OUT / "probe1_distributions.csv")

    # Probe 2: indicator-style lifts
    print("\n=== PROBE 2: P(label=1 | feature present) and lift over baseline ===")
    lift = probe_lifts(df, ["has_competitor", "has_expand", "has_pubmed"])
    # also bucketed-quantile lifts on the log lit feature
    df["log_lit_q"] = pd.qcut(df["log_n_pubmed"], q=4, duplicates="drop", labels=False)
    by_q = df.groupby("log_lit_q").agg(n=("label", "size"), pos_rate=("label", "mean"))
    print(lift.to_string(index=False))
    print("\nP(pos) by log_n_pubmed quartile:")
    print(by_q.to_string())
    lift.to_csv(OUT / "probe2_lifts.csv", index=False)

    # Probe 3: LR partial coefficients
    print("\n=== PROBE 3: leave-one-drug-out LR ===")
    full_feats = [
        "n_competitors",
        "n_ct_completed",
        "n_ct_terminated",
        "term_to_total_ratio",
        "n_expand_terms",
        "log_n_pubmed",
    ]
    res = probe_lr(df, full_feats)
    print(
        f"n={res['n']}  p={res['p']}  pos={res['pos']} ({res['pos_rate']:.2%})  "
        f"groups={res['n_groups']}"
    )
    print(f"ROC-AUC (LOGO):  {res['roc_auc_logo']:.3f}  (baseline 0.500)")
    print(
        f"PR-AUC  (LOGO):  {res['pr_auc_logo']:.3f}  (baseline {res['baseline_pr']:.3f})"
    )
    print("\nLR coefficients (standardized):")
    print(res["coefs"].to_string(index=False))
    res["coefs"].to_csv(OUT / "probe3_coefs.csv", index=False)

    # Probe 4: LR without literature, to check whether literature dominates
    print("\n=== PROBE 4: LR with NO literature feature (honest probe) ===")
    no_lit = [f for f in full_feats if "pubmed" not in f and "lit" not in f]
    res2 = probe_lr(df, no_lit)
    print(f"n={res2['n']}  p={res2['p']}  pos={res2['pos']}  groups={res2['n_groups']}")
    print(f"ROC-AUC (LOGO):  {res2['roc_auc_logo']:.3f}")
    print(
        f"PR-AUC  (LOGO):  {res2['pr_auc_logo']:.3f}  (baseline {res2['baseline_pr']:.3f})"
    )
    print(res2["coefs"].to_string(index=False))
    res2["coefs"].to_csv(OUT / "probe4_coefs_nolit.csv", index=False)

    # Probe 5: literature-only model (does literature alone explain the signal?)
    print("\n=== PROBE 5: LR with ONLY literature volume ===")
    res3 = probe_lr(df, ["log_n_pubmed"])
    print(f"ROC-AUC (LOGO):  {res3['roc_auc_logo']:.3f}")
    print(
        f"PR-AUC  (LOGO):  {res3['pr_auc_logo']:.3f}  (baseline {res3['baseline_pr']:.3f})"
    )

    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
