"""Project-wide constants.

Tunable numeric limits (timeouts, retries, top-k, max results, concurrency,
etc.) live in config.py Settings and are overridable via environment variables
or alternate .env files (ENV_FILE=.env.test).

This file holds values that are structurally fixed: URLs, lookup maps,
keyword lists, and directory paths.
"""

from pathlib import Path

# -- LLM defaults -----------------------------------------------------------
DEFAULT_LLM_MODEL: str = "claude-sonnet-4-6"

# -- Cache ------------------------------------------------------------------
# Anchored to the project root (two levels above this package's src/ dir) so
# that a single _cache/ directory is used regardless of the working directory
# from which tests or scripts are launched.
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DEFAULT_CACHE_DIR: Path = _PROJECT_ROOT / "_cache"
TEST_CACHE_DIR: Path = _PROJECT_ROOT / "_cache_test"
CACHE_TTL: int = 5 * 86400  # 5 days in seconds

# -- Hardcoded FDA approvals (used during temporal holdouts) ----------------
# When the pipeline is run with --date-before, the live FDA-label lookup is
# replaced with a hardcoded {drug: [{disease, approved}]} table loaded from
# this JSON file. See PLAN_date_before.md for context.
DRUG_APPROVALS_PATH: Path = _PROJECT_ROOT / "drug_approvals.json"

# -- Open Targets -----------------------------------------------------------
OPEN_TARGETS_BASE_URL: str = "https://api.platform.opentargets.org/api/v4/graphql"

# -- ChEMBL -----------------------------------------------------------------
CHEMBL_BASE_URL: str = "https://www.ebi.ac.uk/chembl/api/data"

# -- ClinicalTrials.gov -----------------------------------------------------
CLINICAL_TRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_RECENT_START_YEAR: str = "2024"
# Top-N exemplar fetch cap for the pair-scoped trial query tools
# (search_trials / get_completed_trials / get_terminated_trials). Sorted by
# enrollment desc; counts that need the full population go through
# _count_trials_total (cheap countTotal API path) instead of a record fetch.
CLINICAL_TRIALS_FETCH_MAX: int = 50
# Per-source cache TTL for ClinicalTrials.gov pair-scoped queries
# (get_completed_trials / get_terminated_trials). Longer than the global
# CACHE_TTL because trial status transitions are slow.
CLINICAL_TRIALS_CACHE_TTL: int = 14 * 86400  # 14 days in seconds

# -- PubMed / NCBI ----------------------------------------------------------
NCBI_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL: str = f"{NCBI_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL: str = f"{NCBI_BASE_URL}/esummary.fcgi"

# Process-wide cap on concurrent in-flight requests to NCBI eutils. With a
# valid NCBI api_key the per-IP rate ceiling is 10 req/sec; 8 leaves headroom
# for retry/backoff traffic and for the MeSH resolver, which uses the same IP.
PUBMED_MAX_CONCURRENT_REQUESTS: int = 8

# Pre-call sleep on every uncached PubMed search.
PUBMED_SEARCH_SLEEP_SECONDS: float = 1.0

# -- MeSH resolver ----------------------------------------------------------
NCBI_ESEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
NCBI_ESUMMARY_URL: str = f"{NCBI_BASE_URL}/esummary.fcgi"
MESH_RESOLVER_TTL_SECONDS: int = 60 * 60 * 24 * 30  # 30 days
MESH_RESOLVER_MAX_CONCURRENT: int = 5

# -- Clinical stage ranking (Open Targets) ----------------------------------
# Maps maximumClinicalStage / maxClinicalStage string values to numeric ranks
# for comparison. Higher rank = further in pipeline.
CLINICAL_STAGE_RANK: dict[str, int] = {
    "UNKNOWN": 0,
    "PRECLINICAL": 1,
    "IND": 2,
    "EARLY_PHASE_1": 3,
    "PHASE_1": 4,
    "PHASE_1_2": 5,
    "PHASE_2": 6,
    "PHASE_2_3": 7,
    "PHASE_3": 8,
    "PREAPPROVAL": 9,
    "APPROVAL": 10,
}

# -- Interaction type mapping (Open Targets) --------------------------------
INTERACTION_TYPE_MAP: dict[str, str] = {
    "intact": "physical",
    "signor": "signalling",
    "reactome": "enzymatic",
    "string": "functional",
}

# -- Overly broad disease terms (used for filtering in competitors and normalization) --
BROADENING_BLOCKLIST: frozenset[str] = frozenset(
    {
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "neoplasm",
        "malignancy",
        "disease",
        "disorder",
        "syndrome",
        "indication",
        "pain",
    }
)

# -- Stop-reason keywords → category (ClinicalTrials.gov) ------------------
# Order matters: _classify_stop_reason takes the first match, so more specific
# phrases must appear before broader ones. Examples observed in real CT.gov
# stop-reason text that motivated each entry are noted inline.
STOP_KEYWORDS: dict[str, str] = {
    # -- Disambiguate "futility" variants (specific → general) ---------------
    # "enrollment futility" = operational futility, not efficacy
    "enrollment futility": "enrollment",
    "recruitment futility": "enrollment",
    # otherwise "futility" is an efficacy stop
    "futility": "efficacy",
    # -- Efficacy: explicit-phrase forms ------------------------------------
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "no significant difference": "efficacy",
    # seen on solanezumab Phase 3s; common phrasing for endpoint misses
    "did not meet the primary endpoint": "efficacy",
    "did not meet the study's primary endpoint": "efficacy",
    "did not meet the target efficacy": "efficacy",
    "primary endpoint was not met": "efficacy",
    "primary endpoint not met": "efficacy",
    "not meet its primary endpoint": "efficacy",
    # seen on metformin × diabetes (NCT02111096) and similar benefit/risk phrasing
    "benefit-risk profile did not support": "efficacy",
    "not support continued development": "efficacy",
    "insufficient scientific evidence": "efficacy",
    "meaningful benefit": "efficacy",
    "insufficient target engagement": "efficacy",
    # -- Safety: explicit-phrase forms (hepatic is the dominant real-world form)
    "liver safety": "safety",
    "hepatic safety": "safety",
    "liver enzyme": "safety",  # covers "liver enzymes", "liver enzyme elevations"
    "transaminase": "safety",  # covers "transaminases", "transaminase elevations"
    "hepatotoxicity": "safety",
    "nonclinical safety": "safety",
    "safety finding": "safety",  # covers "safety findings"
    # seen on atabecestat × AD (NCT02569398) — canonical hepatotoxicity stop
    "change in benefit-risk profile": "safety",
    "side effect": "safety",
    "adverse event": "safety",
    "toxicity": "safety",
    "safety concern": "safety",
    "safety signal": "safety",
    "clinical hold": "safety",
    # -- Broader terms (these fire if nothing above matched) -----------------
    "efficacy": "efficacy",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
}

NEGATION_PREFIXES: list[str] = ["no ", "not ", "unrelated to ", "without ", "non-"]

# -- Vaccine name keywords (for landscape competitor filtering) -------------
# Biologicals whose names match any of these substrings are classified as
# vaccines and excluded from the competitive landscape — they are not
# mechanism competitors for small-molecule or biologic drugs.
VACCINE_NAME_KEYWORDS: frozenset[str] = frozenset(
    {
        "vaccine",
        "vax",
        "immuniz",
        "immunis",
        "toxoid",
        "vacuna",
    }
)

MECHANISM_SIGNAL_KEYS: frozenset[str] = frozenset(
    {"genetic_association", "literature", "affected_pathway"}
)

# Drug action types that produce loss-of-function on the target.
LOF_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "INHIBITOR",
        "ANTAGONIST",
        "NEGATIVE ALLOSTERIC MODULATOR",
        "NEGATIVE MODULATOR",
        "BLOCKER",
    }
)

# Drug action types that produce gain-of-function on the target.
GOF_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "AGONIST",
        "ACTIVATOR",
        "POSITIVE ALLOSTERIC MODULATOR",
        "POSITIVE MODULATOR",
        "PARTIAL AGONIST",
    }
)

# Number of top POSITIVE repurposing candidates the mechanism agent surfaces.
MECHANISM_TOP_CANDIDATES: int = 5

# -- Supervisor top-N evidence gate -----------------------------------------
# A candidate is excluded from the supervisor's top-N ranking (summary +
# blurbs) when it has both zero clinical trials AND fewer than this many
# PubMed PMIDs. Investigated candidates that fail this gate still appear in
# disease_findings (per-disease Candidate Findings section) but are dropped
# from top_diseases and from the blurbs list.
SUPERVISOR_MIN_PMIDS_NO_TRIALS: int = 5

# -- openFDA ----------------------------------------------------------------
OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/label.json"
OPENFDA_LABEL_LIMIT: int = 5

# Curated per-drug list of candidate disease phrasings to short-circuit as
# FDA-approved (return True without calling the LLM). Acts strictly as an
# LLM backstop: only add candidate phrasings the LLM-against-label flow
# misses (narrower subsets, lay synonyms, unusual phrasings, label coverage
# gaps, etc.) for a given drug. When a candidate disease string EXACTLY
# matches (case-sensitive, no normalization) any string in the drug's list,
# the approval check short-circuits to True without calling the LLM.
# Candidates not in the list fall through to the LLM-against-label flow.
CURATED_FDA_APPROVED_CANDIDATES: dict[str, list[str]] = {
    # Semaglutide — morbid obesity is a clinical subset of approved obesity;
    # CKD lives in the FLOW indication, which the LLM reads ambiguously.
    "semaglutide": ["morbid obesity", "chronic kidney disease"],
    # Atorvastatin — CHD/CAD appear on the label as risk-reduction qualifier
    # populations; the prompt's risk-reduction rule rejects them.
    "atorvastatin": ["coronary heart disease", "coronary artery disease"],
    # Fluoxetine — PMDD is on a Sarafem-era label that doesn't surface in
    # the top-N generic fluoxetine labels openFDA returns. Bipolar I
    # depression is on the Symbyax combination label.
    "fluoxetine": ["premenstrual dysphoric disorder", "bipolar disorder"],
    "sarafem": ["premenstrual dysphoric disorder"],
    # Amoxicillin — pneumonia indication doesn't surface in the top-N labels.
    "amoxicillin": ["pneumonia"],
    # Levothyroxine — goiter approval doesn't surface in top-N labels.
    "levothyroxine": ["goiter"],
    # Rivaroxaban — COMPASS CAD/PAD risk-reduction; the label phrases CAD/PAD
    # as "in patients with CAD/PAD", which the LLM reads as qualifier-only.
    "rivaroxaban": ["coronary artery disease", "peripheral artery disease"],
    # Doxycycline — Lyme disease is on Vibramycin/Acticlate labels but not
    # the top-N generic doxycycline labels openFDA returns.
    "doxycycline": ["lyme disease"],
    # Metronidazole — giardiasis and pseudomembranous colitis (C. diff) are
    # on the full Flagyl label but not the top-N generic labels openFDA
    # returns (only the Flagyl 375 capsules trichomoniasis label appears).
    "metronidazole": ["giardiasis", "clostridium difficile infection"],
    # Empagliflozin — Jardiance label approves bare "heart failure" (covers
    # both HFrEF and HFpEF after EMPEROR-Preserved). The LLM treats
    # narrower-EF candidates as unsupported sub-indications and rejects.
    "empagliflozin": [
        "heart failure with reduced ejection fraction",
        "heart failure with preserved ejection fraction",
    ],
    # Aspirin — cardio-prevention and Kawasaki indications live on specific
    # aspirin product labels (Bayer Cardio, Ecotrin); openFDA's top-N
    # returns generic OTC pain-relief labels that lack these.
    "aspirin": [
        "myocardial infarction",
        "stroke",
        "rheumatoid arthritis",
        "osteoarthritis",
        "kawasaki disease",
    ],
    # Liraglutide — same morbid-obesity → obesity case as semaglutide.
    "liraglutide": ["morbid obesity"],
    # Acetaminophen — Tylenol Arthritis approves OA; openFDA top-N returns
    # generic pain/fever acetaminophen labels that don't mention OA.
    "acetaminophen": ["osteoarthritis"],
    # Ondansetron — Zofran's CINV/PONV/post-radiotherapy indications always
    # qualify nausea/vomiting; the LLM rejects the bare candidates as broad.
    "ondansetron": ["nausea", "vomiting"],
    # Ivermectin — Stromectol's scabies indication is on the topical/cream
    # formulations; generic ivermectin labels openFDA returns are
    # strongyloidiasis-only.
    "ivermectin": ["scabies"],
    # Risperidone — label approves "irritability associated with autistic
    # disorder"; bare "autistic disorder" reads as approved by clinical
    # convention. The LLM is overly strict on the population qualifier.
    "risperidone": ["autistic disorder"],
    # Baricitinib — JIA approval (Sept 2024) is a recent label expansion;
    # openFDA's top-N often returns older Olumiant labels that pre-date it.
    "baricitinib": ["juvenile idiopathic arthritis"],
    # Bupropion — smoking-cessation lives on Bupropion Hydrochloride SR
    # labels (the discontinued Zyban brand has been pulled from openFDA).
    # The SR labels share generic_name with XL/IR labels, so per-alias top-5
    # queries return MDD-only XL labels and miss SR.
    "bupropion": ["smoking cessation", "nicotine dependence"],
    # Mebendazole — pinworm (enterobiasis) is on Vermox/Emverm; openFDA's
    # top-N may return formulations that don't carry the indication text.
    "mebendazole": ["pinworm infection"],
    # Colchicine — pericarditis is standard-of-care globally (CORE/COPE/
    # ICAP/CORP trials) but no current US colchicine label carries the
    # indication; treated here as clinical-truth-over-openFDA-truth.
    "colchicine": ["pericarditis"],
    # Rituximab — WM and MCL fall under the broader CD20-positive B-cell NHL
    # umbrella in the Rituxan/Truxima/Riabni/Ruxience labels, but openFDA's
    # indications_and_usage snippets only enumerate FL and DLBCL by name and
    # never spell out "Waldenström" or "mantle cell". WM is also clinical
    # standard-of-care (NCCN) treated here as clinical-truth-over-openFDA-truth.
    "rituximab": ["waldenstrom macroglobulinemia", "mantle cell lymphoma"],
}


# Curated per-drug list of candidate disease phrasings to short-circuit as
# NOT FDA-approved (return False without calling the LLM). Mirrors
# CURATED_FDA_APPROVED_CANDIDATES but for the rejection direction. Use
# sparingly: only when the LLM over-matches because the label phrases an
# etiology, qualifier, or combination-product context as an indication that
# isn't actually a standalone approval. When a candidate string EXACTLY
# matches (case-sensitive, no normalization) any string in the drug's list,
# the approval check short-circuits to False without calling the LLM.
CURATED_FDA_REJECTED_CANDIDATES: dict[str, list[str]] = {
    # Acetaminophen labels list "the common cold" as a *cause* of "minor
    # aches and pains," not as a treatment indication. The LLM reads the
    # phrase as an approval.
    "acetaminophen": ["common cold"],
    # Lumakras has a CRC approval but only for KRAS G12C-mutated mCRC in
    # combination with panitumumab (~3-4% of mCRC patients). The bare
    # candidate "colorectal cancer" is too broad — the LLM matches on the
    # word's presence in the label without checking how narrow the subset is.
    "sotorasib": ["colorectal cancer"],
    # Semaglutide labels list "non-fatal stroke" as a component of the
    # MACE composite endpoint ("reduce the risk of major adverse cardiovascular
    # events (cardiovascular death, non-fatal myocardial infarction, or
    # non-fatal stroke)..."). The prompt's risk-reduction rule treats stroke
    # as a target, but clinically semaglutide is not a stroke drug — stroke
    # is just one component of the composite endpoint.
    "semaglutide": ["ischemic stroke"],
}

# -- Supervisor: mechanism-sourced candidate threshold -----------------------
# Minimum Open Targets overall_score for a mechanism-surfaced disease
# association to be promoted into the supervisor's investigation allowlist.
MECHANISM_ASSOCIATION_MIN_SCORE: float = 0.3

# -- API: frontend integration -----------------------------------------------
# Origins allowed by CORS — the Vite dev server (dev) and same-origin (prod).
CORS_ALLOW_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
# Built React bundle served by FastAPI in prod (Vite `npm run build` output).
FRONTEND_DIST_DIR: Path = _PROJECT_ROOT / "frontend" / "dist"
