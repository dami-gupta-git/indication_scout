"""Project-wide constants."""

from pathlib import Path

# -- Base client defaults ---------------------------------------------------
DEFAULT_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 3

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

# -- Open Targets -----------------------------------------------------------
OPEN_TARGETS_BASE_URL: str = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_PAGE_SIZE: int = 500

# -- ChEMBL -----------------------------------------------------------------
CHEMBL_BASE_URL: str = "https://www.ebi.ac.uk/chembl/api/data"

# -- ClinicalTrials.gov -----------------------------------------------------
CLINICAL_TRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_WHITESPACE_EXACT_MAX: int = 50
CLINICAL_TRIALS_WHITESPACE_INDICATION_MAX: int = 200
CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER: str = "(PHASE2 OR PHASE3 OR PHASE4)"
CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS: int = 50
CLINICAL_TRIALS_RECENT_START_YEAR: str = "2024"
CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS: int = 50
CLINICAL_TRIALS_TERMINATED_DRUG_PAGE_SIZE: int = 20

# -- PubMed / NCBI ----------------------------------------------------------
NCBI_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL: str = f"{NCBI_BASE_URL}/efetch.fcgi"
PUBMED_MAX_RESULTS: int = 200

# -- RAG pipeline concurrency -----------------------------------------------
RAG_LLM_CONCURRENCY: int = 4
RAG_PUBMED_CONCURRENCY: int = 3
RAG_DISEASE_CONCURRENCY: int = 4

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
        "pain"
    }
)

# -- Stop-reason keywords → category (ClinicalTrials.gov) ------------------
STOP_KEYWORDS: dict[str, str] = {
    # Specific phrases first
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "no significant difference": "efficacy",
    "side effect": "safety",
    "adverse event": "safety",
    "toxicity": "safety",
    "futility": "efficacy",
    # Then broader terms
    "efficacy": "efficacy",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
    # "safety" and "adverse" are too broad without negation handling
    "safety concern": "safety",
    "safety signal": "safety",
    "clinical hold": "safety",
}

NEGATION_PREFIXES: list[str] = ["no ", "not ", "unrelated to ", "without ", "non-"]
