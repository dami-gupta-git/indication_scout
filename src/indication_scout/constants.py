"""Project-wide constants."""

from pathlib import Path

# -- Base client defaults ---------------------------------------------------
DEFAULT_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 3

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
CLINICAL_TRIALS_WHITESPACE_CONDITION_MAX: int = 500
CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER: str = "(PHASE2 OR PHASE3 OR PHASE4)"
CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS: int = 50
CLINICAL_TRIALS_RECENT_START_YEAR: str = "2024"

# -- PubMed / NCBI ----------------------------------------------------------
NCBI_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL: str = f"{NCBI_BASE_URL}/efetch.fcgi"

# -- Interaction type mapping (Open Targets) --------------------------------
INTERACTION_TYPE_MAP: dict[str, str] = {
    "intact": "physical",
    "signor": "signalling",
    "reactome": "enzymatic",
    "string": "functional",
}

# -- Stop-reason keywords → category (ClinicalTrials.gov) ------------------
STOP_KEYWORDS: dict[str, str] = {
    "efficacy": "efficacy",
    "futility": "efficacy",
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "safety": "safety",
    "adverse": "safety",
    "toxicity": "safety",
    "side effect": "safety",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
}
