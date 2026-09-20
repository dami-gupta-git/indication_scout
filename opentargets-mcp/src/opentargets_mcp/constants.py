"""Row caps and output limits. Each cap keeps a tool's result small enough to read in a chat session."""

# Candidates returned by a name lookup: the match, plus enough alternatives to spot a wrong match.
RESOLVE_HITS = 5

# Association rows returned by default; the caller may ask for more, up to the maximum.
DEFAULT_ASSOCIATION_ROWS = 20
MAX_ASSOCIATION_ROWS = 100

# Drug rows. The API returns the full list with no paging argument, so this trims client-side.
DEFAULT_DRUG_ROWS = 25
MAX_DRUG_ROWS = 100

# Diseases listed against a single drug entry before the rest are counted rather than named.
DRUG_DISEASE_ROWS = 5

# Evidence rows for one target-disease pair.
DEFAULT_EVIDENCE_ROWS = 25
MAX_EVIDENCE_ROWS = 100

# Function descriptions run to several thousand characters; keep the opening of the first one.
FUNCTION_DESCRIPTION_CHARS = 600

# Literature ids listed per evidence row.
EVIDENCE_LITERATURE_IDS = 5

ENTITY_KINDS = ("target", "disease", "drug")
