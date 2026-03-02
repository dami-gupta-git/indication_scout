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
