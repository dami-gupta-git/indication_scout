from pathlib import Path

from indication_scout.services.disease_normalizer import normalize_for_pubmed

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def get_pubmed_query(drug_name, disease_name):
    """Build a PubMed search query for a drug-disease pair.

    Normalizes the disease name to a broader, PubMed-friendly term (e.g.
    "atopic eczema" → "eczema OR dermatitis") before combining with the drug name.
    """
    pubmed_diseases = await normalize_for_pubmed(disease_name, drug_name)
    queries = []
    for d in pubmed_diseases.split("OR"):
        queries.append(f"{d.strip()} AND {drug_name}")

    return queries
