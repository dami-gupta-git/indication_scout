"""
Probe: read PubMed abstracts for a drug-disease pair and ask an LLM to identify
safety signals.

This deliberately avoids embeddings and vector search. The point is to find out
what a language model can see in the raw text that BioLORD cosine similarity
could not surface.

Usage:
    python -m indication_scout.ml_models.safety_signal_probe \
        --drug metformin --disease "prostate cancer"

    python -m indication_scout.ml_models.safety_signal_probe \
        --drug thalidomide --disease "Crohn disease" --n 15
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Load both env files before importing any indication_scout modules so that
# ANTHROPIC_API_KEY and all required Settings fields are present when modules
# are imported and Settings is instantiated.
_root = Path(__file__).parents[3]
load_dotenv(_root / ".env", override=True)
load_dotenv(_root / ".env.constants", override=False)

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.services.disease_helper import normalize_for_pubmed, resolve_mesh_id
from indication_scout.services.llm import query_llm

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

_SYSTEM = (
    "You are a pharmacovigilance scientist. Be concise. Do not speculate beyond what the abstracts say."
)

_PROMPT_TEMPLATE = """\
Drug: {drug} | Disease: {disease}

Read these {n} abstracts and report any safety signals, prioritised by severity and strength \
of evidence. Report the most dangerous or best-evidenced signals first. Include signals even \
if the abstract only mentions them briefly — a phrase like "significant toxicity", \
"poorly tolerated", or "early closure due to adverse events" is itself a signal worth reporting.

For each signal write:

**<Signal name>**
One sentence on what the abstract says, including any limitations in the reported detail.
Evidence: PMID(s) and study type.
Severity: CRITICAL / SERIOUS / MODERATE / MILD — one sentence justifying the rating based on \
reversibility, frequency, and whether the event was life-threatening or led to discontinuation.

Only write a section when a signal is present. Do not write sections to report the absence of a signal. \
If truly no safety information appears in any abstract, say so in one line at the end.

---

{abstracts}
"""


async def _search_pubmed(query: str, n: int, date_before: date | None) -> list[str]:
    """Hit NCBI eutils esearch directly — no cache."""
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": n,
        "retmode": "json",
        "sort": "relevance",
    }
    if date_before:
        maxdate = (date_before - timedelta(days=1)).strftime("%Y/%m/%d")
        params.update({"datetype": "pdat", "mindate": "1900/01/01", "maxdate": maxdate})

    async with aiohttp.ClientSession() as session:
        async with session.get(_ESEARCH_URL, params=params) as resp:
            data = await resp.json(content_type=None)

    return data.get("esearchresult", {}).get("idlist", [])


def _format_abstracts(abstracts: list[PubmedAbstract]) -> str:
    parts = []
    for a in abstracts:
        title = a.title or "(no title)"
        body = a.abstract or "(no abstract text)"
        pub_date = a.pub_date or "date unknown"
        parts.append(f"PMID {a.pmid} ({pub_date})\nTitle: {title}\n\n{body}")
    return "\n\n---\n\n".join(parts)


async def run(drug: str, disease: str | None, n: int, date_before: date | None) -> None:
    if disease:
        mesh = await resolve_mesh_id(disease)
        if mesh:
            disease_term = f'"{mesh[1]}"[MeSH Terms]'
            sys.stdout.write(f"Disease resolved: {disease!r} → MeSH: {mesh[1]!r}\n\n")
        else:
            disease_term = await normalize_for_pubmed(disease, drug_name=drug)
            sys.stdout.write(f"Disease expanded: {disease!r} → {disease_term!r}\n\n")
        query = f"{drug} AND ({disease_term})"
    else:
        query = drug

    pmids = await _search_pubmed(query, n, date_before)
    if not pmids:
        sys.stdout.write(f"No PubMed results for: {query!r}\n")
        return

    date_label = f" (before {date_before})" if date_before else ""
    sys.stdout.write(f"Found {len(pmids)} PMIDs{date_label}: {', '.join(pmids)}\n\n")

    async with PubMedClient(cache_dir=DEFAULT_CACHE_DIR) as client:
        abstracts = await client.fetch_abstracts(pmids)

    abstracts_with_text = [a for a in abstracts if a.abstract]
    if not abstracts_with_text:
        sys.stdout.write("No abstracts with text returned. Cannot proceed.\n")
        return

    sys.stdout.write(
        f"Sending {len(abstracts_with_text)} abstracts to LLM for safety signal analysis...\n\n"
    )

    formatted = _format_abstracts(abstracts_with_text)
    prompt = _PROMPT_TEMPLATE.format(
        drug=drug,
        disease=disease or "any indication",
        n=len(abstracts_with_text),
        abstracts=formatted,
    )

    summary = await query_llm(prompt, system=_SYSTEM)

    sys.stdout.write("=" * 70 + "\n")
    disease_label = disease.upper() if disease else "ALL INDICATIONS"
    sys.stdout.write(f"SAFETY SIGNAL SUMMARY: {drug.upper()} / {disease_label}\n")
    sys.stdout.write("=" * 70 + "\n\n")
    sys.stdout.write(summary)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM safety signal probe from PubMed abstracts")
    parser.add_argument("--drug", required=True, help="Drug name (e.g. metformin)")
    parser.add_argument("--disease", default=None, help="Disease name (optional — omit to search full drug literature)")
    parser.add_argument(
        "--n", type=int, default=10, help="Abstracts to fetch (default 10)"
    )
    parser.add_argument(
        "--date-before", default=None, metavar="YYYY-MM-DD",
        help="Only include papers published before this date (e.g. 2020-01-01)"
    )
    args = parser.parse_args()

    date_before = date.fromisoformat(args.date_before) if args.date_before else None
    asyncio.run(run(args.drug, args.disease, args.n, date_before))


if __name__ == "__main__":
    main()
