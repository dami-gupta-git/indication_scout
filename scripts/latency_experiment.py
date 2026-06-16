"""Latency experiment: where does a cold full-pipeline run spend its time?

Runs the REAL supervisor pipeline (`run_analysis`) on a drug that has never been
examined locally, so PubMed fetch + BioLORD embedding are genuinely cold. While the
run executes, an in-process logging handler scrapes the `[TIMING]`/`[LLMTURN]` lines
the pipeline already emits and rolls them into the three buckets the user cares about:

    1. embedding        — BioLORD encode time (fetch_and_cache embed + every
                          semantic_search query embed)
    2. semantic_search  — per-disease pgvector scan + pubtype fetch + query embed
    3. agent loop       — supervisor ReAct orchestration (LLM + overhead remainder)

It also prints a FETCH-VOLUME / RELEVANCE audit: how many queries were issued per
disease, how many PMIDs each returned, how many were newly embedded, and how the
final top-k compares to the candidate pool — to judge whether the pipeline is
fetching more than it uses.

Usage:
    python scripts/latency_experiment.py                 # default cold drug
    python scripts/latency_experiment.py pioglitazone    # explicit drug

Pick a drug NOT present in cache/chembl_id_to_names and NOT in pgvector, or the
embed/fetch phases will be warm and the numbers will understate prod cost.
"""

import asyncio
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Mirror the CLI's env loading BEFORE importing anything that reads settings at
# import time (base_client calls get_settings() at import). .env carries the DB
# URL + API key; .env.constants carries the production tunables.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
_constants = os.environ.get("CONSTANTS_FILE", ".env.constants")
load_dotenv(_PROJECT_ROOT / _constants)
os.environ["CONSTANTS_FILE"] = str(_PROJECT_ROOT / _constants)

from indication_scout.markers import no_review  # noqa: E402

logger = logging.getLogger(__name__)

# A drug absent from cache/chembl_id_to_names, seed_examples, and snapshots as of
# 2026-06-16 — so this is a true cold run (real PubMed fetch + BioLORD embed).
DEFAULT_DRUG = "pioglitazone"


class TimingCollector(logging.Handler):
    """Scrape the pipeline's existing [TIMING]/[LLMTURN] log lines into buckets.

    Read-only — does not change pipeline behaviour. Attaches to the root logger so
    it sees the warnings emitted by retrieval.py, embeddings, and supervisor_agent.
    """

    # fetch_and_cache breakdown line (one per fetch_and_cache call, i.e. per disease):
    #   [TIMING] fetch_and_cache breakdown: search=.. fetch_abstracts=.. embed=..(N new)
    #   insert=.. | M total pmids, S stored
    _FETCH = re.compile(
        r"fetch_and_cache breakdown: search=([\d.]+)s fetch_abstracts=([\d.]+)s "
        r"embed=([\d.]+)s\((\d+) new\) insert=([\d.]+)s \| (\d+) total pmids, (\d+) stored"
    )
    # semantic_search sub-phase lines (one set per disease):
    _SS_EMBED = re.compile(r"semantic_search (.+?) embed_query: ([\d.]+)s")
    _SS_SCAN = re.compile(r"semantic_search (.+?) pgvector_scan: ([\d.]+)s \((\d+) pmids in\)")
    _SS_PT = re.compile(r"semantic_search (.+?) fetch_pubtypes: ([\d.]+)s \((\d+) candidates\)")
    # [RELEVANCE] line: how many fetched PMIDs actually had embeddings and competed.
    _SS_REL = re.compile(
        r"\[RELEVANCE\] semantic_search (.+?): (\d+) pmids fetched, (\d+) had embeddings, "
        r"rerank_cap=(\d+), top_k kept=(\d+)"
    )
    # run-level rollups already emitted by analysis_runner:
    _RUN_TOTAL = re.compile(r"run_analysis\(.+?\) total: ([\d.]+)s")
    _EMBED_TOTAL = re.compile(r"embedding \(BioLORD\) total: ([\d.]+)s .*? (\d+) encode calls")
    _API_TOTAL = re.compile(r"external API total: ([\d.]+)s")
    _LLM_REMAINDER = re.compile(r"LLM \+ overhead \(remainder\): ([\d.]+)s")
    _AGENT_LOOP = re.compile(r"supervisor: (\d+) turns, (\d+) total output tokens, agent loop ([\d.]+)s")

    def __init__(self) -> None:
        super().__init__()
        self.fetch_rows: list[dict] = []
        self.ss_by_disease: dict[str, dict] = defaultdict(dict)
        self.run_total: float | None = None
        self.embed_total: float | None = None
        self.embed_calls: int | None = None
        self.api_total: float | None = None
        self.llm_remainder: float | None = None
        self.agent_loop: float | None = None
        self.agent_turns: int | None = None
        self.agent_out_tokens: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()

        m = self._FETCH.search(msg)
        if m:
            self.fetch_rows.append(
                {
                    "search": float(m.group(1)),
                    "fetch": float(m.group(2)),
                    "embed": float(m.group(3)),
                    "new": int(m.group(4)),
                    "insert": float(m.group(5)),
                    "total_pmids": int(m.group(6)),
                    "stored": int(m.group(7)),
                }
            )
            return

        m = self._SS_EMBED.search(msg)
        if m:
            self.ss_by_disease[m.group(1)]["embed"] = float(m.group(2))
            return
        m = self._SS_SCAN.search(msg)
        if m:
            d = self.ss_by_disease[m.group(1)]
            d["scan"] = float(m.group(2))
            d["pmids_in"] = int(m.group(3))
            return
        m = self._SS_PT.search(msg)
        if m:
            d = self.ss_by_disease[m.group(1)]
            d["pubtypes"] = float(m.group(2))
            d["candidates"] = int(m.group(3))
            return
        m = self._SS_REL.search(msg)
        if m:
            d = self.ss_by_disease[m.group(1)]
            d["rel_fetched"] = int(m.group(2))
            d["rel_embedded"] = int(m.group(3))
            d["rel_cap"] = int(m.group(4))
            d["rel_topk"] = int(m.group(5))
            return

        for attr, rx, cast in (
            ("run_total", self._RUN_TOTAL, float),
            ("api_total", self._API_TOTAL, float),
            ("llm_remainder", self._LLM_REMAINDER, float),
        ):
            m = rx.search(msg)
            if m:
                setattr(self, attr, cast(m.group(1)))
                return

        m = self._EMBED_TOTAL.search(msg)
        if m:
            self.embed_total = float(m.group(1))
            self.embed_calls = int(m.group(2))
            return

        m = self._AGENT_LOOP.search(msg)
        if m:
            self.agent_turns = int(m.group(1))
            self.agent_out_tokens = int(m.group(2))
            self.agent_loop = float(m.group(3))
            return

    # -- reporting ----------------------------------------------------------

    def report(self, drug: str) -> str:
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(f"LATENCY EXPERIMENT — {drug} (cold run)")
        lines.append("=" * 78)

        # --- Three requested buckets -----------------------------------------
        ss_embed = sum(d.get("embed", 0.0) for d in self.ss_by_disease.values())
        ss_scan = sum(d.get("scan", 0.0) for d in self.ss_by_disease.values())
        ss_pt = sum(d.get("pubtypes", 0.0) for d in self.ss_by_disease.values())
        ss_total = ss_embed + ss_scan + ss_pt

        fetch_embed = sum(r["embed"] for r in self.fetch_rows)

        lines.append("")
        lines.append("PHASE TOTALS (the three you asked for)")
        lines.append("-" * 78)
        if self.embed_total is not None:
            lines.append(
                f"  1. EMBEDDING (BioLORD encode, whole run) : {self.embed_total:7.1f}s"
                f"  ({self.embed_calls} encode calls)"
            )
            lines.append(
                f"        of which fetch_and_cache embed      : {fetch_embed:7.1f}s"
            )
            lines.append(
                f"        of which semantic_search query embed: {ss_embed:7.1f}s"
            )
        else:
            lines.append("  1. EMBEDDING : (run-level [TIMING] line not captured)")
        lines.append(
            f"  2. SEMANTIC_SEARCH (scan+pubtypes+embed) : {ss_total:7.1f}s"
            f"  (scan={ss_scan:.1f}s pubtypes={ss_pt:.1f}s embed={ss_embed:.1f}s)"
        )
        if self.agent_loop is not None:
            lines.append(
                f"  3. AGENT LOOP (supervisor ReAct, WRAPS fan-out): {self.agent_loop:7.1f}s"
                f"  ({self.agent_turns} turns, {self.agent_out_tokens} out tokens)"
            )
            lines.append(
                "        ^ NOT orchestration-only: sub-agent fan-out (fetch+embed+search)"
            )
            lines.append(
                "          runs INSIDE this loop, so it ≈ run-total. See remainder for"
            )
            lines.append("          the supervisor's own LLM/overhead cost:")
        if self.llm_remainder is not None:
            lines.append(
                f"     supervisor-only LLM + overhead remainder: {self.llm_remainder:7.1f}s"
            )
        if self.api_total is not None:
            lines.append(
                f"     external API I/O (PubMed/CT/OT/FDA)   : {self.api_total:7.1f}s"
            )
        if self.run_total is not None:
            lines.append(f"     RUN TOTAL                            : {self.run_total:7.1f}s")

        # --- Fetch volume / relevance audit ----------------------------------
        lines.append("")
        lines.append("FETCH-VOLUME / RELEVANCE AUDIT (are we fetching too much?)")
        lines.append("-" * 78)
        if self.fetch_rows:
            total_pmids = sum(r["total_pmids"] for r in self.fetch_rows)
            total_new = sum(r["new"] for r in self.fetch_rows)
            lines.append(
                f"  fetch_and_cache calls (≈ diseases)       : {len(self.fetch_rows)}"
            )
            lines.append(
                f"  total PMIDs returned across all diseases  : {total_pmids}"
            )
            lines.append(
                f"  newly fetched + embedded (cold)           : {total_new}"
            )
            lines.append(
                f"  already in pgvector (reused)              : {total_pmids - total_new}"
            )
        lines.append("")
        lines.append("  Per-disease: fetched -> had embeddings -> kept (MEASURED via [RELEVANCE])")
        lines.append("  (fetched >> kept ⇒ abstracts fetched+embedded that never reach the report)")
        for disease, d in sorted(self.ss_by_disease.items()):
            fetched = d.get("rel_fetched", d.get("pmids_in", "?"))
            embedded = d.get("rel_embedded", "?")
            topk = d.get("rel_topk", "?")
            waste = (
                embedded - topk
                if isinstance(embedded, int) and isinstance(topk, int)
                else "?"
            )
            lines.append(
                f"    {disease[:38]:38s} fetched={str(fetched):>5s} "
                f"embedded={str(embedded):>5s} kept={str(topk):>2s} "
                f"unused_embedded={str(waste):>5s}"
            )
        lines.append("")
        lines.append(
            "  'unused_embedded' = abstracts that had embeddings and competed but did NOT"
        )
        lines.append(
            "  make top-k. This is the MEASURED over-fetch — not inferred from the cap."
        )
        lines.append(
            "  Cross-check against experiments_latency_2026-06-16.md organ-axis finding."
        )
        lines.append("=" * 78)
        return "\n".join(lines)


@no_review
async def main(drug: str) -> None:
    from indication_scout.services.analysis_runner import run_analysis

    collector = TimingCollector()
    collector.setLevel(logging.WARNING)
    logging.getLogger().addHandler(collector)

    logger.info("Cold latency run for %s — this hits live PubMed + cold BioLORD embed", drug)
    output, _report_md = await run_analysis(drug)

    logging.getLogger().removeHandler(collector)
    print(collector.report(drug))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    drug_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DRUG
    asyncio.run(main(drug_arg))
