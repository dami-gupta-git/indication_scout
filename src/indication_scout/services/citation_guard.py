"""Reject record identifiers that a model wrote into free text but was never given.

Every identifier LIST in a report is rebuilt in code from the records that were actually fetched, so
those fields cannot name a record that does not exist. The narrative fields are different: the model
types the digits itself and the renderer prints them verbatim. A single transposed digit there
(metformin x MASLD: key finding cited PMID 18721186; the retrieved paper is 18721166) reads as a real
citation and resolves to an unrelated record or to nothing.

This module extracts the identifiers a passage cites and compares them against the identifiers that
passage's prompt was given. Callers retry once, then drop the offending sentence — omission is the
acceptable direction, an unverifiable citation is not.
"""

import logging
import re

logger = logging.getLogger(__name__)

# "PMID: 12345678", "PMIDs 10634377, 10206447", "(PMID 123)". Only digits in an explicit PMID
# citation context are read — a bare 7-8 digit run in prose is far more often an enrollment count,
# a year range or a confidence bound.
_PMID_CITATION_RE = re.compile(r"PMIDs?\s*:?\s*((?:\d+[\s,;]*)+)", re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\d+")
_NCT_RE = re.compile(r"NCT\d+", re.IGNORECASE)

# Sentence boundary: terminator followed by whitespace. Over-splitting on an abbreviation only ever
# costs a fragment of a passage already known to carry a bad citation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_pmids(text: str) -> list[str]:
    """Every PMID cited in `text`, in order of appearance, deduplicated."""
    found: list[str] = []
    for group in _PMID_CITATION_RE.findall(text or ""):
        for pmid in _DIGIT_RUN_RE.findall(group):
            if pmid not in found:
                found.append(pmid)
    return found


def extract_nct_ids(text: str) -> list[str]:
    """Every NCT id cited in `text`, uppercased, in order of appearance, deduplicated."""
    found: list[str] = []
    for nct in _NCT_RE.findall(text or ""):
        upper = nct.upper()
        if upper not in found:
            found.append(upper)
    return found


def unknown_pmids(text: str, allowed: set[str]) -> list[str]:
    """PMIDs cited in `text` that are not in `allowed` (the PMIDs the prompt was given)."""
    return [pmid for pmid in extract_pmids(text) if pmid not in allowed]


def unknown_nct_ids(text: str, allowed: set[str]) -> list[str]:
    """NCT ids cited in `text` that are not in `allowed` (the ids the prompt was given)."""
    upper_allowed = {nct.upper() for nct in allowed}
    return [nct for nct in extract_nct_ids(text) if nct not in upper_allowed]


def strip_sentences_with_unknown_pmids(
    text: str, allowed: set[str], *, context: str
) -> str:
    """`text` with every sentence citing an unrecognized PMID removed."""
    return _strip_sentences(text, lambda s: unknown_pmids(s, allowed), context=context)


def strip_sentences_with_unknown_nct_ids(
    text: str, allowed: set[str], *, context: str
) -> str:
    """`text` with every sentence citing an unrecognized NCT id removed."""
    return _strip_sentences(
        text, lambda s: unknown_nct_ids(s, allowed), context=context
    )


def strip_findings_with_unknown_pmids(
    findings: list[str], allowed: set[str], *, context: str
) -> list[str]:
    """`findings` without any entry citing an unrecognized PMID. A finding is one claim, so a bad
    citation invalidates the whole entry rather than part of it."""
    kept: list[str] = []
    for finding in findings:
        bad = unknown_pmids(finding, allowed)
        if bad:
            logger.error(
                "citation_guard: dropping key finding citing unknown PMID(s) %s in %s: %s",
                ", ".join(bad),
                context,
                finding,
            )
            continue
        kept.append(finding)
    return kept


def _strip_sentences(text: str, find_unknown, *, context: str) -> str:
    if not text:
        return text
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        bad = find_unknown(sentence)
        if bad:
            logger.error(
                "citation_guard: dropping sentence citing unknown id(s) %s in %s: %s",
                ", ".join(bad),
                context,
                sentence,
            )
            continue
        kept.append(sentence)
    return " ".join(kept).strip()
