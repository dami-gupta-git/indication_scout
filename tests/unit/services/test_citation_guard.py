"""Unit tests for the free-text identifier guard."""

import pytest

from indication_scout.services.citation_guard import (
    extract_nct_ids,
    extract_pmids,
    strip_findings_with_unknown_pmids,
    strip_sentences_with_unknown_nct_ids,
    strip_sentences_with_unknown_pmids,
    unknown_nct_ids,
    unknown_pmids,
)

# The live failure this guard was written for: the retrieved paper is 18721166, the key finding
# cited 18721186.
_MASLD_FINDING = (
    "A randomized controlled trial in 50 obese insulin-resistant adolescents found that "
    "metformin plus lifestyle intervention significantly reduced fatty liver prevalence "
    "(p < 0.04) and severity (p < 0.04) compared to placebo (PMID: 18721186)."
)
_MASLD_POOL = {"18721166", "19811343", "41828627", "20179669", "24304731"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("no citation here", []),
        ("a result (PMID: 18721166).", ["18721166"]),
        ("two of them (PMIDs 10634377, 10206447)", ["10634377", "10206447"]),
        ("repeated (PMID 123) and again (PMID: 123)", ["123"]),
        ("enrollment of 3649 patients over 2022 with no PMID", []),
    ],
)
def test_extract_pmids(text, expected):
    assert extract_pmids(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("NCT06033131 is recruiting", ["NCT06033131"]),
        ("two trials NCT06033131 and nct06452498", ["NCT06033131", "NCT06452498"]),
        ("no trial ids here", []),
    ],
)
def test_extract_nct_ids(text, expected):
    assert extract_nct_ids(text) == expected


def test_unknown_pmids_flags_the_masld_typo():
    assert unknown_pmids(_MASLD_FINDING, _MASLD_POOL) == ["18721186"]


def test_unknown_pmids_accepts_the_real_identifier():
    good = _MASLD_FINDING.replace("18721186", "18721166")
    assert unknown_pmids(good, _MASLD_POOL) == []


def test_unknown_nct_ids_flags_only_the_unsupplied_id():
    text = "NCT06033131 is recruiting and NCT99999999 is not."
    assert unknown_nct_ids(text, {"NCT06033131"}) == ["NCT99999999"]


def test_strip_findings_drops_only_the_offending_entry():
    good = _MASLD_FINDING.replace("18721186", "18721166")
    kept = strip_findings_with_unknown_pmids(
        [good, _MASLD_FINDING], _MASLD_POOL, context="test"
    )
    assert kept == [good]


def test_strip_sentences_keeps_the_verifiable_sentence():
    text = (
        "Metformin did not improve steatosis (PMID: 19811343). "
        "A pediatric trial was positive (PMID: 18721186)."
    )
    assert strip_sentences_with_unknown_pmids(text, _MASLD_POOL, context="test") == (
        "Metformin did not improve steatosis (PMID: 19811343)."
    )


def test_strip_sentences_removes_an_unsupplied_trial():
    text = "NCT06033131 is recruiting. NCT99999999 has completed."
    assert (
        strip_sentences_with_unknown_nct_ids(text, {"NCT06033131"}, context="test")
        == "NCT06033131 is recruiting."
    )
