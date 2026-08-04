"""Unit tests for the Europe PMC article model."""

from indication_scout.models.model_europe_pmc import EuropePMCArticle

# Trimmed from a live resultType=core search result (sildenafil, 2026-08-03).
CORE_RESULT = {
    "id": "42206148",
    "source": "MED",
    "pmid": "42206148",
    "doi": "10.1016/j.jmccpl.2026.100853",
    "title": "Acute sildenafil administration decreases the susceptibility to induce arrhythmias.",
    "abstractText": "Sildenafil is a phosphodiesterase-5 inhibitor.",
    "journalTitle": None,
    "journalInfo": {
        "journal": {"title": "Journal of molecular and cellular cardiology plus"}
    },
    # pubYear is the issue year and firstPublicationDate the online date; they disagree on 16.1%
    # of a measured pool, so this fixture keeps them apart to pin which one the model reads.
    "pubYear": "2026",
    "firstPublicationDate": "2025-11-05",
    "pubType": None,
    "pubTypeList": {"pubType": ["brief-report", "Journal Article"]},
    "citedByCount": 0,
    "isOpenAccess": "Y",
}


def test_from_search_result_core():
    """Journal comes from journalInfo, pub types from pubTypeList; flat fields are null in core."""
    article = EuropePMCArticle.from_search_result(CORE_RESULT)

    assert article.source == "MED"
    assert article.record_id == "42206148"
    assert article.pmid == "42206148"
    assert article.doi == "10.1016/j.jmccpl.2026.100853"
    assert (
        article.title
        == "Acute sildenafil administration decreases the susceptibility to induce arrhythmias."
    )
    assert article.abstract == "Sildenafil is a phosphodiesterase-5 inhibitor."
    assert article.journal == "Journal of molecular and cellular cardiology plus"
    assert article.first_publication_date == "2025-11-05"
    # From firstPublicationDate, not the 2026 pubYear.
    assert article.pub_year == 2025
    assert article.pub_types == ["brief-report", "Journal Article"]
    assert article.cited_by_count == 0
    assert article.is_open_access is True
    assert article.article_key == "MED:42206148"


def test_from_search_result_preprint():
    """Preprint shape: null pmid and journalInfo, which 9/100 and 10/100 of a live sample carry."""
    article = EuropePMCArticle.from_search_result(
        {
            "id": "PPR123456",
            "source": "PPR",
            "pmid": None,
            "doi": "10.1101/2021.01.01.425001",
            "title": "A preprint about sildenafil.",
            "abstractText": "Sildenafil was administered.",
            "journalInfo": None,
            "pubYear": "2021",
            "firstPublicationDate": "2021-01-04",
            "pubTypeList": {"pubType": ["Preprint"]},
            "citedByCount": 0,
            "isOpenAccess": "N",
        }
    )

    assert article.source == "PPR"
    assert article.record_id == "PPR123456"
    assert article.pmid is None
    assert article.doi == "10.1101/2021.01.01.425001"
    assert article.title == "A preprint about sildenafil."
    assert article.abstract == "Sildenafil was administered."
    assert article.journal is None
    assert article.first_publication_date == "2021-01-04"
    assert article.pub_year == 2021
    assert article.pub_types == ["Preprint"]
    assert article.cited_by_count == 0
    assert article.is_open_access is False
    assert article.article_key == "PPR:PPR123456"


# PMC9359323 as returned live: pubYear is null and journalInfo.yearOfPublication is 0, but
# firstPublicationDate is valid — the case that makes pubYear unusable as the year source.
NULL_PUBYEAR_RESULT = {
    "id": "PMC9359323",
    "source": "PMC",
    "pmid": None,
    "pmcid": "PMC9359323",
    "doi": None,
    "title": (
        "Development and validation of an in vitro-in vivo correlation (IVIVC) model for "
        "propranolol hydrochloride extended-release matrix formulations"
    ),
    "abstractText": "The objective of this study was to develop an IVIVC model.",
    "journalInfo": {
        "issue": "2",
        "volume": "22",
        "yearOfPublication": 0,
        "monthOfPublication": 0,
        "journal": {"title": "Journal of food and drug analysis"},
    },
    "pubYear": None,
    "firstPublicationDate": "2022-08-15",
    "pubTypeList": {"pubType": ["research-article", "Journal Article"]},
    "citedByCount": 0,
    "isOpenAccess": "Y",
}


def test_from_search_result_null_pubyear():
    """A record with no pubYear still gets a year, taken from firstPublicationDate."""
    article = EuropePMCArticle.from_search_result(NULL_PUBYEAR_RESULT)

    assert article.source == "PMC"
    assert article.record_id == "PMC9359323"
    assert article.pmid is None
    assert article.doi is None
    assert article.journal == "Journal of food and drug analysis"
    assert article.first_publication_date == "2022-08-15"
    assert article.pub_year == 2022
    assert article.pub_types == ["research-article", "Journal Article"]
    assert article.cited_by_count == 0
    assert article.is_open_access is True
    assert article.article_key == "PMC:PMC9359323"
