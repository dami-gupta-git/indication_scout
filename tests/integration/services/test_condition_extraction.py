"""Integration tests for Europe PMC condition extraction against the real LLM.

The leakage tests are the point of this file. The small model knows every holdout answer from
pretraining — that sildenafil treats pulmonary hypertension, that duloxetine treats generalized
anxiety disorder — so an extraction that draws on its own knowledge rather than the supplied
abstract would recover the post-cutoff indication and invalidate any holdout run built on it.

Negatives are pinned by article key and were selected by matching the indication's full synonym
set, not one keyword: an earlier keyword-only pass mislabeled a minoxidil paper titled "male
pattern baldness" as a negative because it did not contain the word "alopecia".
"""

import re

import pytest

from indication_scout.services.condition_extraction import extract_conditions

# (article_key, drug) pairs whose title and abstract contain no phrasing of the eventual
# indication, verified against the synonym patterns below across the holdout pool.
SILDENAFIL_NEGATIVES = ["MED:12820988", "MED:12735784", "MED:12152115"]
DULOXETINE_NEGATIVES = ["MED:17191749"]
MINOXIDIL_NEGATIVES = ["MED:3480782", "MED:2826267"]

# Full synonym set per indication — what must NOT appear in the extracted conditions.
LEAK_PATTERNS = {
    "sildenafil": r"pulmonary (arterial )?hypertension|\bPAH\b|pulmonary vascular",
    "duloxetine": r"generali[sz]ed anxiety|\bGAD\b|anxiety disorder",
    "minoxidil": r"alopecia|baldness|hair (loss|growth|regrow)|hirsut|hypertrichos",
}


async def _articles_by_key(client, drug, date_before, keys):
    """Fetch the holdout pool and return the pinned articles, preserving the requested order."""
    pool = await client.search_by_drug(drug, date_before=date_before)
    by_key = {a.article_key: a for a in pool}
    missing = [k for k in keys if k not in by_key]
    assert not missing, f"pinned articles absent from the {drug} pool: {missing}"
    return [by_key[k] for k in keys]


@pytest.mark.parametrize(
    "drug, cutoff_year, keys",
    [
        ("sildenafil", 2004, SILDENAFIL_NEGATIVES),
        ("duloxetine", 2007, DULOXETINE_NEGATIVES),
        ("minoxidil", 1988, MINOXIDIL_NEGATIVES),
    ],
)
async def test_extraction_does_not_leak_post_cutoff_indication(
    europe_pmc_client, drug, cutoff_year, keys
):
    """No phrasing of the eventual indication is emitted for an abstract that does not name it."""
    from datetime import date

    articles = await _articles_by_key(
        europe_pmc_client, drug, date(cutoff_year, 1, 1), keys
    )
    result = await extract_conditions(drug, articles)

    assert result.skipped == 0
    assert len(result.articles) == len(keys)

    pattern = re.compile(LEAK_PATTERNS[drug], re.I)
    leaked = [
        (a.article_key, c)
        for a in result.articles
        for c in a.conditions
        if pattern.search(c)
    ]
    assert not leaked, f"{drug}: extraction leaked the post-cutoff indication: {leaked}"


async def test_extraction_finds_indication_when_the_abstract_states_it(
    europe_pmc_client,
):
    """The counterpart to the leakage tests: when the text does name the indication, it is
    extracted. Without this, a model that always returned NONE would pass the leak checks."""
    from datetime import date

    pool = await europe_pmc_client.search_by_drug(
        "sildenafil", date_before=date(2004, 1, 1)
    )
    pattern = re.compile(LEAK_PATTERNS["sildenafil"], re.I)
    positives = [
        a for a in pool if pattern.search(a.title) and "sildenafil" in a.title.lower()
    ][:5]
    assert positives, "no pre-2004 sildenafil paper names pulmonary hypertension in its title"

    result = await extract_conditions("sildenafil", positives)
    hits = [
        a.article_key
        for a in result.articles
        if any(pattern.search(c) for c in a.conditions)
    ]
    assert len(hits) >= 3, (
        f"only {len(hits)} of {len(positives)} title-explicit papers yielded the indication"
    )


async def test_extraction_empty_pool(europe_pmc_client):
    result = await extract_conditions("sildenafil", [])
    assert result.articles == []
    assert result.skipped == 0
