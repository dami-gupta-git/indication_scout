"""Extract stated treated-conditions from Europe PMC abstracts.

One LLM call per article, so a failure affects a single article and the cache is keyed on the
article identifier — a re-run over an expanded pool only pays for articles it has not seen.

The prompt confines the model to the supplied abstract and forbids drawing on its own knowledge of
the drug. Without that, a holdout run recovers the post-cutoff indication from the model rather
than from the literature. See design_europe_pmc.md.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel

from indication_scout.config import get_settings
from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    EUROPE_PMC_EXTRACTION_MAX_TOKENS,
    EUROPE_PMC_EXTRACTION_NS,
    EUROPE_PMC_EXTRACTION_SYSTEM,
    EUROPE_PMC_MAX_CONDITION_WORDS,
)
from indication_scout.models.model_europe_pmc import EuropePMCArticle
from indication_scout.services.llm import query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_NONE_TOKEN = "NONE"


class ArticleConditions(BaseModel):
    """Conditions one article states the drug treats. Empty list means the model returned NONE."""

    article_key: str
    conditions: list[str] = []


class ExtractionResult(BaseModel):
    """Outcome of an extraction pass over a pool.

    ``skipped`` counts articles whose extraction call failed after retries. Those articles
    contribute no conditions and are not silently treated as NONE — a failed read is not evidence
    that a paper names no condition.
    """

    articles: list[ArticleConditions] = []
    skipped: int = 0


def build_prompt(drug: str, article: EuropePMCArticle) -> str:
    """Render the extraction prompt for one article."""
    text = f"{article.title}\n\n{article.abstract}"
    return (
        (_PROMPTS_DIR / "extract_treated_conditions.txt")
        .read_text()
        .format(drug=drug, text=text)
    )


def parse_response(response: str) -> list[str]:
    """Parse the model's reply into condition names.

    One condition per line, NONE for nothing. Lines longer than
    ``EUROPE_PMC_MAX_CONDITION_WORDS`` are discarded: the model occasionally explains its reasoning
    instead of answering, and a sentence is not a condition name. Order-preserving dedup, since the
    same condition often appears in both title and abstract.
    """
    conditions: list[str] = []
    for line in response.strip().splitlines():
        cleaned = line.strip().strip("-•* ").strip()
        if not cleaned or cleaned.upper() == _NONE_TOKEN:
            continue
        if len(cleaned.split()) > EUROPE_PMC_MAX_CONDITION_WORDS:
            logger.warning("extraction: discarding non-condition line %r", cleaned[:80])
            continue
        conditions.append(cleaned.lower())
    return list(dict.fromkeys(conditions))


async def _extract_one(
    drug: str, article: EuropePMCArticle, sem: asyncio.Semaphore
) -> ArticleConditions | None:
    """Extract one article's conditions, or None if the LLM call failed."""
    settings = get_settings()
    cache_params = {
        "article_key": article.article_key,
        "drug": drug,
        "small_llm_model": settings.small_llm_model,
    }
    cached = cache_get(EUROPE_PMC_EXTRACTION_NS, cache_params, DEFAULT_CACHE_DIR)
    if cached is not None:
        return ArticleConditions(article_key=article.article_key, conditions=cached)

    async with sem:
        try:
            response = await query_small_llm(
                build_prompt(drug, article),
                system=EUROPE_PMC_EXTRACTION_SYSTEM,
                max_tokens=EUROPE_PMC_EXTRACTION_MAX_TOKENS,
            )
        except Exception as e:
            logger.warning(
                "extraction failed for %s (%s); article skipped", article.article_key, e
            )
            return None

    conditions = parse_response(response)
    cache_set(EUROPE_PMC_EXTRACTION_NS, cache_params, conditions, DEFAULT_CACHE_DIR)
    return ArticleConditions(article_key=article.article_key, conditions=conditions)


async def extract_conditions(
    drug: str, articles: list[EuropePMCArticle]
) -> ExtractionResult:
    """Extract stated treated-conditions from every article in the pool.

    Concurrency is bounded by ``europe_pmc_extraction_concurrency``. Articles whose extraction
    failed are counted in ``skipped`` rather than raising: a pool runs to thousands of articles and
    a handful of failures should not discard the rest.
    """
    if not articles:
        return ExtractionResult()

    sem = asyncio.Semaphore(get_settings().europe_pmc_extraction_concurrency)
    results = await asyncio.gather(*[_extract_one(drug, a, sem) for a in articles])

    extracted = [r for r in results if r is not None]
    skipped = len(results) - len(extracted)
    if skipped:
        logger.warning(
            "extraction: %s of %s articles skipped after failure", skipped, len(results)
        )
    logger.info(
        "extraction: %s articles, %s with conditions, %s skipped",
        len(articles),
        sum(1 for r in extracted if r.conditions),
        skipped,
    )
    return ExtractionResult(articles=extracted, skipped=skipped)
