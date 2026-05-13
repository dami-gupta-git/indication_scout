"""Application configuration.

Two env files are loaded in order:
  1. .env                — secrets, DB credentials, API keys, model names
  2. .env.constants      — tunable numeric limits (top-k, max results, etc.)

Later files override earlier ones; actual environment variables override both.

To swap the constants file at runtime:
    CONSTANTS_FILE=.env.constants.test pytest
    CONSTANTS_FILE=.env.constants.experiment scout find -d "metformin"
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str
    db_password: str
    # Separate DB used by integration tests — same Postgres instance as
    # database_url but pointing to scout_test. Set in .env for local dev;
    # must be migrated separately before running integration tests.
    test_database_url: str | None = None

    # API Keys
    openai_api_key: str = ""
    pubmed_api_key: str = ""
    anthropic_api_key: str = ""
    ncbi_api_key: str = ""
    openfda_api_key: str = ""
    wandb_api_key: str = ""

    # LLM Settings
    llm_model: str = "claude-sonnet-4-6"
    small_llm_model: str = "claude-haiku-4-5-20251001"
    big_llm_model: str = "claude-opus-4-6"
    embedding_model: str = "FremyCompany/BioLORD-2023"

    # App Settings
    debug: bool = False
    log_level: str = "INFO"

    # -- Tunable limits (values come from .env.constants) ----------------------
    # No defaults here — if .env.constants is missing a field, startup fails
    # immediately so you know what to add.

    # LLM token limits
    llm_max_tokens: int
    small_llm_max_tokens: int

    # Base client
    default_timeout: float
    default_max_retries: int

    # Literature / RAG
    literature_top_k: int
    semantic_search_top_k: int
    pubmed_max_results: int
    pubmed_search_default_max_results: int
    pubmed_esummary_batch_size: int
    pubmed_efetch_batch_size: int
    rag_llm_concurrency: int
    rag_pubmed_concurrency: int
    rag_disease_concurrency: int

    # Clinical trials
    clinical_trials_landscape_max_trials: int
    clinical_trials_cap: int

    # Supervisor
    supervisor_candidate_cap: int

    # Mechanism
    mechanism_signal_threshold: float
    mechanism_associations_cap: int

    # Disease helper
    disease_pubmed_min_results: int

    # Open Targets
    open_targets_page_size: int
    open_targets_competitor_prefetch_max: int
    open_targets_association_min_score: float

    class Config:
        env_file = (
            ".env",
            os.environ.get("CONSTANTS_FILE", ".env.constants"),
        )
        env_file_encoding = "utf-8"
        frozen = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
