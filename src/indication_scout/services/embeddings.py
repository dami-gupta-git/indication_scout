"""BioLORD-2023 embedding service.

BioLORD-2023 is a biomedical sentence embedding model trained on UMLS ontology,
SNOMED-CT, and ~400k GPT-3.5-generated biomedical definitions. It produces
768-dimensional vectors and achieves state-of-the-art on clinical sentence
similarity benchmarks (MedSTS, EHR-Rel-B).

We use it to embed PubMed abstracts at ingest time and queries at search time,
so that cosine similarity in pgvector retrieves semantically relevant papers
rather than just keyword matches.

The model is lazy-loaded on first call to embed() and reused for the lifetime
of the process — loading takes ~10s and uses ~500MB RAM, so we only do it once.

The model must be prefetched into the persistent HF cache (HF_HOME) before
serving — see scripts/prefetch_embedding_model.py. _get_model() always loads
with local_files_only=True and raises if the snapshot is absent, so analysis
never downloads at request time.
"""

import asyncio
import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from indication_scout.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton. None until the first call to embed().
_model: SentenceTransformer | None = None

# Serialises model init + encode(). Created lazily and rebound when the running
# event loop changes — an asyncio.Lock binds to the loop it is created on, so a
# cached one would raise "bound to a different event loop" on a second
# asyncio.run() (e.g. a second Streamlit "Analyse" click).
_MODEL_LOCK: asyncio.Lock | None = None
_MODEL_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _model_lock() -> asyncio.Lock:
    global _MODEL_LOCK, _MODEL_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _MODEL_LOCK is None or _MODEL_LOCK_LOOP is not loop:
        _MODEL_LOCK = asyncio.Lock()
        _MODEL_LOCK_LOOP = loop
    return _MODEL_LOCK


def _is_model_cached(model_name: str) -> bool:
    """Return True if the model snapshot directory exists in the HF cache.

    huggingface_hub stores downloaded models under:
        <HF_HOME>/hub/models--<org>--<name>/snapshots/

    The HF_HOME env var defaults to ~/.cache/huggingface when not set.
    We check for the presence of the snapshots directory (and that it is
    non-empty) rather than individual files so the check stays valid across
    model revisions.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    # Convert "FremyCompany/BioLORD-2023" -> "models--FremyCompany--BioLORD-2023"
    cache_dir_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / cache_dir_name / "snapshots"
    return snapshots_dir.is_dir() and any(snapshots_dir.iterdir())


def _get_model() -> SentenceTransformer:
    """Return the singleton model, instantiating it on first call.

    The model must be prefetched into the HF cache ahead of time (see
    scripts/prefetch_embedding_model.py). We always load with
    local_files_only=True so analysis never downloads at request time — if the
    cache is missing we fail loudly rather than silently pulling ~500MB.

    Not safe to call concurrently — use embed_async() which holds _model_lock
    across both model initialisation and encode().
    """
    global _model
    if _model is None:
        model_name = get_settings().embedding_model
        if not _is_model_cached(model_name):
            hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            raise RuntimeError(
                f"Embedding model {model_name} not found in HF cache (HF_HOME={hf_home}). "
                "Run scripts/prefetch_embedding_model.py to populate the cache before serving."
            )
        logger.info("Loading embedding model %s from local cache", model_name)
        _model = SentenceTransformer(model_name, local_files_only=True)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using BioLORD-2023.

    All texts are encoded in a single batch — callers should pass the full
    list rather than calling this in a loop, to avoid redundant model overhead.

    Args:
        texts: Texts to embed. For abstracts, use "<title>. <abstract text>".
               For queries, use the full therapeutic intent string
               (e.g. "Evidence for metformin as a treatment for colorectal cancer...").

    Returns:
        List of 768-dimensional embedding vectors, one per input text,
        in the same order as the input.
    """
    model = _get_model()
    # convert_to_numpy=True returns an ndarray; we convert to plain Python
    # floats so the vectors can be stored directly via SQLAlchemy/pgvector.
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


async def embed_async(texts: list[str]) -> list[list[float]]:
    """Async-safe wrapper around embed().

    Acquires _model_lock so that concurrent callers (e.g. parallel disease
    iterations in run_rag) do not race during model initialisation or encode().

    embed() is CPU-bound (model.encode blocks for the full batch), so it runs in
    a worker thread via asyncio.to_thread — otherwise it would freeze the event
    loop for the whole encode, stalling every other request (health checks,
    polling, concurrent analyses). The lock still serializes encodes (the model
    is not safe to run concurrently), but the loop stays free while waiting.

    Args:
        texts: Same as embed().

    Returns:
        Same as embed().
    """
    async with _model_lock():
        return await asyncio.to_thread(embed, texts)
