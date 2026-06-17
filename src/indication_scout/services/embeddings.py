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

The model loads from the persistent HF cache (HF_HOME) when present. On a cold
cache (e.g. a fresh volume) the first embed() call downloads the snapshot
(~500MB) into HF_HOME and reuses it thereafter; later container boots find the
populated volume. Optionally prefetch ahead of time — see
scripts/prefetch_embedding_model.py.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from indication_scout.config import get_settings
from indication_scout.constants import EMBED_LOCK_CHUNK_SIZE

logger = logging.getLogger(__name__)

# Module-level singleton. None until the first call to embed().
_model: SentenceTransformer | None = None


def _cgroup_cpu_quota() -> int | None:
    """Return the container's CPU quota in whole vCPUs from the cgroup v2 limit, or None.

    cgroup v2 exposes the limit at /sys/fs/cgroup/cpu.max as "<quota> <period>" (e.g.
    "2400000 100000" = 24 vCPU). Returns None when the file is missing, set to "max"
    (unlimited), or unparseable — caller then leaves torch's default thread count alone.
    No fabricated value: an unknown quota stays unknown.
    """
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except OSError:
        return None
    if len(raw) != 2 or raw[0] == "max":
        return None
    try:
        quota, period = int(raw[0]), int(raw[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def _pin_torch_threads() -> None:
    """Cap torch's intra-op thread count to the cgroup CPU quota to avoid oversubscription.

    In a container, torch defaults its thread count to os.cpu_count(), which reports the HOST
    cores (e.g. 48) — not the cgroup quota (e.g. 24). The extra threads contend for CPU the
    cgroup won't grant, causing scheduler thrash that slows encode. Pin to the quota when known.
    Only ever reduces the count; a no-op when the quota is unknown or already >= cpu_count.
    """
    import torch

    quota = _cgroup_cpu_quota()
    if quota is None:
        return
    if quota < torch.get_num_threads():
        torch.set_num_threads(quota)
        logger.info(
            "Pinned torch threads to cgroup quota: %d (was reporting %d cores)",
            quota,
            os.cpu_count(),
        )

# Cumulative embedding timing → [call_count, total_seconds]. Measures wall-clock spent
# inside model.encode() (CPU-bound BioLORD inference), summed across all embed() calls.
# Read/reset via embed_timing_snapshot() / reset_embed_timing() to attribute how much of
# a run is local embedding vs. external API vs. Claude inference.
_EMBED_TIMING: list[float] = [0.0, 0.0]


def reset_embed_timing() -> None:
    """Clear the cumulative embedding-timing accumulator (call at the start of a run)."""
    _EMBED_TIMING[0] = 0.0
    _EMBED_TIMING[1] = 0.0


def embed_timing_snapshot() -> tuple[int, float]:
    """Return (call_count, total_seconds) of model.encode() time so far."""
    return int(_EMBED_TIMING[0]), _EMBED_TIMING[1]

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

    On the first call we load from the HF cache if present; if the cache is
    cold (e.g. a fresh volume) we download the snapshot (~500MB) into HF_HOME
    on that first call and reuse it for the process lifetime. Subsequent
    container boots find the populated volume and load locally.

    Not safe to call concurrently — use embed_async() which holds _model_lock
    across both model initialisation and encode().
    """
    global _model
    if _model is None:
        _pin_torch_threads()
        model_name = get_settings().embedding_model
        cached = _is_model_cached(model_name)
        if cached:
            logger.info("Loading embedding model %s from local cache", model_name)
        else:
            logger.info(
                "Embedding model %s not cached; downloading into HF cache", model_name
            )
        _model = SentenceTransformer(model_name, local_files_only=cached)
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
    _t0 = time.perf_counter()
    vectors = model.encode(texts, convert_to_numpy=True)
    _EMBED_TIMING[0] += 1
    _EMBED_TIMING[1] += time.perf_counter() - _t0
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

    Head-of-line fairness: a large batch is embedded in chunks of
    EMBED_LOCK_CHUNK_SIZE, releasing and re-acquiring the lock between chunks.
    This lets a concurrent caller's small query-embed slip in between chunks
    instead of waiting for a 700-abstract batch to finish (the `embed_query:
    130s` stall). Total encode compute is unchanged — only the interleaving is
    fairer. Vectors are concatenated in input order so the contract is identical
    to a single encode() call.

    Args:
        texts: Same as embed().

    Returns:
        Same as embed().
    """
    if not texts:
        return []

    chunk_size = EMBED_LOCK_CHUNK_SIZE
    vectors: list[list[float]] = []
    total_wait = 0.0
    total_enc = 0.0
    n_chunks = 0
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        # Acquire the lock per chunk so a waiting small embed can interleave.
        _t_wait = time.perf_counter()
        async with _model_lock():
            total_wait += time.perf_counter() - _t_wait
            _t_enc = time.perf_counter()
            chunk_vectors = await asyncio.to_thread(embed, chunk)
            total_enc += time.perf_counter() - _t_enc
        vectors.extend(chunk_vectors)
        n_chunks += 1

    logger.info(
        "[EMBED] %d texts in %d chunk(s): lock_wait=%.2fs encode=%.2fs",
        len(texts),
        n_chunks,
        total_wait,
        total_enc,
    )
    return vectors
