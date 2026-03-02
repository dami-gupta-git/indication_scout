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
"""

import logging

from sentence_transformers import SentenceTransformer

from indication_scout.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton. None until the first call to embed().
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the singleton model, instantiating it on first call."""
    global _model
    if _model is None:
        model_name = get_settings().embedding_model
        logger.info("Loading embedding model %s", model_name)
        _model = SentenceTransformer(model_name)
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
