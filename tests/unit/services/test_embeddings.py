"""Unit tests for services/embeddings — no model loading."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import indication_scout.services.embeddings as embeddings_module
from indication_scout.services.embeddings import embed


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level _model to None before and after each test.

    Without this, a model instantiated (or mocked) in one test would leak
    into the next, breaking singleton-reuse assertions and causing
    SentenceTransformer to never be called in subsequent tests.
    """
    embeddings_module._model = None
    yield
    embeddings_module._model = None


def _make_mock_model(n_texts: int = 1, dim: int = 768) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns a zero numpy array.

    Shape is (n_texts, dim), matching BioLORD-2023's real output shape (N, 768).
    dtype=float32 matches what sentence-transformers returns by default.
    """
    mock = MagicMock()
    mock.encode.return_value = np.zeros((n_texts, dim), dtype=np.float32)
    return mock


def test_embed_returns_list_of_float_lists():
    """embed() converts the numpy array from encode() into list[list[float]].

    pgvector / SQLAlchemy expect plain Python floats, not numpy scalars,
    so the conversion via .tolist() is load-bearing.
    """
    mock_model = _make_mock_model(n_texts=1)
    with patch(
        "indication_scout.services.embeddings.SentenceTransformer",
        return_value=mock_model,
    ):
        result = embed(["some biomedical text"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 768
    assert all(isinstance(v, float) for v in result[0])


def test_embed_passes_texts_to_encode():
    """embed() forwards the full text list to encode() unchanged.

    Also asserts convert_to_numpy=True is passed — this is required so that
    the return value is an ndarray we can call .tolist() on per vector.
    """
    mock_model = _make_mock_model(n_texts=2)
    texts = ["metformin and colorectal cancer", "AMPK activation in colon"]
    with patch(
        "indication_scout.services.embeddings.SentenceTransformer",
        return_value=mock_model,
    ):
        embed(texts)

    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)


def test_embed_returns_one_vector_per_text():
    """Output list length matches input list length.

    Ensures the index alignment used in fetch_and_cache (zip(abstracts, vectors))
    will be correct when embedding a batch of abstracts.
    """
    mock_model = _make_mock_model(n_texts=3)
    with patch(
        "indication_scout.services.embeddings.SentenceTransformer",
        return_value=mock_model,
    ):
        result = embed(["a", "b", "c"])

    assert len(result) == 3


def test_singleton_not_reinstantiated_across_calls():
    """SentenceTransformer() is called exactly once across multiple embed() calls.

    BioLORD-2023 takes ~10s and ~500MB to load. The singleton ensures that
    cost is paid once per process, not once per abstract or per query.
    """
    mock_model = _make_mock_model(n_texts=1)
    with patch(
        "indication_scout.services.embeddings.SentenceTransformer",
        return_value=mock_model,
    ) as mock_cls:
        embed(["first call"])
        embed(["second call"])

    mock_cls.assert_called_once()
