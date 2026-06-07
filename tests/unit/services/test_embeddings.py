"""Unit tests for services/embeddings — no model loading."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import indication_scout.services.embeddings as embeddings_module
from indication_scout.services.embeddings import _is_model_cached, embed


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


# ---------------------------------------------------------------------------
# _is_model_cached
# ---------------------------------------------------------------------------


def test_is_model_cached_returns_false_when_snapshots_dir_absent(tmp_path):
    """_is_model_cached() returns False when the HF snapshots directory does not exist."""
    with patch.dict("os.environ", {"HF_HOME": str(tmp_path)}):
        assert _is_model_cached("FremyCompany/BioLORD-2023") is False


def test_is_model_cached_returns_false_when_snapshots_dir_empty(tmp_path):
    """_is_model_cached() returns False when the snapshots directory exists but is empty.

    An empty snapshots dir indicates an interrupted download — we should
    re-download rather than attempt to load a partial model.
    """
    snapshots_dir = (
        tmp_path / "hub" / "models--FremyCompany--BioLORD-2023" / "snapshots"
    )
    snapshots_dir.mkdir(parents=True)
    with patch.dict("os.environ", {"HF_HOME": str(tmp_path)}):
        assert _is_model_cached("FremyCompany/BioLORD-2023") is False


def test_is_model_cached_returns_true_when_snapshot_present(tmp_path):
    """_is_model_cached() returns True when a non-empty snapshots directory exists."""
    snapshot_dir = (
        tmp_path
        / "hub"
        / "models--FremyCompany--BioLORD-2023"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")
    with patch.dict("os.environ", {"HF_HOME": str(tmp_path)}):
        assert _is_model_cached("FremyCompany/BioLORD-2023") is True


def test_is_model_cached_uses_hf_home_env_var(tmp_path):
    """_is_model_cached() resolves the cache root from the HF_HOME env var."""
    custom_home = tmp_path / "custom_hf"
    snapshot_dir = (
        custom_home / "hub" / "models--FremyCompany--BioLORD-2023" / "snapshots" / "rev1"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "pytorch_model.bin").write_bytes(b"\x00")
    with patch.dict("os.environ", {"HF_HOME": str(custom_home)}):
        assert _is_model_cached("FremyCompany/BioLORD-2023") is True


# ---------------------------------------------------------------------------
# _get_model() cache-aware loading branches
# ---------------------------------------------------------------------------


def test_get_model_uses_local_files_only_when_cached():
    """When the model is cached, SentenceTransformer is called with local_files_only=True."""
    mock_model = _make_mock_model()
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=True,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls,
    ):
        embed(["text"])

    _, kwargs = mock_cls.call_args
    assert kwargs.get("local_files_only") is True


def test_get_model_downloads_when_not_cached():
    """When the model is absent from cache, SentenceTransformer is called with local_files_only=False."""
    mock_model = _make_mock_model()
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=False,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls,
    ):
        embed(["text"])

    _, kwargs = mock_cls.call_args
    assert kwargs.get("local_files_only") is False


# ---------------------------------------------------------------------------
# embed() / singleton behaviour
# ---------------------------------------------------------------------------


def test_embed_returns_list_of_float_lists():
    """embed() converts the numpy array from encode() into list[list[float]].

    pgvector / SQLAlchemy expect plain Python floats, not numpy scalars,
    so the conversion via .tolist() is load-bearing.
    """
    mock_model = _make_mock_model(n_texts=1)
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=True,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ),
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
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=True,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ),
    ):
        embed(texts)

    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)


def test_embed_returns_one_vector_per_text():
    """Output list length matches input list length.

    Ensures the index alignment used in fetch_and_cache (zip(abstracts, vectors))
    will be correct when embedding a batch of abstracts.
    """
    mock_model = _make_mock_model(n_texts=3)
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=True,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ),
    ):
        result = embed(["a", "b", "c"])

    assert len(result) == 3


def test_singleton_not_reinstantiated_across_calls():
    """SentenceTransformer() is called exactly once across multiple embed() calls.

    BioLORD-2023 takes ~10s and ~500MB to load. The singleton ensures that
    cost is paid once per process, not once per abstract or per query.
    """
    mock_model = _make_mock_model(n_texts=1)
    with (
        patch(
            "indication_scout.services.embeddings._is_model_cached",
            return_value=True,
        ),
        patch(
            "indication_scout.services.embeddings.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls,
    ):
        embed(["first call"])
        embed(["second call"])

    mock_cls.assert_called_once()
