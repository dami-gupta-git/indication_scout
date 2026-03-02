"""Integration tests for services/embeddings — loads the real BioLORD-2023 model."""

import numpy as np

from indication_scout.services.embeddings import embed


def test_embed_determinism():
    """Encoding the same string twice produces identical vectors.

    BioLORD-2023 is a deterministic model (no dropout at inference time),
    so the same input must always yield the same output. If this fails it
    indicates a non-deterministic encode path (e.g. dropout left enabled).
    """
    text = "Metformin activates AMPK and inhibits mTOR in colon cancer cells."
    result_a = embed([text])
    result_b = embed([text])

    assert np.allclose(result_a[0], result_b[0])


def test_embed_batch_shape():
    """Encoding 5 strings returns 5 vectors each of length 768.

    768 is BioLORD-2023's output dimension. Verifying this here catches
    any model loading or configuration issue before it silently produces
    wrong-shaped vectors that would corrupt the pgvector index.
    """
    texts = [
        "Metformin and colorectal cancer",
        "AMPK activation in colon epithelium",
        "mTOR inhibition and cell proliferation",
        "Biguanide class drugs and oncology",
        "PRKAA1 pathway in gastrointestinal tumors",
    ]
    result = embed(texts)

    assert len(result) == 5
    assert all(len(v) == 768 for v in result)
