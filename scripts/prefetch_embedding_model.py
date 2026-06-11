"""Download and cache the BioLORD-2023 embedding model into the HF cache.

Run this once (manually or as a one-off job) against the persistent volume so
the app's first request loads the cached snapshot instantly instead of
downloading ~500MB at startup.

The model name and HF cache location come from the same config/env the app
uses (`embedding_model`, `HF_HOME`), so this populates exactly the path
`_get_model()` reads with `local_files_only=True`.

    HF_HOME=/cache python scripts/prefetch_embedding_model.py
"""

import logging
import os
import sys

from sentence_transformers import SentenceTransformer

from indication_scout.config import get_settings
from indication_scout.services.embeddings import _is_model_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("indication_scout.prefetch_embedding_model")


def main() -> int:
    model_name = get_settings().embedding_model
    hf_home = os.environ.get("HF_HOME", "<default ~/.cache/huggingface>")

    if _is_model_cached(model_name):
        logger.info("Model %s already cached under HF_HOME=%s — nothing to do", model_name, hf_home)
        return 0

    logger.info("Downloading model %s into HF_HOME=%s", model_name, hf_home)
    SentenceTransformer(model_name, local_files_only=False)

    if not _is_model_cached(model_name):
        logger.error("Download finished but model %s is still not detected in cache", model_name)
        return 1

    logger.info("Model %s downloaded and cached", model_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
