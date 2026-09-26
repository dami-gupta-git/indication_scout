"""Download the BioLORD-2023 embedding model into the HF cache so the app's first request does not download ~500MB.

Uses the app's own `embedding_model` and `HF_HOME` settings, so it fills exactly the path `_get_model()` reads.

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
