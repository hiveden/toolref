"""Embedding service backed by BGE-M3.

Uses :pypi:`FlagEmbedding` to generate both **dense** (1024-d float vector)
and **sparse** (lexical-weight dict) embeddings in a single forward pass.

The model is loaded lazily on first call to :meth:`embed_texts` and reused
for the lifetime of the process (singleton pattern).
"""

import logging
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate dense + sparse embeddings using BGE-M3.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Maximum texts per inference batch.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None  # Lazy-loaded BGEM3FlagModel

    # ── Lazy model loading ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the BGE-M3 model into memory (CPU, fp32)."""
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model '%s' …", self.model_name)
        self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        logger.info("Embedding model loaded successfully")

    def warmup(self) -> None:
        """Pre-load the model so the first real call is fast.

        Intended to be called at worker startup.
        """
        if self._model is None:
            self._load_model()

    # ── Public API ───────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Generate dense and sparse embeddings for a list of texts.

        Args:
            texts: Input strings to embed.

        Returns:
            A tuple ``(dense_embeddings, sparse_embeddings)`` where:

            * ``dense_embeddings`` is a numpy array of shape ``(N, 1024)``.
            * ``sparse_embeddings`` is a list of dicts mapping
              ``token_index -> weight`` (lexical weights).
        """
        if self._model is None:
            self._load_model()

        all_dense: list[np.ndarray] = []
        all_sparse: list[dict[str, float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            output = self._model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            all_dense.append(output["dense_vecs"])
            all_sparse.extend(output["lexical_weights"])

        dense_embeddings = (
            np.concatenate(all_dense, axis=0)
            if all_dense
            else np.empty((0, settings.embedding_dim))
        )
        logger.info(
            "Generated embeddings for %d texts (dense shape=%s)",
            len(texts),
            dense_embeddings.shape,
        )
        return dense_embeddings, all_sparse


# Module-level singleton — shared by the ingestion pipeline.
embedding_service = EmbeddingService()
