"""Embedding service backed by FlagEmbedding BGEM3FlagModel.

Generates **dense** (1024-d, L2-normalised) and **sparse** (lexical weights)
embeddings simultaneously using the BGE-M3 model.

Interface::

    embed_texts(texts) -> tuple[np.ndarray, list[dict[str, float]]]

* ``dense_vecs``      — numpy array of shape ``(N, 1024)``
* ``lexical_weights`` — list of dicts mapping str(token_id) → weight

The model is loaded lazily on first call to :meth:`embed_texts` and
reused for the lifetime of the process (singleton pattern).
"""

import logging
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate dense + sparse embeddings using BGE-M3 (FlagEmbedding).

    Attributes:
        model_name: HuggingFace model identifier (default: BAAI/bge-m3).
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
        """Load BGEM3FlagModel into memory (CPU, fp32)."""
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        logger.info("Loading BGE-M3 model '%s' …", self.model_name)
        # use_fp16=False because we are running on CPU
        self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        logger.info("BGE-M3 model loaded successfully")

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

        Processes ``texts`` in batches of ``self.batch_size``, collects
        results, and returns them as a single pair.

        Args:
            texts: Input strings to embed.

        Returns:
            A tuple ``(dense_embeddings, sparse_embeddings)`` where:

            * ``dense_embeddings`` is a numpy array of shape ``(N, 1024)``
              with L2-normalised rows.
            * ``sparse_embeddings`` is a list of N dicts, each mapping
              ``str(token_id)`` → ``float`` weight (lexical_weights from
              BGE-M3).  Downstream code converts keys to ``int`` as needed.
        """
        if self._model is None:
            self._load_model()

        all_dense: list[np.ndarray] = []
        all_sparse: list[dict[str, float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            output = self._model.encode(
                batch,
                batch_size=len(batch),
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            all_dense.append(output["dense_vecs"])
            all_sparse.extend(output["lexical_weights"])

        dense_embeddings: np.ndarray = (
            np.concatenate(all_dense, axis=0)
            if all_dense
            else np.empty((0, settings.embedding_dim))
        )
        logger.info(
            "Generated embeddings for %d texts (dense shape=%s, sparse count=%d)",
            len(texts),
            dense_embeddings.shape,
            len(all_sparse),
        )
        return dense_embeddings, all_sparse


# Module-level singleton — shared by the ingestion pipeline.
embedding_service = EmbeddingService()
