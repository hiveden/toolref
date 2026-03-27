"""Cross-encoder reranking service.

Uses ``FlagReranker`` (BGE-reranker-v2-m3) to rerank candidate documents
after hybrid retrieval. The model is loaded lazily and reused as a
module-level singleton (architecture §4.2.5).
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-encoder reranking using BGE-reranker-v2-m3.

    Attributes:
        model_name: HuggingFace model identifier for the reranker.
        top_k: Default number of top documents to return after reranking.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        top_k: int = settings.reranker_top_k,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._reranker: Any = None  # Lazy-loaded FlagReranker

    # ── Lazy model loading ────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the reranker model into memory."""
        from FlagEmbedding import FlagReranker

        logger.info("Loading reranker model '%s' …", self.model_name)
        self._reranker = FlagReranker(self.model_name, use_fp16=False)
        logger.info("Reranker model loaded successfully")

    def warmup(self) -> None:
        """Pre-load the model so the first real call is fast."""
        if self._reranker is None:
            self._load_model()

    # ── Public API ────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank *documents* against *query* using the cross-encoder.

        Args:
            query: The user query string.
            documents: List of dicts, each must contain a ``text`` key.
            top_k: Number of top documents to return. Falls back to
                ``self.top_k`` when ``None``.

        Returns:
            The top-k documents sorted by reranker score (descending),
            each augmented with a ``rerank_score`` field.
        """
        if self._reranker is None:
            self._load_model()

        if not documents:
            return []

        top_k = top_k or self.top_k

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self._reranker.compute_score(pairs)

        # compute_score returns a single float when there is only one pair
        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_k]


# Module-level singleton — shared across the retrieval pipeline.
reranker_service = RerankerService()
