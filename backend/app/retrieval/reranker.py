"""Reranking service supporting local (CrossEncoder) and Jina API providers.

Provider is selected via ``settings.reranker_provider``:

* ``"local"`` — Uses a HuggingFace ``CrossEncoder`` model loaded in-process
  (default: BGE-reranker-v2-m3).  Model is lazy-loaded and reused as a
  module-level singleton (architecture §4.2.5).
* ``"jina"`` — Calls the Jina Reranker REST API
  (https://api.jina.ai/v1/rerank) using ``httpx.AsyncClient``.  Because
  :meth:`RerankerService.rerank` must remain a synchronous interface (callers
  use ``asyncio.to_thread``), the async HTTP call is driven via
  ``asyncio.run()`` which is safe inside a worker thread that has no running
  event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Reranking service with pluggable local / Jina API provider.

    Attributes:
        model_name: HuggingFace model identifier used in ``"local"`` mode.
        top_k: Default number of top documents to return after reranking.
        provider: Active provider — ``"local"`` or ``"jina"``.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        top_k: int = settings.reranker_top_k,
        provider: str = settings.reranker_provider,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.provider = provider
        self._reranker: Any = None  # Lazy-loaded CrossEncoder (local mode only)

    # ── Local provider: lazy model loading ───────────────────────────────

    def _load_model(self) -> None:
        """Load the CrossEncoder model into memory (local provider only).

        NOTE: Temporarily using cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB)
        instead of BGE-reranker-v2-m3 (~1.1 GB) to unblock development.
        """
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        logger.info("Loading reranker model '%s' …", self.model_name)
        self._reranker = CrossEncoder(self.model_name, max_length=512)
        logger.info("Reranker model loaded successfully")

    def warmup(self) -> None:
        """Pre-load the local model so the first real call is fast.

        No-op when ``provider`` is ``"jina"``.
        """
        if self.provider == "local" and self._reranker is None:
            self._load_model()

    # ── Local provider: scoring ───────────────────────────────────────────

    def _rerank_local(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rerank using the in-process CrossEncoder model.

        Args:
            query: User query string.
            documents: Candidate docs; each must contain a ``"text"`` key.
            top_k: Maximum number of results to return.

        Returns:
            Top-*k* docs sorted by ``rerank_score`` descending.
        """
        if self._reranker is None:
            self._load_model()

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self._reranker.predict(pairs)

        # predict() returns a bare float when given a single pair
        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores, strict=False):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_k]

    # ── Jina provider: async HTTP call ────────────────────────────────────

    async def _rerank_jina_async(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rerank via the Jina Reranker REST API (async).

        Args:
            query: User query string.
            documents: Candidate docs; each must contain a ``"text"`` key.
            top_k: ``top_n`` value forwarded to the Jina API.

        Returns:
            Top-*k* docs sorted by ``rerank_score`` descending, each
            augmented with a ``rerank_score`` field derived from
            ``relevance_score`` returned by the API.

        Raises:
            httpx.HTTPStatusError: When the Jina API returns a non-2xx status.
            ValueError: When ``jina_api_key`` is not configured.
        """
        if not settings.jina_api_key:
            raise ValueError(
                "reranker_provider='jina' requires JINA_API_KEY to be set."
            )

        texts = [doc["text"] for doc in documents]
        payload = {
            "model": settings.jina_reranker_model,
            "query": query,
            "documents": texts,
            "top_n": top_k,
        }
        headers = {
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "Calling Jina Reranker API: model=%s top_n=%d docs=%d",
            settings.jina_reranker_model,
            top_k,
            len(texts),
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.jina_api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        data = response.json()
        results: list[dict] = data.get("results", [])

        # Map API results back to original document dicts
        scored_docs: list[dict] = []
        for result in results:
            idx: int = result["index"]
            score: float = float(result["relevance_score"])
            doc = dict(documents[idx])  # shallow copy to avoid mutating caller data
            doc["rerank_score"] = score
            scored_docs.append(doc)

        # API already returns top_n results, but sort defensively
        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_k]

    def _rerank_jina(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Synchronous wrapper around :meth:`_rerank_jina_async`.

        Safe to call from a worker thread (``asyncio.to_thread``) because no
        running event loop exists in that context.
        """
        return asyncio.run(self._rerank_jina_async(query, documents, top_k))

    # ── Public API ────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank *documents* against *query* using the configured provider.

        Dispatches to the local CrossEncoder or the Jina Reranker API
        depending on ``settings.reranker_provider``.

        Args:
            query: The user query string.
            documents: List of dicts, each must contain a ``"text"`` key.
            top_k: Number of top documents to return.  Falls back to
                ``self.top_k`` when ``None``.

        Returns:
            The top-k documents sorted by ``rerank_score`` (descending),
            each augmented with a ``rerank_score`` field.

        Raises:
            ValueError: For an unknown ``reranker_provider`` value.
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        if self.provider == "local":
            return self._rerank_local(query, documents, top_k)
        elif self.provider == "jina":
            return self._rerank_jina(query, documents, top_k)
        else:
            raise ValueError(
                f"Unknown reranker_provider '{self.provider}'. "
                "Expected 'local' or 'jina'."
            )


# Module-level singleton — shared across the retrieval pipeline.
reranker_service = RerankerService()
