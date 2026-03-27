"""Redis-backed semantic cache for RAG responses.

Implements the caching strategy described in architecture §4.4:

* Cache key: query embedding (dense, 1024-d float32 bytes)
* Cache value: serialised ``{answer, sources, metadata, timestamp}``
* Lookup: scan namespace prefix, compute cosine similarity
* Tiered TTL: hot 72h / default 24h / low-freq 12h
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from app.config import settings
from app.ingestion.embedder import embedding_service

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _embedding_to_b64(embedding: np.ndarray) -> str:
    """Encode a float32 numpy array to a base64 string.

    This is safe for Redis connections with ``decode_responses=True``.
    """
    return base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode("ascii")


def _b64_to_embedding(b64_str: str) -> np.ndarray:
    """Decode a base64 string back to a float32 numpy array."""
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32)


def _make_cache_key(query: str, namespace: str) -> str:
    """Deterministic cache key from query + namespace."""
    h = hashlib.sha256(f"{namespace}:{query}".encode()).hexdigest()[:16]
    return f"rag_cache:{namespace}:{h}"


# ── SemanticCache ────────────────────────────────────────────────────────────


class SemanticCache:
    """Semantic cache backed by Redis.

    Attributes:
        similarity_threshold: Minimum cosine similarity for a cache hit.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        similarity_threshold: float = settings.cache_similarity_threshold,
    ) -> None:
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold

    # ── TTL helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ttl_for_frequency(frequency: str = "normal") -> int:
        """Return TTL in seconds for the given frequency tier."""
        ttl_map = {
            "high": settings.cache_hot_ttl,
            "normal": settings.cache_default_ttl,
            "low": settings.cache_low_freq_ttl,
        }
        return ttl_map.get(frequency, settings.cache_default_ttl)

    # ── Public API ────────────────────────────────────────────────────────

    async def get(self, query: str, namespace: str) -> dict[str, Any] | None:
        """Check cache for a semantically similar query.

        Process:
          1. Embed the incoming query (dense vector).
          2. Scan Redis for cached query embeddings in the namespace.
          3. If cosine similarity ≥ threshold → return cached result.

        Args:
            query: The user query string.
            namespace: Document namespace.

        Returns:
            Cached result dict or ``None`` on miss.
        """
        try:
            # Embed the query
            dense_embs, _ = await asyncio.to_thread(
                embedding_service.embed_texts, [query]
            )
            query_embedding = dense_embs[0]

            # Scan cached entries for this namespace
            pattern = f"rag_cache:{namespace}:*"
            cursor: int | bytes = 0
            scanned = 0

            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                for key in keys:
                    scanned += 1
                    cached_data = await self.redis.hgetall(key)
                    if not cached_data:
                        continue

                    # Retrieve the stored embedding (base64-encoded)
                    emb_b64 = cached_data.get("embedding") or cached_data.get(
                        b"embedding"
                    )
                    if emb_b64 is None:
                        continue

                    if isinstance(emb_b64, bytes):
                        emb_b64 = emb_b64.decode()

                    try:
                        cached_embedding = _b64_to_embedding(emb_b64)
                    except Exception:
                        continue

                    similarity = _cosine_similarity(query_embedding, cached_embedding)
                    if similarity >= self.similarity_threshold:
                        # Cache hit
                        result_raw = cached_data.get("result") or cached_data.get(
                            b"result"
                        )
                        if result_raw is None:
                            continue
                        if isinstance(result_raw, bytes):
                            result_raw = result_raw.decode()

                        logger.info(
                            "Semantic cache HIT (similarity=%.4f, key=%s)",
                            similarity,
                            key,
                        )
                        return json.loads(result_raw)

                if cursor == 0:
                    break

            logger.debug(
                "Semantic cache MISS for query='%s' namespace=%s (scanned %d entries)",
                query[:60],
                namespace,
                scanned,
            )
            return None

        except Exception:
            logger.exception("Semantic cache get failed — treating as MISS")
            return None

    async def put(
        self,
        query: str,
        namespace: str,
        result: dict[str, Any],
        frequency: str = "normal",
    ) -> None:
        """Cache a RAG result with tiered TTL.

        Args:
            query: The original query string.
            namespace: Document namespace.
            result: The result dict to cache (answer, sources, etc.).
            frequency: TTL tier — ``high``, ``normal``, or ``low``.
        """
        try:
            # Embed the query
            dense_embs, _ = await asyncio.to_thread(
                embedding_service.embed_texts, [query]
            )
            query_embedding = dense_embs[0]

            key = _make_cache_key(query, namespace)
            ttl = self._ttl_for_frequency(frequency)

            embedding_b64 = _embedding_to_b64(query_embedding)
            result_json = json.dumps(result, ensure_ascii=False, default=str)

            await self.redis.hset(
                key,
                mapping={
                    "embedding": embedding_b64,
                    "result": result_json,
                    "query": query,
                    "namespace": namespace,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            await self.redis.expire(key, ttl)

            logger.info(
                "Semantic cache PUT key=%s ttl=%ds frequency=%s",
                key,
                ttl,
                frequency,
            )

        except Exception:
            logger.exception("Semantic cache put failed — skipping cache write")

    async def invalidate_namespace(self, namespace: str) -> int:
        """Delete all cached entries for a given namespace.

        Args:
            namespace: The namespace whose cache entries should be purged.

        Returns:
            Number of keys deleted.
        """
        pattern = f"rag_cache:{namespace}:*"
        deleted = 0
        cursor: int | bytes = 0

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor, match=pattern, count=100
            )
            if keys:
                await self.redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        logger.info("Invalidated %d cache entries for namespace=%s", deleted, namespace)
        return deleted
