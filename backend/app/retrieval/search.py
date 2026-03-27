"""Milvus hybrid search with RRF fusion.

Implements the dense + sparse hybrid retrieval strategy described in
architecture §4.2.4. Uses BGE-M3 embeddings (via :mod:`app.ingestion.embedder`)
and Reciprocal Rank Fusion to combine results.
"""

from __future__ import annotations

import asyncio
import logging

from app.ingestion.embedder import embedding_service
from app.vectorstore.milvus import CHILD_CHUNKS_COLLECTION

logger = logging.getLogger(__name__)

# ── RRF fusion ────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) for each ranking list.

    Args:
        dense_results: Documents ranked by dense (vector) search.
        sparse_results: Documents ranked by sparse (lexical) search.
        k: RRF constant (standard value = 60).

    Returns:
        Fused list sorted by RRF score (descending), each dict
        augmented with ``rrf_score``.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        chunk_id = doc["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
        doc_map[chunk_id] = doc

    for rank, doc in enumerate(sparse_results):
        chunk_id = doc["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [{**doc_map[cid], "rrf_score": scores[cid]} for cid in sorted_ids]


# ── Milvus search helpers ────────────────────────────────────────────────────


def _milvus_hybrid_search_sync(
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    namespace: str,
    top_k: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Run dense + sparse ANN search against Milvus (synchronous).

    Returns a tuple ``(dense_results, sparse_results)`` where each is a
    list of dicts with ``chunk_id``, ``doc_id``, ``parent_chunk_id``,
    ``namespace``, and ``score``.
    """
    from pymilvus import AnnSearchRequest, Collection, RRFRanker

    collection = Collection(CHILD_CHUNKS_COLLECTION)
    collection.load()

    # Filter by namespace
    namespace_filter = f'namespace == "{namespace}"'

    # Dense ANN search request
    dense_req = AnnSearchRequest(
        data=[dense_embedding],
        anns_field="dense_embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 128},
        },
        limit=top_k,
        expr=namespace_filter,
    )

    # Sparse ANN search request
    sparse_req = AnnSearchRequest(
        data=[sparse_embedding],
        anns_field="sparse_embedding",
        param={
            "metric_type": "IP",
            "params": {},
        },
        limit=top_k,
        expr=namespace_filter,
    )

    # Execute hybrid search using Milvus built-in RRF ranker
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["chunk_id", "doc_id", "parent_chunk_id", "namespace"],
    )

    # Parse results — hybrid_search returns a list of hits lists
    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    # When using hybrid_search with two reqs, results are already fused.
    # We extract all hits as a single ranked list.
    if results:
        for hit in results[0]:
            entry = {
                "chunk_id": hit.id,
                "doc_id": hit.entity.get("doc_id", ""),
                "parent_chunk_id": hit.entity.get("parent_chunk_id", ""),
                "namespace": hit.entity.get("namespace", ""),
                "score": hit.distance,
            }
            dense_results.append(entry)

    return dense_results, sparse_results


def _milvus_search_single_sync(
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    namespace: str,
    top_k: int = 20,
) -> list[dict]:
    """Search Milvus and fuse using application-level RRF.

    Performs two independent ANN searches (dense + sparse) and fuses
    results with :func:`reciprocal_rank_fusion`.
    """
    from pymilvus import Collection

    collection = Collection(CHILD_CHUNKS_COLLECTION)
    collection.load()

    namespace_filter = f'namespace == "{namespace}"'
    output_fields = ["chunk_id", "doc_id", "parent_chunk_id", "namespace"]

    # Dense search
    dense_raw = collection.search(
        data=[dense_embedding],
        anns_field="dense_embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        expr=namespace_filter,
        output_fields=output_fields,
    )

    dense_results: list[dict] = []
    for hits in dense_raw:
        for hit in hits:
            dense_results.append({
                "chunk_id": hit.id,
                "doc_id": hit.entity.get("doc_id", ""),
                "parent_chunk_id": hit.entity.get("parent_chunk_id", ""),
                "namespace": hit.entity.get("namespace", ""),
                "score": hit.distance,
            })

    # Sparse search
    sparse_raw = collection.search(
        data=[sparse_embedding],
        anns_field="sparse_embedding",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
        expr=namespace_filter,
        output_fields=output_fields,
    )

    sparse_results: list[dict] = []
    for hits in sparse_raw:
        for hit in hits:
            sparse_results.append({
                "chunk_id": hit.id,
                "doc_id": hit.entity.get("doc_id", ""),
                "parent_chunk_id": hit.entity.get("parent_chunk_id", ""),
                "namespace": hit.entity.get("namespace", ""),
                "score": hit.distance,
            })

    # Fuse with application-level RRF
    return reciprocal_rank_fusion(dense_results, sparse_results, k=60)


# ── Public async API ─────────────────────────────────────────────────────────


async def hybrid_search(
    query_text: str,
    namespace: str,
    top_k: int = 20,
) -> list[dict]:
    """Run hybrid dense + sparse search against Milvus.

    Steps:
      1. Embed the query using BGE-M3 (dense + sparse).
      2. Perform two independent ANN searches in Milvus.
      3. Fuse results with Reciprocal Rank Fusion (k=60).

    Args:
        query_text: The search query string.
        namespace: Document namespace to search within.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts, each containing ``chunk_id``, ``doc_id``,
        ``parent_chunk_id``, ``namespace``, ``score``, and ``rrf_score``.
    """
    # Generate query embeddings (sync model, run in thread)
    dense_embs, sparse_embs = await asyncio.to_thread(
        embedding_service.embed_texts, [query_text]
    )

    dense_embedding = dense_embs[0].tolist()
    # Convert sparse keys to int for Milvus compatibility
    raw_sparse = sparse_embs[0]
    sparse_embedding = {int(k): float(v) for k, v in raw_sparse.items()}

    # Run Milvus search in thread (pymilvus is synchronous)
    try:
        results = await asyncio.to_thread(
            _milvus_search_single_sync,
            dense_embedding,
            sparse_embedding,
            namespace,
            top_k,
        )
    except Exception:
        logger.exception("Milvus hybrid search failed for namespace=%s", namespace)
        results = []

    logger.info(
        "Hybrid search returned %d results for query='%s' namespace=%s",
        len(results),
        query_text[:60],
        namespace,
    )
    return results[:top_k]
