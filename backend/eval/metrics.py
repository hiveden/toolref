"""Custom IR metrics for ToolRef RAG evaluation.

These metrics don't require LLM-as-judge — computed purely from
retrieval results and expected documents. Fast, deterministic,
and reproducible.

Metrics:
    - Hit Rate@K: Did at least one expected document appear in top-K?
    - MRR (Mean Reciprocal Rank): Position of first relevant document.
    - Precision@K: Fraction of top-K that are relevant.
    - Recall@K: Fraction of expected docs found in top-K.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IRMetrics:
    """IR evaluation result for a single query."""

    hit: bool             # Did any expected doc appear in results?
    reciprocal_rank: float  # 1/rank of first relevant result (0 if miss)
    precision_at_k: float   # relevant_in_k / k
    recall_at_k: float      # relevant_in_k / total_expected


def compute_ir_metrics(
    retrieved_doc_titles: list[str],
    expected_doc_titles: list[str],
    k: int = 5,
) -> IRMetrics:
    """Compute IR metrics for a single query.

    Args:
        retrieved_doc_titles: Titles of documents returned by retrieval,
            ordered by rank (best first).
        expected_doc_titles: Titles of documents that should be retrieved.
        k: Cutoff for top-K evaluation.

    Returns:
        IRMetrics for this query.
    """
    if not expected_doc_titles:
        # Out-of-scope query — no expected docs.
        # Hit=True means system correctly returned nothing relevant.
        return IRMetrics(hit=True, reciprocal_rank=1.0, precision_at_k=1.0, recall_at_k=1.0)

    top_k = retrieved_doc_titles[:k]
    expected_set = set(expected_doc_titles)

    # Find first relevant hit
    first_rank = 0
    relevant_count = 0
    for i, title in enumerate(top_k):
        if title in expected_set:
            relevant_count += 1
            if first_rank == 0:
                first_rank = i + 1

    hit = first_rank > 0
    rr = 1.0 / first_rank if first_rank > 0 else 0.0
    precision = relevant_count / k if k > 0 else 0.0
    recall = relevant_count / len(expected_doc_titles) if expected_doc_titles else 0.0

    return IRMetrics(
        hit=hit,
        reciprocal_rank=rr,
        precision_at_k=precision,
        recall_at_k=recall,
    )


def aggregate_ir_metrics(results: list[IRMetrics]) -> dict[str, float]:
    """Aggregate IR metrics across multiple queries.

    Returns:
        Dict with mean Hit Rate, MRR, Precision@K, Recall@K.
    """
    n = len(results)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

    return {
        "hit_rate": sum(1 for r in results if r.hit) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "precision_at_k": sum(r.precision_at_k for r in results) / n,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
    }
