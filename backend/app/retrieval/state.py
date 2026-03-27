"""RAG state schema for the LangGraph Agentic RAG graph.

Defines :class:`RAGState`, the shared TypedDict that flows through every
node in the retrieval state machine (architecture §4.2.2).
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    """State schema for the Agentic RAG graph.

    All fields use ``total=False`` so that nodes can return *partial*
    updates — LangGraph merges them automatically.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query: str
    namespace: str
    conversation_id: str | None

    # ── Query Analysis ────────────────────────────────────────────────────
    query_type: Literal["simple", "complex"] | None
    sub_queries: list[str]
    entities: list[str]

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_docs: list[dict]      # [{chunk_id, text, score, source, …}]
    reranked_docs: list[dict]       # post cross-encoder reranking

    # ── Grading ───────────────────────────────────────────────────────────
    relevance_scores: list[float]
    is_relevant: bool | None

    # ── Self-Correction ───────────────────────────────────────────────────
    rewrite_count: int              # track retry attempts
    rewritten_query: str | None

    # ── Consistency Check (V1) ────────────────────────────────────────────
    consistency_passed: bool | None
    divergence_query: str | None

    # ── Generation ────────────────────────────────────────────────────────
    answer: str | None
    sources: list[dict]             # [{doc_title, chunk_text, url, score}]
    cached: bool

    # ── Metadata ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]  # conversation history
    latency_ms: dict                # per-node latency tracking
