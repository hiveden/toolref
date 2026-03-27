"""LangGraph Agentic RAG graph construction.

Builds and compiles the retrieval state machine described in
architecture §4.2.1–4.2.3.  The compiled graph is the single entry
point for executing a full RAG pipeline.

Usage::

    graph = build_rag_graph()
    result = await graph.ainvoke({
        "query": "What is ...",
        "namespace": "default",
        "rewrite_count": 0,
        ...
    })
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledGraph

from app.config import settings
from app.retrieval.nodes import (
    analyze_query_node,
    consistency_check_node,
    decompose_query_node,
    generate_node,
    grade_documents_node,
    hybrid_retrieve_node,
    rerank_node,
    rewrite_query_node,
    route_node,
)
from app.retrieval.state import RAGState

logger = logging.getLogger(__name__)


# ── Routing functions ────────────────────────────────────────────────────────


def route_after_analysis(state: RAGState) -> str:
    """Route based on query complexity (simple → search, complex → decompose)."""
    return state.get("query_type", "simple")


def route_after_grading(state: RAGState) -> str:
    """Decide next step after document grading.

    * ``relevant``  → proceed to consistency check.
    * ``rewrite``   → rewrite query and re-retrieve.
    * ``fallback``  → max retries exceeded; generate with what we have.
    """
    if state.get("is_relevant"):
        return "relevant"
    if state.get("rewrite_count", 0) >= settings.max_rewrite_count:
        return "fallback"
    return "rewrite"


def route_after_consistency(state: RAGState) -> str:
    """Decide next step after consistency check (V1).

    * ``skip``       → MVP default; consistency check is disabled.
    * ``consistent`` → answers agree; proceed to generate.
    * ``divergent``  → answers conflict; re-retrieve with divergence query.
    """
    if state.get("consistency_passed") is None:
        return "skip"  # MVP: node not active
    if state["consistency_passed"]:
        return "consistent"
    return "divergent"


# ── Graph builder ────────────────────────────────────────────────────────────


def build_rag_graph() -> CompiledGraph:
    """Build and compile the Agentic RAG LangGraph state machine.

    Returns:
        A compiled :class:`CompiledGraph` ready for ``ainvoke``.
    """
    graph = StateGraph(RAGState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("route", route_node)
    graph.add_node("decompose_query", decompose_query_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("consistency_check", consistency_check_node)
    graph.add_node("generate", generate_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("analyze_query")

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge("analyze_query", "route")

    # Conditional routing: simple → search, complex → decompose
    graph.add_conditional_edges(
        "route",
        route_after_analysis,
        {
            "simple": "hybrid_retrieve",
            "complex": "decompose_query",
        },
    )

    graph.add_edge("decompose_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")

    # Grading → generate or rewrite
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "relevant": "consistency_check",
            "rewrite": "rewrite_query",
            "fallback": "generate",  # max retries exceeded
        },
    )

    graph.add_edge("rewrite_query", "hybrid_retrieve")

    # Consistency check → generate or re-retrieve
    graph.add_conditional_edges(
        "consistency_check",
        route_after_consistency,
        {
            "consistent": "generate",
            "divergent": "hybrid_retrieve",
            "skip": "generate",  # MVP: skip consistency check
        },
    )

    graph.add_edge("generate", END)

    compiled = graph.compile()
    logger.info("RAG graph compiled successfully")
    return compiled


# Module-level compiled graph — reused across requests.
rag_graph = build_rag_graph()
