"""Retrieval Engine — LangGraph Agentic RAG state machine.

This package implements the core retrieval pipeline described in
architecture §4.2:

* **state** — ``RAGState`` TypedDict shared across all graph nodes.
* **nodes** — Individual node functions (analyze, retrieve, rerank, …).
* **graph** — LangGraph ``StateGraph`` wiring and compilation.
* **search** — Milvus hybrid search (dense + sparse + RRF fusion).
* **reranker** — Cross-encoder reranking service.
* **cache** — Redis-backed semantic cache.
* **llm** — LLM client abstraction (Ollama / OpenAI-compatible).
"""
