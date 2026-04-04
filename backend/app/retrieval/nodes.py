"""LangGraph node functions for the Agentic RAG state machine.

Each function takes the current :class:`RAGState` and returns a *partial*
state update dict.  LangGraph merges the returned dict into the graph
state automatically.

Node overview (architecture §4.2.3–4.2.7):

* :func:`analyze_query_node` — Classify intent, extract entities.
* :func:`route_node` — No-op; used as a branching point.
* :func:`decompose_query_node` — Split complex query into sub-queries.
* :func:`hybrid_retrieve_node` — Dense + sparse Milvus search + parent fetch.
* :func:`rerank_node` — Cross-encoder reranking.
* :func:`grade_documents_node` — LLM-as-judge relevance scoring.
* :func:`rewrite_query_node` — Rewrite query for better retrieval.
* :func:`consistency_check_node` — V1 answer-consistency verification.
* :func:`generate_node` — Final LLM answer generation with citations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db.engine import async_session
from app.db.models import Chunk, Document
from app.retrieval.llm import get_llm
from app.retrieval.reranker import reranker_service
from app.retrieval.search import hybrid_search
from app.retrieval.state import RAGState

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Prompts — defined as constants for easy tuning
# ═════════════════════════════════════════════════════════════════════════════

ANALYZE_QUERY_PROMPT = """\
You are a query analysis assistant. Analyze the following user query and return a JSON object.

Query: {query}

Return JSON with:
- "query_type": "simple" or "complex" (complex if multi-hop,
  involves comparisons, or requires multiple facts)
- "entities": list of key entities/concepts mentioned in the query
- "intent": brief description of what the user wants to know

Respond ONLY with valid JSON, no other text.
"""

DECOMPOSE_QUERY_PROMPT = """\
You are a query decomposition assistant. Break the following complex query \
into 2-4 simpler sub-queries that can each be answered independently.

Original query: {query}
Entities identified: {entities}

Return a JSON object with:
- "sub_queries": list of 2-4 simpler search queries

Respond ONLY with valid JSON, no other text.
"""

GRADE_DOCUMENT_PROMPT = """\
You are a document relevance grader. Your job is to decide if a document is \
useful for answering a query.

DEFINITION: A document is RELEVANT if it contains ANY information that could \
help answer the query — even partially. It does NOT need to fully answer the query.

EXAMPLES:
- Query: "What is self-corrective RAG?" | Document discusses LangGraph agentic \
RAG patterns → relevant: true
- Query: "How does CRAG work?" | Document explains corrective retrieval and \
query rewriting steps → relevant: true
- Query: "Python async best practices" | Document is about JavaScript promises \
→ relevant: false
- Query: "LangChain tool calling" | Document mentions LangChain agents and \
function calling → relevant: true

Now grade this pair:
Query: {query}
Document: {document}

Respond with JSON only — no explanation outside the JSON:
{{"relevant": true, "reason": "..."}}
or
{{"relevant": false, "reason": "..."}}"""

REWRITE_QUERY_PROMPT = """\
The original query did not retrieve sufficiently relevant documents. Generate a better search query.

Original query: {query}
Documents retrieved (not relevant enough):
{doc_summaries}

Generate a single, improved search query that would find the right information. \
Focus on key terms and specificity.

Return ONLY the rewritten query text, nothing else.
"""

GENERATE_PROMPT = """\
You are a knowledgeable assistant. Answer the user's question based ONLY on \
the provided context documents. Always cite your sources.

{memory_section}\
Question: {query}

Context documents:
{context}

Instructions:
1. Answer the question based ONLY on the provided context.
2. If the context doesn't contain enough information, say so explicitly.
3. Cite sources using [Source N] notation where N is the document number.
4. Be concise but thorough.
"""

CONSISTENCY_CHECK_PROMPT = """\
Compare these two answers to the same question and determine if they are consistent.

Question: {query}
Answer A: {answer_a}
Answer B: {answer_b}

Return a JSON object with:
- "consistent": true or false
- "divergence": if inconsistent, describe the specific point of divergence; otherwise null

Respond ONLY with valid JSON, no other text.
"""


# ═════════════════════════════════════════════════════════════════════════════
# JSON parsing helper
# ═════════════════════════════════════════════════════════════════════════════


def _safe_parse_json(text: str, fallback: dict | None = None) -> dict:
    """Parse JSON from LLM output with multi-stage fallback on failure.

    Strategy:
    1. Strip markdown code fences, then try standard ``json.loads``.
    2. Use regex to extract the first ``{...}`` block from freeform text.
    3. Keyword heuristic: detect yes/no/true/false/relevant/irrelevant to
       synthesise a ``{"relevant": bool}`` result for grading prompts.

    The interface is unchanged: always returns a dict (never raises).
    """
    if fallback is None:
        fallback = {}

    # ── Stage 1: strip markdown code fences ─────────────────────────────────
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # fall through to next strategy

    # ── Stage 2: regex — extract first {...} block from surrounding prose ────
    # Use DOTALL so the pattern spans newlines (common for multi-line JSON).
    # Also handle single-quoted keys/values that some small models emit.
    json_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)
        # Normalise single quotes → double quotes as a best-effort fix.
        candidate_dq = re.sub(r"(?<![\\])'", '"', candidate)
        for attempt in (candidate_dq, candidate):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    # ── Stage 3: keyword heuristic (grading-specific) ───────────────────────
    # Only activate when the fallback dict has a "relevant" key, which signals
    # this is a grading call.  Avoids polluting other parse sites.
    if "relevant" in fallback:
        lower = text.lower()
        positive_signals = ("yes", "true", "relevant", "pertinent", "useful", "related")
        negative_signals = ("no", "false", "irrelevant", "unrelated", "not relevant", "not useful")
        pos_count = sum(1 for kw in positive_signals if re.search(r"\b" + kw + r"\b", lower))
        neg_count = sum(1 for kw in negative_signals if re.search(r"\b" + kw + r"\b", lower))
        if pos_count > neg_count:
            logger.debug("_safe_parse_json: keyword heuristic → relevant=true (%s)", text[:120])
            return {"relevant": True, "reason": "keyword_heuristic"}
        if neg_count > pos_count:
            logger.debug("_safe_parse_json: keyword heuristic → relevant=false (%s)", text[:120])
            return {"relevant": False, "reason": "keyword_heuristic"}

    logger.warning("Failed to parse JSON from LLM output: %s", text[:200])
    return fallback


# ═════════════════════════════════════════════════════════════════════════════
# Node functions
# ═════════════════════════════════════════════════════════════════════════════


async def analyze_query_node(state: RAGState) -> dict[str, Any]:
    """Analyze query intent, extract entities, determine complexity.

    Uses LLM to classify the query as simple or complex and extract
    key entities for downstream processing.
    """
    t0 = time.monotonic()
    query = state["query"]
    llm = get_llm()

    try:
        prompt = ANALYZE_QUERY_PROMPT.format(query=query)
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = _safe_parse_json(
            content,
            fallback={"query_type": "simple", "entities": [], "intent": ""},
        )

        query_type = parsed.get("query_type", "simple")
        if query_type not in ("simple", "complex"):
            query_type = "simple"

        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []

    except Exception:
        logger.exception("analyze_query LLM call failed — defaulting to simple")
        query_type = "simple"
        entities = []

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["analyze_query"] = elapsed_ms

    logger.info(
        "analyze_query: type=%s entities=%s elapsed=%dms",
        query_type,
        entities,
        elapsed_ms,
    )

    return {
        "query_type": query_type,
        "entities": entities,
        "latency_ms": latency,
    }


async def route_node(state: RAGState) -> dict[str, Any]:
    """No-op routing node — used only as a conditional-edge branching point."""
    return {}


async def decompose_query_node(state: RAGState) -> dict[str, Any]:
    """Decompose a complex query into 2–4 simpler sub-queries."""
    t0 = time.monotonic()
    query = state["query"]
    entities = state.get("entities", [])
    llm = get_llm()

    try:
        prompt = DECOMPOSE_QUERY_PROMPT.format(
            query=query,
            entities=", ".join(entities) if entities else "none",
        )
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = _safe_parse_json(content, fallback={"sub_queries": [query]})
        sub_queries = parsed.get("sub_queries", [query])

        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [query]

        # Limit to 4 sub-queries
        sub_queries = sub_queries[:4]

    except Exception:
        logger.exception("decompose_query LLM call failed — using original query")
        sub_queries = [query]

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["decompose_query"] = elapsed_ms

    logger.info(
        "decompose_query: %d sub-queries elapsed=%dms",
        len(sub_queries),
        elapsed_ms,
    )

    return {
        "sub_queries": sub_queries,
        "latency_ms": latency,
    }


async def hybrid_retrieve_node(state: RAGState) -> dict[str, Any]:
    """Perform hybrid retrieval (dense + sparse + RRF) from Milvus.

    If sub-queries exist, each is searched independently and results
    are merged and deduplicated. After retrieving child chunks, their
    parent chunk text is fetched from PostgreSQL.
    """
    t0 = time.monotonic()
    query = state["query"]
    namespace = state["namespace"]
    sub_queries = state.get("sub_queries", [])

    top_k_per_source = 20
    final_top_k = 10

    # Determine which queries to search
    queries_to_search = sub_queries if sub_queries else [query]

    all_results: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for q in queries_to_search:
        try:
            results = await hybrid_search(q, namespace, top_k=top_k_per_source)
            for result in results:
                chunk_id = result.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_results.append(result)
        except Exception:
            logger.exception("Hybrid search failed for sub-query: %s", q[:60])

    # Sort by rrf_score and take top results
    all_results.sort(key=lambda d: d.get("rrf_score", 0), reverse=True)
    all_results = all_results[:final_top_k]

    # Fetch parent chunk text from PostgreSQL for richer context
    if all_results:
        all_results = await _enrich_with_parent_text(all_results)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["hybrid_retrieve"] = elapsed_ms

    logger.info(
        "hybrid_retrieve: %d results elapsed=%dms",
        len(all_results),
        elapsed_ms,
    )

    return {
        "retrieved_docs": all_results,
        "latency_ms": latency,
    }


async def _enrich_with_parent_text(docs: list[dict]) -> list[dict]:
    """Look up parent chunk text from PostgreSQL for each retrieved child chunk.

    Enriches each doc dict with ``parent_text`` and ``text`` (using parent
    text as the context to send to the LLM for generation).
    """
    parent_ids = [d.get("parent_chunk_id", "") for d in docs if d.get("parent_chunk_id")]
    if not parent_ids:
        return docs

    try:
        # Convert string IDs to UUIDs for the query
        parent_uuids = []
        for pid in parent_ids:
            with contextlib.suppress(ValueError, AttributeError):
                parent_uuids.append(uuid.UUID(pid))

        if not parent_uuids:
            return docs

        async with async_session() as session:
            stmt = select(Chunk).where(Chunk.id.in_(parent_uuids))
            result = await session.execute(stmt)
            parent_chunks = {str(c.id): c for c in result.scalars().all()}

        # Also get child chunk text from PG (Milvus doesn't store text)
        child_ids = [d.get("chunk_id", "") for d in docs]
        child_uuids = []
        for cid in child_ids:
            with contextlib.suppress(ValueError, AttributeError):
                child_uuids.append(uuid.UUID(cid))

        child_chunk_map: dict[str, Chunk] = {}
        if child_uuids:
            async with async_session() as session:
                stmt = select(Chunk).where(Chunk.id.in_(child_uuids))
                result = await session.execute(stmt)
                child_chunk_map = {str(c.id): c for c in result.scalars().all()}

        for doc in docs:
            chunk_id = doc.get("chunk_id", "")
            parent_id = doc.get("parent_chunk_id", "")

            # Set child text from PG
            if chunk_id in child_chunk_map:
                doc["text"] = child_chunk_map[chunk_id].content

            # Set parent text for richer context
            if parent_id in parent_chunks:
                doc["parent_text"] = parent_chunks[parent_id].content
                # Use parent text as the primary context for LLM generation
                doc["text"] = parent_chunks[parent_id].content

    except Exception:
        logger.exception("Failed to enrich results with parent chunk text")

    return docs


async def rerank_node(state: RAGState) -> dict[str, Any]:
    """Cross-encoder reranking of retrieved documents.

    Uses BGE-reranker-v2-m3 to rescore (query, document) pairs and
    selects the top-k most relevant.
    """
    t0 = time.monotonic()
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        latency = state.get("latency_ms", {})
        latency["rerank"] = elapsed_ms
        return {"reranked_docs": [], "latency_ms": latency}

    # Ensure all docs have text for reranking
    docs_with_text = [d for d in retrieved_docs if d.get("text")]
    if not docs_with_text:
        logger.warning("No documents have text for reranking — passing through")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        latency = state.get("latency_ms", {})
        latency["rerank"] = elapsed_ms
        return {"reranked_docs": retrieved_docs, "latency_ms": latency}

    try:
        reranked = await asyncio.to_thread(
            reranker_service.rerank,
            query,
            docs_with_text,
            settings.reranker_top_k,
        )
    except Exception:
        logger.exception("Reranking failed — passing through retrieved docs")
        reranked = docs_with_text[: settings.reranker_top_k]

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["rerank"] = elapsed_ms

    logger.info("rerank: %d → %d docs elapsed=%dms", len(docs_with_text), len(reranked), elapsed_ms)

    return {
        "reranked_docs": reranked,
        "latency_ms": latency,
    }


async def grade_documents_node(state: RAGState) -> dict[str, Any]:
    """Grade retrieved documents for relevance using LLM-as-judge.

    Evaluates each document against the query. If the average relevance
    score falls below the configured threshold, marks ``is_relevant=False``
    to trigger query rewrite.
    """
    t0 = time.monotonic()
    query = state["query"]
    reranked_docs = state.get("reranked_docs", [])
    llm = get_llm()

    if not reranked_docs:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        latency = state.get("latency_ms", {})
        latency["grade_documents"] = elapsed_ms
        return {
            "relevance_scores": [],
            "is_relevant": False,
            "latency_ms": latency,
        }

    # ── Fast-path: trust reranker when confidence is high ────────────────────
    # Cross-encoder reranker score is a reliable relevance signal.
    # Skip expensive LLM grading if the top reranker score exceeds threshold.
    top_rerank_score = max(doc.get("rerank_score", 0.0) for doc in reranked_docs)
    if top_rerank_score >= settings.reranker_confidence_threshold:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        latency = state.get("latency_ms", {})
        latency["grade_documents"] = elapsed_ms
        logger.info(
            "grade_documents: FAST-PATH top_rerank_score=%.4f >= %.2f → is_relevant=True elapsed=%dms",
            top_rerank_score,
            settings.reranker_confidence_threshold,
            elapsed_ms,
        )
        return {
            "relevance_scores": [1.0] * len(reranked_docs),
            "is_relevant": True,
            "latency_ms": latency,
        }

    scores: list[float] = []

    for doc_idx, doc in enumerate(reranked_docs):
        doc_text = doc.get("text", "")[:1000]  # Truncate for grading prompt
        chunk_id_short = str(doc.get("chunk_id", "?"))[-8:]  # last 8 chars for readability
        try:
            prompt = GRADE_DOCUMENT_PROMPT.format(query=query, document=doc_text)
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            parsed = _safe_parse_json(
                content,
                fallback={"relevant": False, "reason": "parse_error"},
            )

            is_rel = parsed.get("relevant", False)
            reason = parsed.get("reason", "")
            score = 1.0 if is_rel else 0.0
            scores.append(score)

            logger.debug(
                "grade_documents[%d/%d] chunk=...%s relevant=%s reason='%s' "
                "raw_output='%s'",
                doc_idx + 1,
                len(reranked_docs),
                chunk_id_short,
                is_rel,
                reason[:120],
                content[:200],
            )

        except Exception:
            logger.exception(
                "Grading LLM call failed for document %d/%d (chunk=...%s)",
                doc_idx + 1,
                len(reranked_docs),
                chunk_id_short,
            )
            scores.append(0.0)

    avg_relevance = sum(scores) / len(scores) if scores else 0.0
    is_relevant = avg_relevance >= settings.grading_relevance_threshold

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["grade_documents"] = elapsed_ms

    logger.info(
        "grade_documents: avg_relevance=%.2f is_relevant=%s elapsed=%dms",
        avg_relevance,
        is_relevant,
        elapsed_ms,
    )

    return {
        "relevance_scores": scores,
        "is_relevant": is_relevant,
        "latency_ms": latency,
    }


async def rewrite_query_node(state: RAGState) -> dict[str, Any]:
    """Rewrite the query to improve retrieval on the next attempt.

    Uses LLM to generate a reformulated query based on the original
    query and the documents that were deemed insufficiently relevant.
    """
    t0 = time.monotonic()
    query = state["query"]
    reranked_docs = state.get("reranked_docs", [])
    rewrite_count = state.get("rewrite_count", 0)
    llm = get_llm()

    # Summarise top docs for the prompt
    doc_summaries = "\n".join(
        f"- {doc.get('text', '')[:200]}" for doc in reranked_docs[:3]
    )

    try:
        prompt = REWRITE_QUERY_PROMPT.format(
            query=query,
            doc_summaries=doc_summaries or "(no documents retrieved)",
        )
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = content.strip()

        if not rewritten:
            rewritten = query

    except Exception:
        logger.exception("rewrite_query LLM call failed — keeping original query")
        rewritten = query

    new_rewrite_count = rewrite_count + 1
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["rewrite_query"] = elapsed_ms

    logger.info(
        "rewrite_query: attempt=%d rewritten='%s' elapsed=%dms",
        new_rewrite_count,
        rewritten[:80],
        elapsed_ms,
    )

    return {
        "query": rewritten,
        "rewritten_query": rewritten,
        "rewrite_count": new_rewrite_count,
        "latency_ms": latency,
    }


async def consistency_check_node(state: RAGState) -> dict[str, Any]:
    """Check answer consistency (V1 — optional, MVP skips via routing).

    Generates two independent answers from the same documents and compares
    them. If they contradict, extracts the divergence point for re-retrieval.

    When ``settings.consistency_check_enabled`` is ``False`` (MVP default),
    this node returns ``consistency_passed=None`` and the router skips to
    ``generate``.

    Architecture reference: §4.2.7 (inspired by MA-RAG, arxiv 2603.03292).
    """
    t0 = time.monotonic()

    # MVP: skip consistency check
    if not settings.consistency_check_enabled:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        latency = state.get("latency_ms", {})
        latency["consistency_check"] = elapsed_ms
        return {
            "consistency_passed": None,
            "divergence_query": None,
            "latency_ms": latency,
        }

    # ── Full implementation (V1) ──────────────────────────────────────────
    query = state["query"]
    reranked_docs = state.get("reranked_docs", [])
    llm = get_llm()

    # Format docs for the prompt
    docs_context = _format_docs_for_prompt(reranked_docs)

    try:
        # Generate two answers with different temperatures
        prompt = GENERATE_PROMPT.format(query=query, context=docs_context)

        # Answer A — lower temperature
        from langchain_core.messages import HumanMessage

        response_a = await llm.ainvoke(
            [HumanMessage(content=prompt)],
            temperature=0.3,
        )
        answer_a = response_a.content if hasattr(response_a, "content") else str(response_a)

        # Answer B — higher temperature
        response_b = await llm.ainvoke(
            [HumanMessage(content=prompt)],
            temperature=0.7,
        )
        answer_b = response_b.content if hasattr(response_b, "content") else str(response_b)

        # Compare consistency
        consistency_prompt = CONSISTENCY_CHECK_PROMPT.format(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        response_c = await llm.ainvoke(consistency_prompt)
        content = response_c.content if hasattr(response_c, "content") else str(response_c)

        parsed = _safe_parse_json(
            content,
            fallback={"consistent": True, "divergence": None},
        )

        is_consistent = parsed.get("consistent", True)
        divergence = parsed.get("divergence")

        if not is_consistent and divergence:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            latency = state.get("latency_ms", {})
            latency["consistency_check"] = elapsed_ms

            logger.info(
                "consistency_check: DIVERGENT divergence='%s' elapsed=%dms",
                str(divergence)[:80],
                elapsed_ms,
            )

            return {
                "consistency_passed": False,
                "divergence_query": str(divergence),
                "query": str(divergence),  # re-retrieve with divergence query
                "rewrite_count": state.get("rewrite_count", 0) + 1,
                "latency_ms": latency,
            }

    except Exception:
        logger.exception("consistency_check failed — treating as passed")

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["consistency_check"] = elapsed_ms

    return {
        "consistency_passed": True,
        "divergence_query": None,
        "latency_ms": latency,
    }


async def generate_node(state: RAGState) -> dict[str, Any]:
    """Generate the final answer with source citations.

    Formats reranked documents as context and uses the LLM to produce
    an answer with ``[Source N]`` citations.

    When ``state["conversation_id"]`` is set, conversation memory context
    (short-term history + episode summaries) is injected into the prompt
    and the user query / assistant answer are persisted to the DB.
    """
    t0 = time.monotonic()
    query = state["query"]
    reranked_docs = state.get("reranked_docs", [])
    session_id: str | None = state.get("conversation_id")
    llm = get_llm()

    # ── Memory context injection ──────────────────────────────────────────
    memory_section = ""
    if session_id:
        try:
            from app.memory.manager import ConversationMemoryManager  # local import avoids cycles
            from app.db.engine import async_session as _async_session

            mgr = ConversationMemoryManager()
            async with _async_session() as mem_db:
                ctx = await mgr.get_context_for_query(session_id, mem_db)

            short_term = ctx.get("short_term", [])
            episodes = ctx.get("episodes", [])

            memory_lines: list[str] = []

            if short_term:
                memory_lines.append("[Conversation History]")
                for msg in short_term[-10:]:  # last 5 turns (user+assistant = 10 msgs)
                    speaker = "User" if msg["role"] == "user" else "Assistant"
                    memory_lines.append(f"{speaker}: {msg['content']}")
                memory_lines.append("")

            if episodes:
                memory_lines.append("[Previous Context]")
                for ep in episodes:
                    memory_lines.append(ep["content"])
                memory_lines.append("")

            if memory_lines:
                memory_section = "\n".join(memory_lines) + "\n"

        except Exception:
            logger.exception("generate_node: failed to load memory context (session=%s)", session_id)

    # Format context
    docs_context = _format_docs_for_prompt(reranked_docs)

    # Batch-fetch document titles from the DB so sources show human-readable names
    doc_id_to_title: dict[str, str] = {}
    try:
        raw_doc_ids = [doc.get("doc_id") for doc in reranked_docs if doc.get("doc_id")]
        if raw_doc_ids:
            doc_uuids = []
            for raw in raw_doc_ids:
                with contextlib.suppress(Exception):
                    doc_uuids.append(uuid.UUID(str(raw)))
            if doc_uuids:
                async with async_session() as session:
                    result = await session.execute(
                        select(Document.id, Document.title).where(Document.id.in_(doc_uuids))
                    )
                    for row in result.all():
                        doc_id_to_title[str(row.id)] = row.title
    except Exception:
        logger.warning("Failed to fetch document titles — falling back to index labels")

    # Build sources list
    sources: list[dict] = []
    for idx, doc in enumerate(reranked_docs):
        raw_doc_id = doc.get("doc_id", "")
        doc_title = doc_id_to_title.get(str(raw_doc_id), f"Document {idx + 1}")
        sources.append({
            "doc_title": doc_title,
            "chunk_text": (doc.get("text", ""))[:500],
            "source_url": doc.get("source_url", ""),
            "relevance_score": doc.get("rerank_score", doc.get("rrf_score", 0.0)),
            "chunk_id": doc.get("chunk_id", ""),
        })

    try:
        prompt = GENERATE_PROMPT.format(
            query=query,
            context=docs_context,
            memory_section=memory_section,
        )
        response = await llm.ainvoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("generate LLM call failed")
        if reranked_docs:
            answer = (
                "I encountered an error generating the answer. "
                "Here are the most relevant documents I found:\n\n"
                + "\n\n".join(
                    f"[Source {i+1}]: {d.get('text', '')[:300]}"
                    for i, d in enumerate(reranked_docs[:3])
                )
            )
        else:
            answer = "I couldn't find relevant information to answer your question."

    # ── Persist messages to memory (best-effort) ──────────────────────────
    if session_id:
        try:
            from app.memory.manager import ConversationMemoryManager  # noqa: F811
            from app.db.engine import async_session as _async_session  # noqa: F811

            mgr = ConversationMemoryManager()
            # Use the original query stored in state (before any rewrites for this turn)
            original_query: str = state.get("query", query)
            async with _async_session() as mem_db:
                await mgr.add_message(session_id, "user", original_query, mem_db)
                await mgr.add_message(session_id, "assistant", answer, mem_db)
        except Exception:
            logger.exception("generate_node: failed to persist messages (session=%s)", session_id)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    latency = state.get("latency_ms", {})
    latency["generate"] = elapsed_ms

    logger.info("generate: answer_len=%d elapsed=%dms", len(answer), elapsed_ms)

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════


def _format_docs_for_prompt(docs: list[dict]) -> str:
    """Format documents into a numbered context block for LLM prompts."""
    if not docs:
        return "(No documents available)"

    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        text = doc.get("text", "(no content)")
        score = doc.get("rerank_score", doc.get("rrf_score", 0.0))
        parts.append(f"[Source {idx}] (score={score:.4f}):\n{text}")

    return "\n\n".join(parts)
