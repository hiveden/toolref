"""RAG query REST API.

Endpoints:
    POST  /api/v1/query          — Execute a RAG query (JSON response)
    POST  /api/v1/query/stream   — Execute a RAG query (SSE streaming)
    GET   /api/v1/query/history  — List query history
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis import get_redis
from app.config import settings
from app.db.engine import get_session
from app.db.models import QueryHistory
from app.retrieval.cache import SemanticCache
from app.retrieval.graph import rag_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["query"])


# ── Request / Response models ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """RAG query request payload."""

    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    namespace: str = Field(default="default", max_length=128, description="Document namespace")
    conversation_id: str | None = Field(default=None, description="Conversation ID for context")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source documents")
    use_cache: bool = Field(default=True, description="Whether to check semantic cache")


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[dict[str, Any]]
    cached: bool
    latency_ms: int
    rewrite_count: int


# ── POST /api/v1/query ──────────────────────────────────────────────────────


@router.post("", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    """Execute an Agentic RAG query.

    Flow:
      1. Check semantic cache (if enabled).
      2. On cache miss → run LangGraph pipeline.
      3. Cache the result.
      4. Persist to QueryHistory.
      5. Return answer + sources.
    """
    t_start = time.monotonic()

    # ── 1. Check semantic cache ───────────────────────────────────────────
    if request.use_cache:
        try:
            redis_client = await get_redis()
            cache = SemanticCache(redis_client)
            cached_result = await cache.get(request.query, request.namespace)

            if cached_result is not None:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                logger.info(
                    "Cache HIT for query='%s' latency=%dms",
                    request.query[:60],
                    latency_ms,
                )

                # Persist cache hit to history
                await _save_query_history(
                    session=session,
                    query=request.query,
                    namespace=request.namespace,
                    answer=cached_result.get("answer", ""),
                    sources=cached_result.get("sources"),
                    latency_ms=latency_ms,
                    cache_hit=True,
                    rewrite_count=0,
                )

                return QueryResponse(
                    answer=cached_result.get("answer", ""),
                    sources=cached_result.get("sources", []),
                    cached=True,
                    latency_ms=latency_ms,
                    rewrite_count=0,
                )
        except Exception:
            logger.exception("Semantic cache check failed — proceeding without cache")

    # ── 2. Run LangGraph pipeline ─────────────────────────────────────────
    initial_state = {
        "query": request.query,
        "namespace": request.namespace,
        "conversation_id": request.conversation_id,
        "query_type": None,
        "sub_queries": [],
        "entities": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "relevance_scores": [],
        "is_relevant": None,
        "rewrite_count": 0,
        "rewritten_query": None,
        "consistency_passed": None,
        "divergence_query": None,
        "answer": None,
        "sources": [],
        "cached": False,
        "messages": [],
        "latency_ms": {},
    }

    try:
        result = await rag_graph.ainvoke(initial_state)
    except Exception:
        logger.exception("RAG graph execution failed for query='%s'", request.query[:60])
        result = {
            "answer": "I'm sorry, I encountered an error processing your query. Please try again.",
            "sources": [],
            "rewrite_count": 0,
            "latency_ms": {},
        }

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    rewrite_count = result.get("rewrite_count", 0)

    total_latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── 3. Cache the result ───────────────────────────────────────────────
    # Only cache if no rewrites occurred (per architecture §4.4.1)
    if request.use_cache and rewrite_count == 0:
        try:
            redis_client = await get_redis()
            cache = SemanticCache(redis_client)
            await cache.put(
                query=request.query,
                namespace=request.namespace,
                result={"answer": answer, "sources": sources},
            )
        except Exception:
            logger.exception("Failed to cache RAG result")

    # ── 4. Persist to QueryHistory ────────────────────────────────────────
    await _save_query_history(
        session=session,
        query=request.query,
        namespace=request.namespace,
        answer=answer,
        sources=sources,
        latency_ms=total_latency_ms,
        cache_hit=False,
        rewrite_count=rewrite_count,
    )

    logger.info(
        "Query completed: query='%s' latency=%dms rewrites=%d",
        request.query[:60],
        total_latency_ms,
        rewrite_count,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        cached=False,
        latency_ms=total_latency_ms,
        rewrite_count=rewrite_count,
    )


# ── POST /api/v1/query/stream ────────────────────────────────────────────────


@router.post("/stream")
async def execute_query_stream(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Execute an Agentic RAG query and stream the answer via Server-Sent Events.

    SSE event types
    ---------------
    ``{"type": "chunk",  "content": "<token>"}``
        Incremental answer fragment — append to the message bubble.

    ``{"type": "done",   "sources": [...], "cached": bool,
       "latency_ms": int, "rewrite_count": int}``
        Pipeline finished.  Contains all metadata.

    ``{"type": "error",  "message": "<detail>"}``
        Unrecoverable error.
    """

    async def _sse_generator() -> AsyncIterator[str]:
        t_start = time.monotonic()

        # ── 1. Semantic cache ─────────────────────────────────────────────
        cached_answer: str | None = None
        cached_sources: list[dict] = []

        if request.use_cache:
            try:
                redis_client = await get_redis()
                cache = SemanticCache(redis_client)
                cached_result = await cache.get(request.query, request.namespace)

                if cached_result is not None:
                    cached_answer = cached_result.get("answer", "")
                    cached_sources = cached_result.get("sources", [])
            except Exception:
                logger.exception("Semantic cache check failed — proceeding without cache")

        if cached_answer is not None:
            # Stream cached answer word-by-word
            words = cached_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.018)  # ~55 tokens/s

            latency_ms = int((time.monotonic() - t_start) * 1000)
            done_payload = {
                'type': 'done', 'sources': cached_sources,
                'cached': True, 'latency_ms': latency_ms, 'rewrite_count': 0,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

            await _save_query_history(
                session=session,
                query=request.query,
                namespace=request.namespace,
                answer=cached_answer,
                sources=cached_sources,
                latency_ms=latency_ms,
                cache_hit=True,
                rewrite_count=0,
            )
            return

        # ── 2. Run LangGraph pipeline ─────────────────────────────────────
        initial_state = {
            "query": request.query,
            "namespace": request.namespace,
            "conversation_id": request.conversation_id,
            "query_type": None,
            "sub_queries": [],
            "entities": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "relevance_scores": [],
            "is_relevant": None,
            "rewrite_count": 0,
            "rewritten_query": None,
            "consistency_passed": None,
            "divergence_query": None,
            "answer": None,
            "sources": [],
            "cached": False,
            "messages": [],
            "latency_ms": {},
        }

        try:
            result = await rag_graph.ainvoke(initial_state)
        except Exception:
            logger.exception("RAG graph execution failed for query='%s'", request.query[:60])
            err = {'type': 'error', 'message': 'Pipeline error — please try again.'}
            yield f"data: {json.dumps(err)}\n\n"
            return

        answer: str = result.get("answer", "")
        sources: list[dict] = result.get("sources", [])
        rewrite_count: int = result.get("rewrite_count", 0)

        # ── 3. Stream answer word-by-word ─────────────────────────────────
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            await asyncio.sleep(0.018)

        total_latency_ms = int((time.monotonic() - t_start) * 1000)

        # ── 4. Send done event ────────────────────────────────────────────
        done_payload = {
            'type': 'done', 'sources': sources, 'cached': False,
            'latency_ms': total_latency_ms, 'rewrite_count': rewrite_count,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

        # ── 5. Cache & persist (best-effort, after streaming) ─────────────
        if request.use_cache and rewrite_count == 0:
            try:
                redis_client = await get_redis()
                cache = SemanticCache(redis_client)
                await cache.put(
                    query=request.query,
                    namespace=request.namespace,
                    result={"answer": answer, "sources": sources},
                )
            except Exception:
                logger.exception("Failed to cache RAG result")

        await _save_query_history(
            session=session,
            query=request.query,
            namespace=request.namespace,
            answer=answer,
            sources=sources,
            latency_ms=total_latency_ms,
            cache_hit=False,
            rewrite_count=rewrite_count,
        )

        logger.info(
            "Stream query completed: query='%s' latency=%dms rewrites=%d",
            request.query[:60],
            total_latency_ms,
            rewrite_count,
        )

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── GET /api/v1/query/history ────────────────────────────────────────────────


@router.get("/history")
async def list_query_history(
    namespace: str | None = Query(default=None, description="Filter by namespace"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List RAG query history with optional namespace filtering and pagination."""
    stmt = select(QueryHistory).order_by(QueryHistory.created_at.desc())
    count_stmt = select(func.count(QueryHistory.id))

    if namespace is not None:
        stmt = stmt.where(QueryHistory.namespace == namespace)
        count_stmt = count_stmt.where(QueryHistory.namespace == namespace)

    # Total count
    total_result = await session.execute(count_stmt)
    total: int = total_result.scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    result = await session.execute(stmt)
    entries = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": str(e.id),
                "namespace": e.namespace,
                "query": e.query,
                "answer": e.answer[:200] + "…" if len(e.answer) > 200 else e.answer,
                "sources": e.sources,
                "latency_ms": e.latency_ms,
                "model_used": e.model_used,
                "cache_hit": e.cache_hit,
                "rewrite_count": e.rewrite_count,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _save_query_history(
    *,
    session: AsyncSession,
    query: str,
    namespace: str,
    answer: str,
    sources: list[dict] | None,
    latency_ms: int,
    cache_hit: bool,
    rewrite_count: int,
) -> None:
    """Persist a query result to the QueryHistory table."""
    try:
        entry = QueryHistory(
            namespace=namespace,
            query=query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            model_used=f"{settings.llm_provider}/{settings.llm_model}",
            cache_hit=cache_hit,
            rewrite_count=rewrite_count,
        )
        session.add(entry)
        await session.commit()
    except Exception:
        logger.exception("Failed to save query history")
        await session.rollback()
