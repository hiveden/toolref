"""Session memory REST API.

Endpoints:
    GET    /api/v1/sessions                          — List distinct session IDs
    GET    /api/v1/sessions/{session_id}/history     — Full conversation history
    GET    /api/v1/sessions/{session_id}/episodes    — Long-term memory summaries
    DELETE /api/v1/sessions/{session_id}             — Wipe all data for a session
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import ConversationMessage, MemoryEpisode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


# ── GET /api/v1/sessions ─────────────────────────────────────────────────────


@router.get("")
async def list_sessions(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return all session IDs that have at least one conversation message.

    A session appears here as soon as its first message is written.
    Sessions that only have memory episodes (all messages compressed) are
    also included via a UNION.
    """
    # Sessions from messages
    msg_subq = select(
        ConversationMessage.session_id.label("session_id"),
        func.max(ConversationMessage.created_at).label("last_active"),
    ).group_by(ConversationMessage.session_id)

    # Sessions from episodes (may have had all messages compressed)
    ep_subq = select(
        MemoryEpisode.session_id.label("session_id"),
        func.max(MemoryEpisode.created_at).label("last_active"),
    ).group_by(MemoryEpisode.session_id)

    # Union and re-aggregate
    union = msg_subq.union_all(ep_subq).subquery()
    agg = (
        select(union.c.session_id, func.max(union.c.last_active).label("last_active"))
        .group_by(union.c.session_id)
        .order_by(func.max(union.c.last_active).desc())
    )

    # Total count (distinct sessions)
    count_subq = select(func.count(distinct(union.c.session_id)))
    total_result = await db.execute(count_subq)
    total: int = total_result.scalar_one() or 0

    # Paginate
    offset = (page - 1) * page_size
    paginated = agg.offset(offset).limit(page_size)
    rows_result = await db.execute(paginated)
    rows = rows_result.all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "session_id": r.session_id,
                "last_active": r.last_active.isoformat() if r.last_active else None,
            }
            for r in rows
        ],
    }


# ── GET /api/v1/sessions/{session_id}/history ───────────────────────────────


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return paginated conversation messages for a session (oldest first)."""
    base = select(ConversationMessage).where(
        ConversationMessage.session_id == session_id
    )
    count_stmt = select(func.count(ConversationMessage.id)).where(
        ConversationMessage.session_id == session_id
    )

    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one() or 0

    offset = (page - 1) * page_size
    stmt = base.order_by(ConversationMessage.created_at.asc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    msgs = result.scalars().all()

    return {
        "session_id": session_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "token_count": m.token_count,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ],
    }


# ── GET /api/v1/sessions/{session_id}/episodes ──────────────────────────────


@router.get("/{session_id}/episodes")
async def get_session_episodes(
    session_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return memory episode summaries for a session (newest first)."""
    count_stmt = select(func.count(MemoryEpisode.id)).where(
        MemoryEpisode.session_id == session_id
    )
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one() or 0

    offset = (page - 1) * page_size
    stmt = (
        select(MemoryEpisode)
        .where(MemoryEpisode.session_id == session_id)
        .order_by(MemoryEpisode.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    episodes = result.scalars().all()

    return {
        "session_id": session_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": str(ep.id),
                "content": ep.content,
                "importance_score": ep.importance_score,
                "access_count": ep.access_count,
                "last_accessed_at": (
                    ep.last_accessed_at.isoformat() if ep.last_accessed_at else None
                ),
                "created_at": ep.created_at.isoformat() if ep.created_at else None,
            }
            for ep in episodes
        ],
    }


# ── DELETE /api/v1/sessions/{session_id} ─────────────────────────────────────


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Delete all messages and memory episodes for a session.

    Returns the number of rows deleted from each table.
    """
    try:
        msg_result = await db.execute(
            delete(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
        )
        ep_result = await db.execute(
            delete(MemoryEpisode).where(MemoryEpisode.session_id == session_id)
        )
        await db.commit()
        deleted_messages = msg_result.rowcount
        deleted_episodes = ep_result.rowcount
    except Exception:
        logger.exception("delete_session: failed to delete session=%s", session_id)
        await db.rollback()
        deleted_messages = 0
        deleted_episodes = 0

    logger.info(
        "delete_session: session=%s deleted messages=%d episodes=%d",
        session_id,
        deleted_messages,
        deleted_episodes,
    )

    return {
        "session_id": session_id,
        "deleted_messages": deleted_messages,
        "deleted_episodes": deleted_episodes,
    }
