"""Conversation Memory Manager for ToolRef.

Implements a four-layer memory architecture:
    1. Short-term memory  — recent ``ConversationMessage`` rows per session.
    2. Long-term memory   — compressed ``MemoryEpisode`` summaries per session.
    3. Decay / importance — importance_score assigned at compression time.
    4. Overflow / compression — when limits exceeded, older messages are
       compressed into an episode and removed from the messages table.

Usage::

    mgr = ConversationMemoryManager()

    # At query time
    ctx = await mgr.get_context_for_query(session_id, db)

    # After answer generation
    await mgr.add_message(session_id, "user", query, db)
    await mgr.add_message(session_id, "assistant", answer, db)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConversationMessage, MemoryEpisode
from app.retrieval.llm import get_llm

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
以下是一段对话历史。请提炼成 2-3 句话的摘要，保留关键信息和用户意图。

{history_text}"""


class ConversationMemoryManager:
    """Manages per-session conversation memory with overflow compression.

    Attributes:
        MAX_MESSAGES: Hard ceiling on short-term message count per session.
        MAX_TOKENS: Approximate token ceiling for short-term memory.
        KEEP_RECENT: Number of most-recent messages to *always* preserve
            when compressing (i.e. never compress these).
        SUMMARY_TRIGGER: Fraction of the limit that triggers compression
            (0.8 → compress when 80 % full).
    """

    MAX_MESSAGES: int = 20
    MAX_TOKENS: int = 8000
    KEEP_RECENT: int = 3
    SUMMARY_TRIGGER: float = 0.8

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: word-count × 1.3."""
        return int(len(text.split()) * 1.3)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get_history(
        self,
        session_id: str,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """Return short-term message history for *session_id*, oldest first.

        Returns:
            List of dicts with keys ``role``, ``content``, ``created_at``.
        """
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.asc())
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        db: AsyncSession,
    ) -> None:
        """Persist a single message and check for overflow.

        Args:
            session_id: Arbitrary string identifying the conversation.
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
            db: Active async SQLAlchemy session.
        """
        token_count = self._estimate_tokens(content)
        msg = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        db.add(msg)
        await db.commit()
        logger.debug(
            "add_message: session=%s role=%s tokens=%d",
            session_id,
            role,
            token_count,
        )

        # Check overflow asynchronously — best-effort
        try:
            await self._check_overflow(session_id, db)
        except Exception:
            logger.exception(
                "add_message: overflow check failed for session=%s", session_id
            )

    async def get_context_for_query(
        self,
        session_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Build a context dict for injecting into the generation prompt.

        Returns:
            ``{"short_term": [...], "episodes": [...]}``

            * ``short_term`` — all current :class:`ConversationMessage` rows
              (oldest first).
            * ``episodes`` — up to 5 most-recent :class:`MemoryEpisode` rows
              (newest first).
        """
        short_term = await self.get_history(session_id, db)

        ep_stmt = (
            select(MemoryEpisode)
            .where(MemoryEpisode.session_id == session_id)
            .order_by(MemoryEpisode.created_at.desc())
            .limit(5)
        )
        ep_result = await db.execute(ep_stmt)
        episodes = ep_result.scalars().all()

        # Update access stats for retrieved episodes
        for ep in episodes:
            ep.access_count = (ep.access_count or 0) + 1
            ep.last_accessed_at = datetime.now(tz=timezone.utc)
        if episodes:
            await db.commit()

        return {
            "short_term": short_term,
            "episodes": [
                {
                    "content": ep.content,
                    "importance_score": ep.importance_score,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                }
                for ep in episodes
            ],
        }

    # ------------------------------------------------------------------
    # Internal — overflow logic
    # ------------------------------------------------------------------

    async def _check_overflow(self, session_id: str, db: AsyncSession) -> None:
        """Trigger compression when short-term memory nears its limits."""
        # Count messages
        count_stmt = select(func.count(ConversationMessage.id)).where(
            ConversationMessage.session_id == session_id
        )
        count_result = await db.execute(count_stmt)
        msg_count: int = count_result.scalar_one() or 0

        # Sum tokens
        token_stmt = select(func.sum(ConversationMessage.token_count)).where(
            ConversationMessage.session_id == session_id
        )
        token_result = await db.execute(token_stmt)
        total_tokens: int = token_result.scalar_one() or 0

        msg_threshold = int(self.MAX_MESSAGES * self.SUMMARY_TRIGGER)
        token_threshold = int(self.MAX_TOKENS * self.SUMMARY_TRIGGER)

        if msg_count >= msg_threshold or total_tokens >= token_threshold:
            logger.info(
                "_check_overflow: triggering compression "
                "session=%s msgs=%d tokens=%d",
                session_id,
                msg_count,
                total_tokens,
            )
            await self._compress_to_episode(session_id, db)

    async def _compress_to_episode(self, session_id: str, db: AsyncSession) -> None:
        """Compress older messages into a ``MemoryEpisode`` summary.

        Keeps the ``KEEP_RECENT`` most-recent messages intact; everything
        older is summarised by the LLM and then deleted.
        """
        # Fetch all messages oldest-first
        all_stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.asc())
        )
        all_result = await db.execute(all_stmt)
        all_msgs = all_result.scalars().all()

        if len(all_msgs) <= self.KEEP_RECENT:
            logger.debug(
                "_compress_to_episode: only %d msgs — nothing to compress",
                len(all_msgs),
            )
            return

        to_compress = all_msgs[: -self.KEEP_RECENT]  # everything except last N

        # Build the history text for the LLM
        history_lines: list[str] = []
        for m in to_compress:
            speaker = "用户" if m.role == "user" else "助手"
            history_lines.append(f"{speaker}: {m.content}")
        history_text = "\n".join(history_lines)

        # Call LLM to generate summary
        summary: str
        try:
            llm = get_llm()
            prompt = SUMMARY_PROMPT.format(history_text=history_text)
            response = await llm.ainvoke(prompt)
            summary = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
            if not summary:
                summary = history_text[:500]  # fallback: truncated raw history
        except Exception:
            logger.exception(
                "_compress_to_episode: LLM summarisation failed — using raw history"
            )
            summary = history_text[:500]

        # Persist episode
        episode = MemoryEpisode(
            session_id=session_id,
            content=summary,
            importance_score=0.5,
        )
        db.add(episode)

        # Delete compressed messages
        ids_to_delete = [m.id for m in to_compress]
        await db.execute(
            delete(ConversationMessage).where(
                ConversationMessage.id.in_(ids_to_delete)
            )
        )
        await db.commit()

        logger.info(
            "_compress_to_episode: compressed %d msgs → 1 episode (session=%s)",
            len(to_compress),
            session_id,
        )
