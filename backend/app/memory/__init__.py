"""Conversation memory subsystem for ToolRef.

Exposes :class:`ConversationMemoryManager` as the primary interface.
"""

from app.memory.manager import ConversationMemoryManager

__all__ = ["ConversationMemoryManager"]
