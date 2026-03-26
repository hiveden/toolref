"""Hierarchical chunking module.

Implements the two-level chunking strategy described in architecture §4.1.2:

* **Parent chunks** (~1024 tokens) — provide rich context to the LLM.
* **Child chunks** (~256 tokens) — stored with embeddings for precise retrieval.

Token counting uses ``tiktoken`` with the *cl100k_base* tokenizer (GPT-4
compatible, good proxy for BGE-M3 sub-word distribution).
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from app.config import settings
from app.ingestion.parser import DocumentElement

logger = logging.getLogger(__name__)

# Module-level tokenizer instance (thread-safe, reusable).
_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkConfig:
    """Configuration knobs for hierarchical chunking.

    Attributes:
        parent_chunk_size: Target token count for parent chunks.
        parent_overlap: Token overlap between consecutive parent chunks.
        child_chunk_size: Target token count for child chunks.
        child_overlap: Token overlap between consecutive child chunks.
    """

    parent_chunk_size: int = settings.chunk_parent_size
    parent_overlap: int = settings.chunk_parent_overlap
    child_chunk_size: int = settings.chunk_child_size
    child_overlap: int = settings.chunk_child_overlap


@dataclass
class ParentChunk:
    """A parent chunk containing broader context.

    Attributes:
        chunk_id: Unique identifier (UUID).
        text: Full text of the parent chunk.
        token_count: Number of tokens in *text*.
        chunk_index: Sequential index among all parent chunks.
        metadata: Inherited and computed metadata.
    """

    chunk_id: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A child chunk for precise retrieval.

    Attributes:
        chunk_id: Unique identifier (UUID).
        parent_chunk_id: ID of the parent chunk this child belongs to.
        text: Text content.
        token_count: Number of tokens.
        chunk_index: Global sequential index among all child chunks.
        metadata: Inherited and computed metadata.
    """

    chunk_id: str
    parent_chunk_id: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_tokenizer.encode(text))


def _split_text_by_tokens(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* tokens.

    Uses token-level slicing with configurable overlap.

    Args:
        text: The text to split.
        chunk_size: Target size of each chunk in tokens.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of text chunks.
    """
    tokens = _tokenizer.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start += step

    return chunks


# ── Public API ───────────────────────────────────────────────────────────────


def chunk_document(
    elements: list[DocumentElement],
    config: ChunkConfig | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Apply hierarchical chunking to parsed document elements.

    1. Merge all elements into a single continuous text.
    2. Split into parent-level chunks (1024 tokens by default).
    3. For each parent, split further into child chunks (256 tokens).

    Args:
        elements: Output of :func:`parse_document`.
        config: Chunking parameters.  Uses :class:`ChunkConfig` defaults
            (from ``settings``) when ``None``.

    Returns:
        A tuple ``(parent_chunks, child_chunks)`` ready for embedding
        and database storage.
    """
    if config is None:
        config = ChunkConfig()

    # 1. Merge elements into continuous text, preserving paragraph breaks.
    full_text = "\n\n".join(el.text for el in elements if el.text.strip())

    if not full_text.strip():
        logger.warning("No text content in elements — returning empty chunks")
        return [], []

    # 2. Parent-level chunking.
    parent_texts = _split_text_by_tokens(
        full_text,
        chunk_size=config.parent_chunk_size,
        overlap=config.parent_overlap,
    )

    parent_chunks: list[ParentChunk] = []
    child_chunks: list[ChildChunk] = []
    child_global_idx = 0

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = str(uuid.uuid4())
        parent_chunks.append(
            ParentChunk(
                chunk_id=parent_id,
                text=p_text,
                token_count=_count_tokens(p_text),
                chunk_index=p_idx,
                metadata={"level": "parent"},
            )
        )

        # 3. Child-level chunking within this parent.
        child_texts = _split_text_by_tokens(
            p_text,
            chunk_size=config.child_chunk_size,
            overlap=config.child_overlap,
        )

        for c_text in child_texts:
            child_chunks.append(
                ChildChunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_chunk_id=parent_id,
                    text=c_text,
                    token_count=_count_tokens(c_text),
                    chunk_index=child_global_idx,
                    metadata={"level": "child", "parent_index": p_idx},
                )
            )
            child_global_idx += 1

    logger.info(
        "Chunked document: %d parent chunks, %d child chunks",
        len(parent_chunks),
        len(child_chunks),
    )
    return parent_chunks, child_chunks
