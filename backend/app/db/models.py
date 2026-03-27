"""SQLAlchemy ORM models for ToolRef.

Tables:
    - documents: uploaded document metadata
    - chunks: document chunks with hierarchy
    - query_history: RAG query audit trail
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ── Enums ────────────────────────────────────────────────────────────────────


class DocType(enum.StrEnum):
    """Supported document types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"


class DocStatus(enum.StrEnum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Models ───────────────────────────────────────────────────────────────────


class Document(Base):
    """A document uploaded into the knowledge base."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    doc_type: Mapped[DocType] = mapped_column(Enum(DocType), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, comment="SHA-256")
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocStatus] = mapped_column(
        Enum(DocStatus), nullable=False, default=DocStatus.PENDING
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id!s}, title={self.title!r})>"


class Chunk(Base):
    """A document chunk with optional parent-child hierarchy."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Milvus chunk_id"
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id!s}, doc={self.document_id!s}, idx={self.chunk_index})>"


class QueryHistory(Base):
    """Audit log for RAG queries."""

    __tablename__ = "query_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    model_used: Mapped[str] = mapped_column(String(128), nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rewrite_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<QueryHistory(id={self.id!s}, query={self.query[:40]!r})>"
