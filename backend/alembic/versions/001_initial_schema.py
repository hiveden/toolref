"""001 — initial schema: documents, chunks, query_history.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-03-26
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create initial tables."""
    # ── documents ──────��──────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace", sa.String(128), nullable=False, index=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column(
            "doc_type",
            sa.Enum("pdf", "markdown", "html", "txt", name="doctype"),
            nullable=False,
        ),
        sa.Column("source_url", sa.Text, nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=False, comment="SHA-256"),
        sa.Column("total_chunks", sa.Integer, server_default="0"),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "completed", "failed", name="docstatus"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # ── chunks ────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parent_chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column(
            "embedding_id", sa.String(64), nullable=True, comment="Milvus chunk_id"
        ),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # ── query_history ─────────────────────────────────────────────────────
    op.create_table(
        "query_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace", sa.String(128), nullable=False, index=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("sources", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("model_used", sa.String(128), nullable=False),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("rewrite_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    """Drop initial tables."""
    op.drop_table("query_history")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS doctype")
    op.execute("DROP TYPE IF EXISTS docstatus")
