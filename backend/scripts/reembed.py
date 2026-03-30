"""Re-embed all child chunks with real BGE-M3 sparse embeddings.

Usage (inside Docker container):
    docker exec toolref-backend python3 -m scripts.reembed

Usage (local, from repo root):
    python3 backend/scripts/reembed.py

What it does:
1. Reads all child chunks from PostgreSQL (chunks where parent_chunk_id IS NOT NULL)
2. Re-encodes every chunk text with the current embedding_service (BGE-M3)
3. Drops and recreates the Milvus child_chunks collection (fresh schema + indexes)
4. Bulk-inserts all chunks with real dense + sparse embeddings in batches
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Logging — set up before any app imports so we see model-load messages too
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("reembed")

# ---------------------------------------------------------------------------
# App imports (must come after logging setup)
# ---------------------------------------------------------------------------
import psycopg  # noqa: E402  (psycopg v3 sync driver)
from pymilvus import Collection, connections, utility  # noqa: E402

from app.config import settings  # noqa: E402
from app.ingestion.embedder import embedding_service  # noqa: E402
from app.vectorstore.milvus import (  # noqa: E402
    CHILD_CHUNKS_COLLECTION,
    HNSW_INDEX_PARAMS,
    SPARSE_INDEX_PARAMS,
    _child_chunks_schema,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE: int = settings.embedding_batch_size  # reuse app setting (default 32)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ChildChunkRow:
    """A single child-chunk row read from PostgreSQL."""

    chunk_id: str        # chunks.id (UUID string) == Milvus primary key
    doc_id: str          # chunks.document_id
    parent_chunk_id: str # chunks.parent_chunk_id
    namespace: str       # documents.namespace
    text: str            # chunks.content


# ---------------------------------------------------------------------------
# Step 1 — Read child chunks from PostgreSQL
# ---------------------------------------------------------------------------

def _sync_db_url() -> str:
    """Build a psycopg (sync, v3) connection string from settings."""
    return (
        f"host={settings.postgres_host} "
        f"port={settings.postgres_port} "
        f"dbname={settings.postgres_db} "
        f"user={settings.postgres_user} "
        f"password={settings.postgres_password}"
    )


def fetch_child_chunks() -> list[ChildChunkRow]:
    """Return every child chunk row joined with its document namespace."""
    logger.info("Connecting to PostgreSQL at %s:%s …", settings.postgres_host, settings.postgres_port)
    connstr = _sync_db_url()

    with psycopg.connect(connstr) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id::text          AS chunk_id,
                    c.document_id::text AS doc_id,
                    c.parent_chunk_id::text AS parent_chunk_id,
                    d.namespace,
                    c.content           AS text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.parent_chunk_id IS NOT NULL   -- child chunks only
                ORDER BY c.document_id, c.chunk_index
                """
            )
            rows = cur.fetchall()

    chunks = [
        ChildChunkRow(
            chunk_id=row[0],
            doc_id=row[1],
            parent_chunk_id=row[2],
            namespace=row[3],
            text=row[4],
        )
        for row in rows
    ]
    logger.info("Fetched %d child chunks from PostgreSQL", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Step 2 — Drop and recreate Milvus collection
# ---------------------------------------------------------------------------

def reset_milvus_collection() -> None:
    """Drop the child-chunks collection and recreate it with indexes."""
    logger.info(
        "Connecting to Milvus at %s:%s …", settings.milvus_host, settings.milvus_port
    )
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )

    if utility.has_collection(CHILD_CHUNKS_COLLECTION):
        logger.info("Dropping existing collection '%s' …", CHILD_CHUNKS_COLLECTION)
        utility.drop_collection(CHILD_CHUNKS_COLLECTION)
        logger.info("Collection dropped.")

    logger.info("Creating collection '%s' …", CHILD_CHUNKS_COLLECTION)
    schema = _child_chunks_schema()
    collection = Collection(name=CHILD_CHUNKS_COLLECTION, schema=schema)

    logger.info("Building dense (HNSW) index …")
    collection.create_index(field_name="dense_embedding", index_params=HNSW_INDEX_PARAMS)

    logger.info("Building sparse (SPARSE_INVERTED_INDEX) index …")
    collection.create_index(field_name="sparse_embedding", index_params=SPARSE_INDEX_PARAMS)

    collection.load()
    logger.info("Collection '%s' recreated and loaded.", CHILD_CHUNKS_COLLECTION)


# ---------------------------------------------------------------------------
# Step 3 — Embed + insert in batches
# ---------------------------------------------------------------------------

def insert_batch(
    collection: Collection,
    batch: list[ChildChunkRow],
    dense_embeddings: Any,
    sparse_embeddings: list[dict],
) -> None:
    """Insert one batch of embedded chunks into Milvus."""
    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    parent_ids: list[str] = []
    namespaces: list[str] = []
    dense_list: list[list[float]] = []
    sparse_list: list[dict[int, float]] = []

    for idx, row in enumerate(batch):
        chunk_ids.append(row.chunk_id)
        doc_ids.append(row.doc_id)
        parent_ids.append(row.parent_chunk_id)
        namespaces.append(row.namespace)
        dense_list.append(dense_embeddings[idx].tolist())
        # Convert str token-id keys → int for Milvus SPARSE_FLOAT_VECTOR
        raw_sparse = sparse_embeddings[idx]
        sparse_list.append({int(k): float(v) for k, v in raw_sparse.items()})

    data = [chunk_ids, doc_ids, parent_ids, namespaces, dense_list, sparse_list]
    collection.insert(data)
    collection.flush()


def reembed_and_insert(chunks: list[ChildChunkRow]) -> tuple[int, int]:
    """Embed all chunks in batches and insert into Milvus.

    Returns:
        (success_count, failure_count)
    """
    if not chunks:
        logger.warning("No chunks to process — nothing to do.")
        return 0, 0

    logger.info(
        "Warming up embedding model '%s' …", settings.embedding_model
    )
    embedding_service.warmup()

    collection = Collection(CHILD_CHUNKS_COLLECTION)
    total = len(chunks)
    success = 0
    failure = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        batch_total = (total + BATCH_SIZE - 1) // BATCH_SIZE
        texts = [row.text for row in batch]

        try:
            t0 = time.monotonic()
            dense_embeddings, sparse_embeddings = embedding_service.embed_texts(texts)
            embed_secs = time.monotonic() - t0

            t1 = time.monotonic()
            insert_batch(collection, batch, dense_embeddings, sparse_embeddings)
            insert_secs = time.monotonic() - t1

            success += len(batch)
            logger.info(
                "[batch %d/%d] chunks %d–%d  |  embed %.2fs  insert %.2fs  |  "
                "cumulative success=%d failure=%d",
                batch_num,
                batch_total,
                batch_start + 1,
                batch_start + len(batch),
                embed_secs,
                insert_secs,
                success,
                failure,
            )

        except Exception:
            failure += len(batch)
            logger.exception(
                "[batch %d/%d] FAILED for chunks %d–%d — skipping batch",
                batch_num,
                batch_total,
                batch_start + 1,
                batch_start + len(batch),
            )

    return success, failure


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    overall_start = time.monotonic()
    logger.info("=" * 60)
    logger.info("  ToolRef re-embed script")
    logger.info("  Model : %s", settings.embedding_model)
    logger.info("  Dim   : %d", settings.embedding_dim)
    logger.info("  Batch : %d", BATCH_SIZE)
    logger.info("=" * 60)

    # 1 — Fetch
    chunks = fetch_child_chunks()
    total = len(chunks)

    if total == 0:
        logger.warning("PostgreSQL has no child chunks — nothing to re-embed.")
        return

    # 2 — Reset collection
    reset_milvus_collection()

    # 3 — Embed + insert
    success, failure = reembed_and_insert(chunks)

    elapsed = time.monotonic() - overall_start
    logger.info("=" * 60)
    logger.info("  Re-embed complete in %.1fs", elapsed)
    logger.info("  Total chunks : %d", total)
    logger.info("  Success      : %d", success)
    logger.info("  Failed       : %d", failure)
    logger.info("=" * 60)

    if failure > 0:
        logger.warning(
            "%d chunks failed — Milvus index is INCOMPLETE. "
            "Review the error logs above and re-run the script.",
            failure,
        )
        sys.exit(1)
    else:
        logger.info("All chunks successfully re-embedded with real BGE-M3 sparse vectors.")


if __name__ == "__main__":
    main()
