"""End-to-end document ingestion pipeline.

Orchestrates the full flow:

1. Download file from MinIO
2. Parse document into structural elements
3. Hierarchical chunking (parent + child)
4. Generate embeddings (dense + sparse)
5. Write child chunk vectors to Milvus
6. Write parent + child chunks to PostgreSQL
7. Update document status
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from sqlalchemy import update

from app.db.engine import async_session
from app.db.models import Chunk, DocStatus, DocType, Document
from app.ingestion.chunker import ChildChunk, ChunkConfig, ParentChunk, chunk_document
from app.ingestion.embedder import embedding_service
from app.ingestion.parser import parse_document
from app.storage.minio import download_file
from app.vectorstore.milvus import CHILD_CHUNKS_COLLECTION

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Stateless pipeline that processes a single document end-to-end.

    Each public method corresponds to one stage; :meth:`process` runs
    them all in sequence with timing and error handling.
    """

    # ── Stage helpers ────────────────────────────────────────────────────

    @staticmethod
    async def _download(object_name: str) -> bytes:
        """Stage 1: Download file bytes from MinIO."""
        return await asyncio.to_thread(download_file, object_name)

    @staticmethod
    async def _parse(file_bytes: bytes, doc_type: DocType, metadata: dict[str, Any]) -> list:
        """Stage 2: Parse document into elements."""
        return await asyncio.to_thread(parse_document, file_bytes, doc_type, metadata)

    @staticmethod
    async def _chunk(elements: list) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Stage 3: Hierarchical chunking."""
        config = ChunkConfig()
        return await asyncio.to_thread(chunk_document, elements, config)

    @staticmethod
    async def _embed(texts: list[str]) -> tuple:
        """Stage 4: Generate embeddings."""
        return await asyncio.to_thread(embedding_service.embed_texts, texts)

    @staticmethod
    async def _write_milvus(
        child_chunks: list[ChildChunk],
        dense_embeddings: Any,
        sparse_embeddings: list[dict],
        doc_id: str,
        namespace: str,
    ) -> None:
        """Stage 5: Insert child chunk vectors into Milvus.

        Both dense (1024-d COSINE) and sparse (SPARSE_FLOAT_VECTOR IP) fields
        are populated from the real BGE-M3 embeddings produced by
        :meth:`EmbeddingService.embed_texts`.

        Args:
            child_chunks: Child chunks to insert.
            dense_embeddings: Numpy array of shape ``(N, 1024)``.
            sparse_embeddings: List of N dicts mapping ``str(token_id)`` →
                weight, as returned by BGEM3FlagModel ``lexical_weights``.
            doc_id: UUID string of the parent document.
            namespace: Document namespace.
        """
        from pymilvus import Collection

        def _insert() -> None:
            collection = Collection(CHILD_CHUNKS_COLLECTION)

            chunk_ids: list[str] = []
            doc_ids: list[str] = []
            parent_ids: list[str] = []
            namespaces: list[str] = []
            dense_list: list[list[float]] = []
            sparse_list: list[dict[int, float]] = []

            for idx, child in enumerate(child_chunks):
                chunk_ids.append(child.chunk_id)
                doc_ids.append(doc_id)
                parent_ids.append(child.parent_chunk_id)
                namespaces.append(namespace)
                dense_list.append(dense_embeddings[idx].tolist())
                # Convert str token-id keys → int for Milvus SPARSE_FLOAT_VECTOR.
                raw_sparse = sparse_embeddings[idx]
                sparse_list.append({int(k): float(v) for k, v in raw_sparse.items()})

            data = [
                chunk_ids,
                doc_ids,
                parent_ids,
                namespaces,
                dense_list,
                sparse_list,
            ]
            collection.insert(data)
            collection.flush()

        await asyncio.to_thread(_insert)

    @staticmethod
    async def _write_postgres(
        doc_id: str,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> int:
        """Stage 6: Persist chunk rows in PostgreSQL.

        Returns:
            Total number of chunk rows created (parents + children).
        """
        async with async_session() as session, session.begin():
            # Parent chunks (no embedding_id — not indexed in Milvus).
            for pc in parent_chunks:
                session.add(
                    Chunk(
                        id=uuid.UUID(pc.chunk_id),
                        document_id=uuid.UUID(doc_id),
                        parent_chunk_id=None,
                        chunk_index=pc.chunk_index,
                        content=pc.text,
                        token_count=pc.token_count,
                        embedding_id=None,
                        metadata_=pc.metadata,
                    )
                )

            # Child chunks (embedding_id points to Milvus chunk_id).
            for cc in child_chunks:
                session.add(
                    Chunk(
                        id=uuid.UUID(cc.chunk_id),
                        document_id=uuid.UUID(doc_id),
                        parent_chunk_id=uuid.UUID(cc.parent_chunk_id),
                        chunk_index=cc.chunk_index,
                        content=cc.text,
                        token_count=cc.token_count,
                        embedding_id=cc.chunk_id,
                        metadata_=cc.metadata,
                    )
                )

        return len(parent_chunks) + len(child_chunks)

    @staticmethod
    async def _update_status(
        doc_id: str,
        status: DocStatus,
        total_chunks: int = 0,
        error: str | None = None,
    ) -> None:
        """Update the document row status and optional error metadata."""
        async with async_session() as session, session.begin():
            values: dict[str, Any] = {"status": status, "total_chunks": total_chunks}
            if error:
                values["metadata_"] = {"error": error}
            await session.execute(
                update(Document)
                .where(Document.id == uuid.UUID(doc_id))
                .values(**values)
            )

    # ── Main entry point ─────────────────────────────────────────────────

    async def process(
        self,
        doc_id: str,
        namespace: str,
        object_name: str,
        doc_type_str: str,
    ) -> None:
        """Run the complete ingestion pipeline for a single document.

        On any failure the document status is set to ``FAILED`` with the
        error message stored in the document's ``metadata`` column.

        Args:
            doc_id: UUID string of the document row.
            namespace: Document namespace.
            object_name: MinIO object key.
            doc_type_str: Document type as a string value.
        """
        logger.info("Starting ingestion for doc_id=%s", doc_id)
        overall_start = time.monotonic()

        try:
            doc_type = DocType(doc_type_str)
        except ValueError:
            await self._update_status(
                doc_id, DocStatus.FAILED, error=f"Unknown doc type: {doc_type_str}",
            )
            return

        # Mark as processing.
        await self._update_status(doc_id, DocStatus.PROCESSING)

        try:
            # Stage 1 — Download
            t0 = time.monotonic()
            file_bytes = await self._download(object_name)
            logger.info("[%s] Stage 1 (download): %.2fs", doc_id[:8], time.monotonic() - t0)

            # Stage 2 — Parse
            t0 = time.monotonic()
            elements = await self._parse(
                file_bytes, doc_type, {"doc_id": doc_id, "namespace": namespace},
            )
            logger.info(
                "[%s] Stage 2 (parse): %.2fs — %d elements",
                doc_id[:8], time.monotonic() - t0, len(elements),
            )

            if not elements:
                await self._update_status(
                    doc_id, DocStatus.FAILED, error="Parser returned no elements",
                )
                return

            # Stage 3 — Chunk
            t0 = time.monotonic()
            parent_chunks, child_chunks = await self._chunk(elements)
            logger.info(
                "[%s] Stage 3 (chunk): %.2fs — %d parents, %d children",
                doc_id[:8],
                time.monotonic() - t0,
                len(parent_chunks),
                len(child_chunks),
            )

            if not child_chunks:
                await self._update_status(
                    doc_id, DocStatus.FAILED, error="Chunker produced no child chunks",
                )
                return

            # Stage 4 — Embed (child chunks only — they are indexed in Milvus)
            t0 = time.monotonic()
            child_texts = [c.text for c in child_chunks]
            dense_embeddings, sparse_embeddings = await self._embed(child_texts)
            logger.info("[%s] Stage 4 (embed): %.2fs", doc_id[:8], time.monotonic() - t0)

            # Stage 5 — Write to Milvus
            t0 = time.monotonic()
            await self._write_milvus(
                child_chunks, dense_embeddings, sparse_embeddings, doc_id, namespace,
            )
            logger.info("[%s] Stage 5 (milvus): %.2fs", doc_id[:8], time.monotonic() - t0)

            # Stage 6 — Write to PostgreSQL
            t0 = time.monotonic()
            total_chunks = await self._write_postgres(doc_id, parent_chunks, child_chunks)
            logger.info(
                "[%s] Stage 6 (postgres): %.2fs — %d rows",
                doc_id[:8], time.monotonic() - t0, total_chunks,
            )

            # Stage 7 — Update status
            await self._update_status(doc_id, DocStatus.COMPLETED, total_chunks=total_chunks)

            elapsed = time.monotonic() - overall_start
            logger.info(
                "Ingestion complete for doc_id=%s — %d chunks in %.2fs",
                doc_id,
                total_chunks,
                elapsed,
            )

        except Exception as exc:
            logger.exception("Ingestion failed for doc_id=%s", doc_id)
            await self._update_status(doc_id, DocStatus.FAILED, error=str(exc))
