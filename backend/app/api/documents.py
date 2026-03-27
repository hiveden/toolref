"""Document management REST API.

Endpoints:
    POST   /api/v1/documents          — Upload a new document
    GET    /api/v1/documents          — List documents (filterable)
    GET    /api/v1/documents/{doc_id} — Get document details
    DELETE /api/v1/documents/{doc_id} — Delete document + chunks + vectors
"""

import asyncio
import hashlib
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Chunk, DocStatus, DocType, Document
from app.ingestion.queue import publish
from app.storage.minio import delete_file, upload_file
from app.vectorstore.milvus import CHILD_CHUNKS_COLLECTION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Maximum upload size: 50 MB.
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024

# Map common MIME types / extensions to DocType.
_EXT_TO_DOCTYPE: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    ".html": DocType.HTML,
    ".htm": DocType.HTML,
    ".txt": DocType.TXT,
    ".text": DocType.TXT,
}


def _detect_doc_type(filename: str) -> DocType:
    """Infer :class:`DocType` from the file extension.

    Raises:
        HTTPException: If the extension is not recognised.
    """
    ext = "." + filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    doc_type = _EXT_TO_DOCTYPE.get(ext)
    if doc_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Supported: {list(_EXT_TO_DOCTYPE.keys())}",
        )
    return doc_type


# ── POST /api/v1/documents ──────────────────────────────────────────────────


@router.post("", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = Form(...),
    title: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Upload a document for asynchronous ingestion.

    Accepts a multipart file upload together with a *namespace* and an
    optional *title*.  The file is stored in MinIO, a database record is
    created, and a message is pushed to the ingestion queue.

    Returns the newly created document object.
    """
    filename = file.filename or "untitled"

    # Validate file type.
    doc_type = _detect_doc_type(filename)

    # Read file contents (enforce size limit).
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Maximum: {MAX_UPLOAD_BYTES} bytes.",
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # SHA-256 hash for deduplication.
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for duplicate (same hash in same namespace).
    dup_stmt = select(Document).where(
        Document.file_hash == file_hash,
        Document.namespace == namespace,
    )
    dup_result = await session.execute(dup_stmt)
    existing = dup_result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate document: a file with the same hash already exists "
            f"(id={existing.id})",
        )

    # Generate IDs.
    doc_id = uuid.uuid4()
    object_name = f"{namespace}/{doc_id!s}/{filename}"

    # Upload to MinIO (sync SDK → run in thread).
    content_type = file.content_type or "application/octet-stream"
    await asyncio.to_thread(upload_file, file_bytes, object_name, content_type)

    # Create DB record.
    doc = Document(
        id=doc_id,
        namespace=namespace,
        title=title or filename,
        doc_type=doc_type,
        file_hash=file_hash,
        status=DocStatus.PENDING,
        metadata_={"object_name": object_name, "original_filename": filename},
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Enqueue ingestion job.
    await publish(
        doc_id=str(doc_id),
        namespace=namespace,
        object_name=object_name,
        doc_type=doc_type.value,
    )

    logger.info("Document uploaded: id=%s namespace=%s type=%s", doc_id, namespace, doc_type.value)

    return {
        "id": str(doc.id),
        "namespace": doc.namespace,
        "title": doc.title,
        "doc_type": doc.doc_type.value,
        "file_hash": doc.file_hash,
        "status": doc.status.value,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


# ── GET /api/v1/documents ──────────────────────────────────────────────────


@router.get("")
async def list_documents(
    namespace: str | None = Query(default=None, description="Filter by namespace"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List documents with optional namespace filtering and pagination."""
    stmt = select(Document).order_by(Document.created_at.desc())
    count_stmt = select(func.count(Document.id))

    if namespace is not None:
        stmt = stmt.where(Document.namespace == namespace)
        count_stmt = count_stmt.where(Document.namespace == namespace)

    # Total count.
    total_result = await session.execute(count_stmt)
    total: int = total_result.scalar_one()

    # Paginate.
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    result = await session.execute(stmt)
    docs = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": str(d.id),
                "namespace": d.namespace,
                "title": d.title,
                "doc_type": d.doc_type.value,
                "status": d.status.value,
                "total_chunks": d.total_chunks,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ],
    }


# ── GET /api/v1/documents/{doc_id} ─────────────────────────────────────────


@router.get("/{doc_id}")
async def get_document(
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Get detailed information about a single document, including chunk count."""
    stmt = select(Document).where(Document.id == doc_id)
    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Count chunks.
    chunk_count_stmt = select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)
    chunk_result = await session.execute(chunk_count_stmt)
    chunk_count: int = chunk_result.scalar_one()

    return {
        "id": str(doc.id),
        "namespace": doc.namespace,
        "title": doc.title,
        "doc_type": doc.doc_type.value,
        "source_url": doc.source_url,
        "file_hash": doc.file_hash,
        "status": doc.status.value,
        "total_chunks": doc.total_chunks,
        "actual_chunk_count": chunk_count,
        "metadata": doc.metadata_,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


# ── DELETE /api/v1/documents/{doc_id} ──────────────────────────────────────


@router.delete("/{doc_id}", status_code=200)
async def delete_document(
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Delete a document and its associated chunks, MinIO file, and Milvus vectors.

    Cascade order:
    1. Delete vectors from Milvus.
    2. Delete file from MinIO.
    3. Delete chunks + document rows from PostgreSQL (cascade).
    """
    stmt = select(Document).where(Document.id == doc_id)
    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_id_str = str(doc.id)

    # 1. Delete vectors from Milvus.
    try:
        await asyncio.to_thread(_delete_milvus_vectors, doc_id_str)
    except Exception:
        logger.exception("Failed to delete Milvus vectors for doc_id=%s", doc_id_str)

    # 2. Delete file from MinIO.
    object_name = (doc.metadata_ or {}).get("object_name")
    if object_name:
        try:
            await asyncio.to_thread(delete_file, object_name)
        except Exception:
            logger.exception("Failed to delete MinIO file for doc_id=%s", doc_id_str)

    # 3. Delete from PostgreSQL (chunks cascade via FK).
    await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    await session.execute(delete(Document).where(Document.id == doc.id))
    await session.commit()

    logger.info("Deleted document doc_id=%s", doc_id_str)
    return {"detail": f"Document {doc_id_str} deleted"}


def _delete_milvus_vectors(doc_id: str) -> None:
    """Remove all child chunk vectors belonging to a document from Milvus."""
    from pymilvus import Collection

    collection = Collection(CHILD_CHUNKS_COLLECTION)
    collection.delete(expr=f'doc_id == "{doc_id}"')
    collection.flush()
