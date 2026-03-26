"""Standalone ingestion worker process.

Run with::

    python -m app.ingestion.worker

The worker:

1. Initialises PostgreSQL, Redis, Milvus, and MinIO connections.
2. Pre-loads the BGE-M3 embedding model.
3. Listens on the ``toolref:ingestion`` Redis Stream for new jobs.
4. For each job, runs :class:`IngestPipeline.process`.
5. Acknowledges the message on success; retries up to 3 times on failure.
"""

import asyncio
import logging
import signal

from app.config import settings

# ── Logging (mirrors main.py format) ─────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingestion-worker")

# Maximum number of retries per message before giving up.
MAX_RETRIES: int = 3

# Graceful shutdown flag.
_shutdown_event: asyncio.Event | None = None


async def _init_connections() -> None:
    """Establish connections to all external services."""
    from app.cache.redis import connect_redis
    from app.db.engine import engine
    from app.storage.minio import ensure_bucket
    from app.vectorstore.milvus import connect_milvus

    # PostgreSQL — verify connectivity.
    async with engine.begin() as conn:
        await conn.run_sync(lambda _: None)
    logger.info("PostgreSQL connected")

    # Redis
    await connect_redis()

    # Milvus (sync API)
    await asyncio.to_thread(connect_milvus)

    # MinIO — ensure bucket exists.
    await asyncio.to_thread(ensure_bucket)


async def _close_connections() -> None:
    """Gracefully close all external connections."""
    from app.cache.redis import close_redis
    from app.db.engine import engine
    from app.vectorstore.milvus import disconnect_milvus

    disconnect_milvus()
    await close_redis()
    await engine.dispose()
    logger.info("All connections closed")


async def _process_with_retry(
    pipeline: "IngestPipeline",
    doc_id: str,
    namespace: str,
    object_name: str,
    doc_type: str,
) -> bool:
    """Run the pipeline with up to :data:`MAX_RETRIES` attempts.

    Returns ``True`` if processing eventually succeeded.
    """
    from app.ingestion.pipeline import IngestPipeline

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await pipeline.process(doc_id, namespace, object_name, doc_type)
            return True
        except Exception:
            logger.exception(
                "Attempt %d/%d failed for doc_id=%s",
                attempt,
                MAX_RETRIES,
                doc_id,
            )
            if attempt < MAX_RETRIES:
                # Exponential back-off: 2s, 4s
                await asyncio.sleep(2 ** attempt)
    return False


async def run() -> None:
    """Main worker loop: initialise → consume → process → ack."""
    global _shutdown_event

    from app.ingestion.embedder import embedding_service
    from app.ingestion.pipeline import IngestPipeline
    from app.ingestion.queue import ack, consume

    _shutdown_event = asyncio.Event()

    # ── Setup signal handlers for graceful shutdown ──────────────────────
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _shutdown_event.set())  # type: ignore[arg-type]

    # ── Init ─────────────────────────────────────────────────────────────
    logger.info("Initialising connections …")
    await _init_connections()

    logger.info("Pre-loading embedding model …")
    await asyncio.to_thread(embedding_service.warmup)

    pipeline = IngestPipeline()

    logger.info("Worker ready — listening for ingestion jobs")

    # ── Consume loop ─────────────────────────────────────────────────────
    try:
        async for msg_id, payload in consume():
            if _shutdown_event.is_set():
                logger.info("Shutdown signal received — stopping consumer")
                break

            doc_id = payload.get("doc_id", "")
            namespace = payload.get("namespace", "")
            object_name = payload.get("object_name", "")
            doc_type = payload.get("doc_type", "")

            logger.info(
                "Received job msg_id=%s doc_id=%s type=%s",
                msg_id,
                doc_id,
                doc_type,
            )

            success = await _process_with_retry(
                pipeline, doc_id, namespace, object_name, doc_type
            )

            if success:
                await ack(msg_id)
                logger.info("Job completed and acknowledged: doc_id=%s", doc_id)
            else:
                logger.error(
                    "Job permanently failed after %d attempts: doc_id=%s",
                    MAX_RETRIES,
                    doc_id,
                )
                # Acknowledge anyway to avoid infinite retry loop in the stream.
                await ack(msg_id)

    except asyncio.CancelledError:
        logger.info("Worker task cancelled")
    finally:
        await _close_connections()
        logger.info("Ingestion worker shut down")


def main() -> None:
    """Entry point when running ``python -m app.ingestion.worker``."""
    logger.info("Starting ToolRef ingestion worker")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")


if __name__ == "__main__":
    main()
