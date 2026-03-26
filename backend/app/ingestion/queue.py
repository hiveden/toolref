"""Redis Streams message queue for asynchronous document ingestion.

Producers (API layer) call :func:`publish` to enqueue a document for
processing.  The ingestion worker calls :func:`consume` to receive
jobs via ``XREADGROUP`` with blocking, then acknowledges each message
after successful processing.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis

from app.cache.redis import get_redis

logger = logging.getLogger(__name__)

STREAM_NAME: str = "toolref:ingestion"
CONSUMER_GROUP: str = "ingestion-workers"
CONSUMER_NAME: str = "worker-1"


async def _ensure_consumer_group(r: aioredis.Redis) -> None:
    """Create the consumer group if it does not exist.

    Uses ``MKSTREAM`` so the stream is auto-created on first call.
    """
    try:
        await r.xgroup_create(
            name=STREAM_NAME,
            groupname=CONSUMER_GROUP,
            id="0",
            mkstream=True,
        )
        logger.info(
            "Created consumer group '%s' on stream '%s'",
            CONSUMER_GROUP,
            STREAM_NAME,
        )
    except aioredis.ResponseError as exc:
        # Group already exists — safe to ignore.
        if "BUSYGROUP" in str(exc):
            logger.debug("Consumer group '%s' already exists", CONSUMER_GROUP)
        else:
            raise


async def publish(
    doc_id: str,
    namespace: str,
    object_name: str,
    doc_type: str,
) -> str:
    """Add an ingestion job to the Redis Stream.

    Args:
        doc_id: UUID of the document row in PostgreSQL.
        namespace: Document namespace.
        object_name: MinIO object key for the uploaded file.
        doc_type: Document type (pdf, markdown, html, txt).

    Returns:
        The stream message ID assigned by Redis.
    """
    r = await get_redis()
    await _ensure_consumer_group(r)

    message: dict[str, str] = {
        "doc_id": doc_id,
        "namespace": namespace,
        "object_name": object_name,
        "doc_type": doc_type,
    }
    msg_id: str = await r.xadd(STREAM_NAME, message)  # type: ignore[assignment]
    logger.info("Published ingestion job for doc_id=%s (msg_id=%s)", doc_id, msg_id)
    return msg_id


async def consume(
    consumer_name: str = CONSUMER_NAME,
    block_ms: int = 5000,
) -> AsyncIterator[tuple[str, dict[str, str]]]:
    """Async generator that yields ``(message_id, payload)`` tuples.

    Uses ``XREADGROUP`` with ``BLOCK`` to long-poll for new messages,
    avoiding busy-wait loops.

    Args:
        consumer_name: Unique name for this consumer instance.
        block_ms: Maximum milliseconds to block waiting for new data.

    Yields:
        ``(message_id, payload)`` where *payload* is a dict with keys
        ``doc_id``, ``namespace``, ``object_name``, ``doc_type``.
    """
    r = await get_redis()
    await _ensure_consumer_group(r)

    logger.info(
        "Consumer '%s' listening on stream '%s' (group '%s')",
        consumer_name,
        STREAM_NAME,
        CONSUMER_GROUP,
    )

    while True:
        # XREADGROUP returns list of (stream_name, [(msg_id, fields), ...])
        results: list[Any] = await r.xreadgroup(  # type: ignore[assignment]
            groupname=CONSUMER_GROUP,
            consumername=consumer_name,
            streams={STREAM_NAME: ">"},
            count=1,
            block=block_ms,
        )

        if not results:
            # Timeout — no new messages; loop and block again.
            continue

        for _stream_name, messages in results:
            for msg_id, fields in messages:
                yield msg_id, fields


async def ack(message_id: str) -> None:
    """Acknowledge a successfully processed message.

    Args:
        message_id: The Redis Stream message ID to acknowledge.
    """
    r = await get_redis()
    await r.xack(STREAM_NAME, CONSUMER_GROUP, message_id)
    logger.debug("Acknowledged message %s", message_id)
