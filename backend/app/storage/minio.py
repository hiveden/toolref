"""MinIO object storage client for document files.

Provides helpers to upload, download, and delete document files
stored in a MinIO bucket.  All public functions are thin wrappers
around the ``minio`` Python SDK and are safe to call from async code
via ``asyncio.to_thread``.
"""

import io
import logging

from minio import Minio
from minio.error import S3Error

from app.config import settings

logger = logging.getLogger(__name__)

_client: Minio | None = None


def _get_client() -> Minio:
    """Return the module-level MinIO client, creating it on first call."""
    global _client
    if _client is None:
        _client = Minio(
            endpoint=f"{settings.minio_host}:{settings.minio_port}",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        logger.info(
            "MinIO client created for %s:%s",
            settings.minio_host,
            settings.minio_port,
        )
    return _client


def ensure_bucket() -> None:
    """Create the configured bucket if it does not already exist.

    Should be called once at application / worker startup.
    """
    client = _get_client()
    bucket = settings.minio_bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created MinIO bucket '%s'", bucket)
    else:
        logger.info("MinIO bucket '%s' already exists", bucket)


def upload_file(
    file_data: bytes, object_name: str, content_type: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to MinIO and return the object name.

    Args:
        file_data: Raw file bytes.
        object_name: Target object key in the bucket.
        content_type: MIME type of the file.

    Returns:
        The ``object_name`` that was stored.
    """
    client = _get_client()
    data_stream = io.BytesIO(file_data)
    client.put_object(
        bucket_name=settings.minio_bucket,
        object_name=object_name,
        data=data_stream,
        length=len(file_data),
        content_type=content_type,
    )
    logger.info("Uploaded '%s' (%d bytes) to MinIO", object_name, len(file_data))
    return object_name


def download_file(object_name: str) -> bytes:
    """Download an object from MinIO and return its content as bytes.

    Args:
        object_name: Key of the object to download.

    Returns:
        Raw file bytes.
    """
    client = _get_client()
    response = None
    try:
        response = client.get_object(settings.minio_bucket, object_name)
        data = response.read()
        logger.info("Downloaded '%s' (%d bytes) from MinIO", object_name, len(data))
        return data
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def delete_file(object_name: str) -> None:
    """Delete an object from MinIO.

    Args:
        object_name: Key of the object to remove.
    """
    client = _get_client()
    try:
        client.remove_object(settings.minio_bucket, object_name)
        logger.info("Deleted '%s' from MinIO", object_name)
    except S3Error as exc:
        logger.warning("Failed to delete '%s' from MinIO: %s", object_name, exc)
