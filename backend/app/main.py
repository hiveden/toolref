"""FastAPI application entry point.

Sets up CORS, lifespan events (DB/Redis/Milvus connections), and routes.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.cache.redis import close_redis, connect_redis
from app.config import settings
from app.db.engine import engine
from app.storage.minio import ensure_bucket
from app.vectorstore.milvus import connect_milvus, disconnect_milvus

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of external connections."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # PostgreSQL — engine is lazily created; we just verify connectivity
    async with engine.begin() as conn:
        await conn.run_sync(lambda _: None)
    logger.info("PostgreSQL connected")

    # Redis
    await connect_redis()

    # Milvus (sync pymilvus API, run in default thread)
    connect_milvus()

    # MinIO — ensure the document bucket exists
    ensure_bucket()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.app_name)
    disconnect_milvus()
    await close_redis()
    await engine.dispose()
    logger.info("All connections closed")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(documents_router)

    return app


app = create_app()
