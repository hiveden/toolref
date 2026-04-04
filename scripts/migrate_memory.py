"""DB migration: create conversation memory tables.

Creates the following tables if they do not already exist:
    - conversation_messages
    - memory_episodes

Run with::

    docker compose exec backend python scripts/migrate_memory.py

This script is idempotent (safe to re-run; uses CREATE TABLE IF NOT EXISTS
via SQLAlchemy ``checkfirst=True`` in ``create_all``).
"""

import asyncio
import sys

# Make sure the app package is importable when running from the repo root
sys.path.insert(0, "/app")

from app.db.engine import engine  # noqa: E402
from app.db.models import Base  # noqa: E402  (imports all models, including new ones)


async def main() -> None:
    """Run the migration."""
    print("Running memory migration …")
    async with engine.begin() as conn:
        # checkfirst=True → CREATE TABLE IF NOT EXISTS semantics
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    print("Migration complete ✓")
    print("  Tables managed: conversation_messages, memory_episodes")


if __name__ == "__main__":
    asyncio.run(main())
