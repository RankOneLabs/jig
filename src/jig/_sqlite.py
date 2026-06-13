"""Internal SQLite helpers: lazy connection management and JSON thin wrappers.

This module is private (underscore-prefixed) and not exported from jig.__init__.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aiosqlite


class LazyConnection:
    """Event-loop-safe lazy aiosqlite connection with schema initialization.

    The lock is created on first get() call (not in __init__) so it binds
    to the running event loop rather than whatever loop (if any) existed
    at construction time.
    """

    def __init__(self, db_path: str, schema: str) -> None:
        self._db_path = db_path
        self._schema = schema
        self._db: aiosqlite.Connection | None = None
        # Constructed lazily inside get() so the lock binds to the
        # event loop that's actually running, not whatever loop existed
        # (or didn't) at __init__ time.
        self._lock: asyncio.Lock | None = None

    async def get(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Re-check after acquiring — another waiter may have
            # initialized the connection while we blocked on the lock.
            if self._db is None:
                conn = await aiosqlite.connect(self._db_path)
                try:
                    await conn.executescript(self._schema)
                except:
                    await conn.close()
                    raise
                self._db = conn
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        # Acquire the lock so close() cannot race with an in-flight get():
        # without this, close() could see _db=None, no-op, and the connect
        # completing in get() would leave the connection open permanently.
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None
        # Reset the lock so re-use after close() works across event loops.
        self._lock = None


def json_dumps(obj: Any) -> str:
    return json.dumps(obj)


def json_loads(s: str) -> Any:
    return json.loads(s)
