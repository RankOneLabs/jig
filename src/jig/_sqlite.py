"""Internal SQLite helpers: lazy connection management and JSON thin wrappers.

This module is private (underscore-prefixed) and not exported from jig.__init__.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import aiosqlite


class LazyConnection:
    """Event-loop-safe lazy aiosqlite connection with schema initialization.

    A single lock is created on the first get() or close() call and kept
    for the lifetime of the instance. All state transitions (_db, _lock)
    must go through this one stable lock so that queued callers always
    observe a coherent state — close() resetting the lock would leave
    callers already waiting on the old lock in a broken state machine.
    """

    def __init__(self, db_path: str, schema: str) -> None:
        self._db_path = db_path
        self._schema = schema
        self._db: aiosqlite.Connection | None = None
        # Created lazily on first use so it binds to the running event loop,
        # not whatever loop (if any) existed at __init__ time. Once set, it
        # is never replaced — this is the stable-lock guarantee.
        self._lock: asyncio.Lock | None = None

    async def get(self) -> aiosqlite.Connection:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._db is None:
                conn = await aiosqlite.connect(self._db_path)
                try:
                    await conn.execute("PRAGMA foreign_keys=ON")
                    await conn.executescript(self._schema)
                except BaseException:
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
        # Do NOT reset self._lock here. Callers already queued on the lock
        # hold a reference to it; replacing it would strand them on a lock
        # that is never released, and future get() calls would create a
        # second lock racing with the first. The lock is stable for life.


def json_dumps(obj: Any) -> str:
    return json.dumps(obj)


def json_loads(s: str) -> Any:
    return json.loads(s)


def parse_aware_utc(s: str) -> datetime:
    """Parse an ISO timestamp, treating a legacy naive string as UTC.

    New rows are always written with ``datetime.now(UTC).isoformat()``
    (a ``+00:00`` offset). Rows written before this existed have no
    offset — their original timezone can't be reconstructed, so they're
    interpreted as UTC on read without rewriting the stored value.
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
