"""TOCTOU regression: lazy ``_get_db()`` initialization is single-flight.

Three classes share a lazy-init pattern for their aiosqlite connection:
``SQLiteTracer``, ``SqliteStore`` (memory.local), ``SQLiteFeedbackLoop``
(feedback.loop). Without locking, two concurrent first-callers would each
see ``self._db is None``, both call ``aiosqlite.connect(...)``, and one
connection ends up orphaned — a silent fd leak whose schema script would
also run twice.

These tests force N concurrent first-callers and assert (a) ``connect``
was called exactly once, (b) all callers got the same connection object.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jig.feedback.loop import SQLiteFeedbackLoop
from jig.memory.local import SqliteStore
from jig.tracing.sqlite import SQLiteTracer


def _make_fake_connect_factory() -> tuple[Any, list[Any]]:
    """Return a fake ``aiosqlite.connect`` that yields a fresh mock conn
    per call, plus the list of conns created (so callers can count).
    """
    created: list[Any] = []

    async def fake_connect(*_args: Any, **_kwargs: Any) -> Any:
        # Mimic real connect: a real one would wait on I/O, so yield to
        # simulate that + give other waiters a chance to enter the
        # critical section if locking is broken.
        await asyncio.sleep(0)
        conn = MagicMock(name=f"conn-{len(created)}")
        conn.executescript = AsyncMock()
        created.append(conn)
        return conn

    return fake_connect, created


@pytest.mark.asyncio
async def test_sqlite_tracer_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    tracer = SQLiteTracer(db_path=str(tmp_path / "t.db"))
    with patch("jig.tracing.sqlite.aiosqlite.connect", side_effect=fake_connect):
        # 8 concurrent first-callers — pre-fix all 8 would race past
        # ``self._db is None`` before any connect resolved.
        results = await asyncio.gather(*[tracer._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)


@pytest.mark.asyncio
async def test_sqlite_store_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    store = SqliteStore(db_path=str(tmp_path / "m.db"))
    with patch("jig.memory.local.aiosqlite.connect", side_effect=fake_connect):
        results = await asyncio.gather(*[store._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)


@pytest.mark.asyncio
async def test_feedback_loop_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    feedback = SQLiteFeedbackLoop(db_path=str(tmp_path / "f.db"))
    with patch("jig.feedback.loop.aiosqlite.connect", side_effect=fake_connect):
        results = await asyncio.gather(*[feedback._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)
