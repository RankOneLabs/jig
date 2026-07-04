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
        conn.execute = AsyncMock()
        conn.executescript = AsyncMock()
        created.append(conn)
        return conn

    return fake_connect, created


@pytest.mark.asyncio
async def test_sqlite_tracer_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    tracer = SQLiteTracer(db_path=str(tmp_path / "t.db"))
    with patch("jig._sqlite.aiosqlite.connect", side_effect=fake_connect):
        # 8 concurrent first-callers — pre-fix all 8 would race past
        # ``self._db is None`` before any connect resolved.
        results = await asyncio.gather(*[tracer._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)


@pytest.mark.asyncio
async def test_sqlite_store_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    store = SqliteStore(db_path=str(tmp_path / "m.db"))
    with patch("jig._sqlite.aiosqlite.connect", side_effect=fake_connect):
        results = await asyncio.gather(*[store._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)


@pytest.mark.asyncio
async def test_feedback_loop_get_db_is_single_flight(tmp_path: Any) -> None:
    fake_connect, created = _make_fake_connect_factory()
    feedback = SQLiteFeedbackLoop(db_path=str(tmp_path / "f.db"))
    with patch("jig._sqlite.aiosqlite.connect", side_effect=fake_connect):
        results = await asyncio.gather(*[feedback._get_db() for _ in range(8)])

    assert len(created) == 1, f"expected 1 connect call, got {len(created)}"
    assert all(r is created[0] for r in results)


# ---------------------------------------------------------------------------
# Stable-lock regression: close()/get() races must not double-connect
# ---------------------------------------------------------------------------


from jig._sqlite import LazyConnection  # noqa: E402 — kept below class imports for grouping


@pytest.mark.asyncio
async def test_lock_not_reset_after_close(tmp_path: Any) -> None:
    """close() must not reset self._lock to None.

    Pre-fix: close() set ``self._lock = None`` after releasing it, so a
    get() caller that arrived after the reset would create a second lock
    rather than queuing on the original one.  Callers already waiting on
    the original lock would be on a different lock from the new arrival —
    two concurrent state machines on the same instance.

    Post-fix: the lock is created once and kept for the instance lifetime.
    This is the core invariant the stable-lock fix establishes.
    """
    lc = LazyConnection(str(tmp_path / "lock.db"), "")
    await lc.get()  # creates the lock
    lock_before_close = lc._lock
    assert lock_before_close is not None

    await lc.close()

    # Pre-fix: _lock would be None here.  Post-fix: still the same object.
    assert lc._lock is lock_before_close, (
        "close() must not replace self._lock; "
        "callers already queued on the old lock would be stranded"
    )
    await lc.close()  # cleanup (second close is idempotent)


@pytest.mark.asyncio
async def test_close_is_idempotent_and_instance_reusable(tmp_path: Any) -> None:
    """close() is idempotent; get() reconnects after close().

    Two contracts the stable-lock fix must preserve: multiple close() calls
    must not raise, and a subsequent get() must produce a new live connection.
    """
    lc = LazyConnection(str(tmp_path / "reuse.db"), "")
    c1 = await lc.get()

    await lc.close()
    await lc.close()  # second call must not raise

    c2 = await lc.get()  # must reconnect
    assert c2 is not c1, "post-close get() must return a new connection object"

    await lc.close()


@pytest.mark.asyncio
async def test_concurrent_close_and_get_no_error(tmp_path: Any) -> None:
    """Concurrent close() and get() must not raise, deadlock, or produce
    inconsistent connection state.

    close() holds the lock during ``await conn.close()``.  A get() that
    arrives while the lock is held will queue.  After close() releases the
    lock and sets _db=None, the queued get() reconnects cleanly.
    Pre-fix: the lock was reset to None AFTER releasing it, which could
    strand waiters already holding a reference to the old lock object.
    """
    lc = LazyConnection(str(tmp_path / "conc.db"), "")
    await lc.get()  # establish connection

    # Race close() and get() — ordering is scheduler-dependent; what matters
    # is no exception and consistent state.
    results = await asyncio.gather(lc.close(), lc.get(), lc.get())
    get_results = [r for r in results if r is not None]
    if get_results:
        assert all(c is get_results[0] for c in get_results), (
            "concurrent get() calls must return the same connection"
        )

    await lc.close()  # cleanup (idempotent if already closed)
