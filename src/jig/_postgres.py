"""Internal PostgreSQL helpers for the optional ``postgres`` extra.

Private (underscore-prefixed) and not exported from ``jig.__init__``: importing
it requires psycopg, which SQLite-only consumers never install.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable
from typing import Any

import psycopg

# Schema names are interpolated into DDL and queries (identifiers cannot be
# bound parameters), so they are restricted to plain lowercase identifiers
# and always double-quoted on use.
_SCHEMA_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")

# A JSON ``\u0000`` escape: a backslash preceded by an even number of
# backslashes (zero included), so an escaped literal ``\\u0000`` in the text
# is left alone. jsonb rejects the NUL code point outright.
_JSON_NUL_ESCAPE = re.compile(r"(?<!\\)((?:\\\\)*)\\u0000")


def quoted_schema(schema: str) -> str:
    """Validate ``schema`` and return it double-quoted for SQL interpolation."""
    if not _SCHEMA_NAME.fullmatch(schema):
        raise ValueError(
            f"PostgreSQL schema name must match {_SCHEMA_NAME.pattern!r}, got {schema!r}"
        )
    return f'"{schema}"'


def pg_text(value: str) -> str:
    """PostgreSQL ``text`` cannot hold NUL; substitute U+FFFD for it."""
    return value.replace("\x00", "�")


def _finite(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` throughout parsed JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def jsonb_text(obj: Any, default: Callable[[Any], Any]) -> str | None:
    """Serialize ``obj`` to text a ``::jsonb`` cast accepts, never raising.

    Same fallback contract as the SQLite tracer's ``_safe_json`` (``default``
    for unknown types, ``repr`` when serialization fails outright), plus the
    two things jsonb rejects that SQLite's JSON column stored verbatim:
    ``NaN``/``Infinity`` become ``null`` (JSON has no spelling for them) and
    the NUL code point becomes U+FFFD.
    """
    if obj is None:
        return None
    try:
        text = json.dumps(obj, default=default, allow_nan=False)
    except ValueError:
        try:
            # json.loads parses NaN/Infinity back into floats, so one
            # lenient round-trip yields a structure _finite can clean.
            reparsed = json.loads(json.dumps(obj, default=default))
            text = json.dumps(_finite(reparsed), allow_nan=False)
        except (TypeError, ValueError):
            text = json.dumps(repr(obj))
    except TypeError:
        text = json.dumps(repr(obj))
    return _JSON_NUL_ESCAPE.sub(r"\1\\ufffd", text)


class LazyPgConnection:
    """Event-loop-safe lazy psycopg connection with optional schema setup.

    Mirrors :class:`jig._sqlite.LazyConnection`: one stable lock created on
    first use, every state transition under it. Unlike a SQLite file, a
    PostgreSQL connection can die underneath us — a closed or broken
    connection is discarded and reopened on the next :meth:`get`, so a
    server outage heals once the server is back.

    The connection runs in autocommit mode with the session time zone pinned
    to UTC; multi-statement writes open an explicit ``transaction()``.
    """

    def __init__(self, dsn: str, ddl: str | None) -> None:
        self._dsn = dsn
        self._ddl = ddl
        self._conn: psycopg.AsyncConnection[Any] | None = None
        self._lock: asyncio.Lock | None = None

    async def get(self) -> psycopg.AsyncConnection[Any]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._conn is not None and (self._conn.closed or self._conn.broken):
                await self._conn.close()
                self._conn = None
            if self._conn is None:
                conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
                try:
                    await conn.execute("SET TIME ZONE 'UTC'")
                    if self._ddl is not None:
                        # Concurrent CREATE ... IF NOT EXISTS from two
                        # processes can still collide on the catalog's unique
                        # index; the advisory lock serializes first-time setup.
                        async with conn.transaction():
                            await conn.execute(
                                "SELECT pg_advisory_xact_lock(hashtext(%s))", (self._ddl,),
                            )
                            # No bound parameters, so psycopg sends this as one
                            # simple-protocol query and multiple statements run.
                            await conn.execute(self._ddl)  # type: ignore[arg-type]
                except BaseException:
                    await conn.close()
                    raise
                self._conn = conn
            return self._conn

    async def close(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
