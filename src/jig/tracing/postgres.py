"""PostgreSQL tracing backend — requires the ``postgres`` extra.

Import from this module directly (``from jig.tracing.postgres import
PostgresTracer``); ``jig.tracing`` does not re-export it, so SQLite-only
consumers never import psycopg.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg

from jig._postgres import LazyPgConnection, jsonb_text, pg_text, quoted_schema
from jig.core.types import Span, SpanKind, TracingLogger, Usage
from jig.tracing.sqlite import _default_serializer

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUFFERED_SPANS = 10_000

# How many distinct trace ids a failed-flush log line names before eliding.
_LOGGED_TRACE_IDS = 5

_SPAN_COLUMNS = (
    "id, trace_id, parent_id, kind, name, input, output, started_at, ended_at, "
    "duration_ms, metadata, error, usage_input_tokens, usage_output_tokens, usage_cost"
)


def tracer_schema_sql(schema: str = "jig") -> str:
    """DDL for :class:`PostgresTracer`'s tables, idempotent, in ``schema``.

    Exposed so a consumer with its own versioned migrator can apply it there
    and construct the tracer with ``create_schema=False``. Mirrors the SQLite
    span schema column-for-column, with ``namespace`` added (per-trial
    isolation without separate databases), real timestamps, and ``jsonb``
    payloads so span metadata stays queryable.
    """
    s = quoted_schema(schema)
    return f"""
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.spans (
    id text PRIMARY KEY,
    namespace text NOT NULL,
    trace_id text NOT NULL,
    parent_id text,
    kind text NOT NULL,
    name text NOT NULL,
    input jsonb,
    output jsonb,
    started_at timestamptz NOT NULL,
    ended_at timestamptz,
    duration_ms double precision,
    metadata jsonb,
    error text,
    usage_input_tokens bigint,
    usage_output_tokens bigint,
    usage_cost double precision
);

CREATE INDEX IF NOT EXISTS spans_namespace_trace_idx ON {s}.spans (namespace, trace_id);
CREATE INDEX IF NOT EXISTS spans_namespace_roots_idx
    ON {s}.spans (namespace, started_at DESC) WHERE parent_id IS NULL;
CREATE INDEX IF NOT EXISTS spans_kind_idx ON {s}.spans (kind);
"""


class PostgresTracer(TracingLogger):
    """Buffered span writer over PostgreSQL.

    Same hot-path shape as :class:`~jig.tracing.sqlite.SQLiteTracer`:
    ``start_span`` / ``end_span`` only touch memory, and :meth:`flush`
    writes. Two differences follow from the store living across a network:

    - **A database failure never fails the run.** A failed flush is logged
      with the affected trace ids, counted in :attr:`failed_flush_count`,
      and every span stays buffered for the next flush; the connection is
      reopened once the server answers again.
    - **The buffer is bounded** at ``max_buffered_spans``. Admitting a span
      past the bound evicts the oldest *finished* span (the oldest open one
      only when every buffered span is still open, so a long-running trace
      root survives an outage), counts it in :attr:`dropped_span_count`, and
      logs the drop with its trace id.

    ``namespace`` scopes every write and read, standing in for the separate
    database file a SQLite consumer would open per trial.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "jig",
        namespace: str = "default",
        create_schema: bool = True,
        max_buffered_spans: int = DEFAULT_MAX_BUFFERED_SPANS,
    ) -> None:
        if isinstance(max_buffered_spans, bool) or max_buffered_spans < 1:
            raise ValueError(
                f"max_buffered_spans must be a positive integer, got {max_buffered_spans!r}"
            )
        s = quoted_schema(schema)
        self._namespace = namespace
        self._max_buffered_spans = max_buffered_spans
        self._spans: dict[str, Span] = {}
        self._dropped_spans = 0
        self._failed_flushes = 0
        self._db_lock: asyncio.Lock | None = None
        self._conn = LazyPgConnection(dsn, tracer_schema_sql(schema) if create_schema else None)
        self._upsert = (
            f"INSERT INTO {s}.spans (namespace, {_SPAN_COLUMNS}) VALUES "
            "(%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "trace_id = EXCLUDED.trace_id, parent_id = EXCLUDED.parent_id, "
            "kind = EXCLUDED.kind, name = EXCLUDED.name, input = EXCLUDED.input, "
            "output = EXCLUDED.output, started_at = EXCLUDED.started_at, "
            "ended_at = EXCLUDED.ended_at, duration_ms = EXCLUDED.duration_ms, "
            "metadata = EXCLUDED.metadata, error = EXCLUDED.error, "
            "usage_input_tokens = EXCLUDED.usage_input_tokens, "
            "usage_output_tokens = EXCLUDED.usage_output_tokens, "
            "usage_cost = EXCLUDED.usage_cost"
        )
        self._select_trace = (
            f"SELECT {_SPAN_COLUMNS} FROM {s}.spans "
            "WHERE namespace = %s AND trace_id = %s ORDER BY started_at ASC"
        )
        self._select_roots = (
            f"SELECT {_SPAN_COLUMNS} FROM {s}.spans "
            "WHERE namespace = %s AND parent_id IS NULL AND kind = ANY(%s)"
        )

    @property
    def dropped_span_count(self) -> int:
        """Spans evicted from a full buffer before they were ever written."""
        return self._dropped_spans

    @property
    def failed_flush_count(self) -> int:
        """Flushes that hit a database error and left their spans buffered."""
        return self._failed_flushes

    def _admit(self, span: Span) -> None:
        self._spans[span.id] = span
        while len(self._spans) > self._max_buffered_spans:
            self._evict_one()

    def _evict_one(self) -> None:
        # Insertion order is start order, so the first match is the oldest.
        victim_id = next(
            (sid for sid, s in self._spans.items() if s.ended_at is not None),
            next(iter(self._spans)),
        )
        victim = self._spans.pop(victim_id)
        self._dropped_spans += 1
        logger.warning(
            "postgres tracer buffer full (max=%d): dropped span %s (trace %s, %s %r, %s); "
            "dropped so far=%d",
            self._max_buffered_spans, victim.id, victim.trace_id, victim.kind.value,
            victim.name, "open" if victim.ended_at is None else "finished",
            self._dropped_spans,
        )

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        span = Span(
            id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            kind=kind,
            name=name,
            started_at=datetime.now(UTC),
            metadata=metadata,
        )
        self._admit(span)
        return span

    def start_span(
        self,
        parent_id: str,
        kind: SpanKind,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        parent = self._spans.get(parent_id)
        trace_id = parent.trace_id if parent else parent_id
        span = Span(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(UTC),
            parent_id=parent_id,
            input=input,
            metadata=metadata,
        )
        self._admit(span)
        return span

    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Usage | None = None) -> None:
        span = self._spans.get(span_id)
        if not span:
            return
        span.ended_at = datetime.now(UTC)
        span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        span.output = output
        span.error = error
        span.usage = usage

    def _row(self, span: Span) -> tuple[Any, ...]:
        return (
            self._namespace,
            span.id,
            span.trace_id,
            span.parent_id,
            span.kind.value,
            pg_text(span.name),
            jsonb_text(span.input, _default_serializer),
            jsonb_text(span.output, _default_serializer),
            span.started_at,
            span.ended_at,
            span.duration_ms,
            jsonb_text(span.metadata, _default_serializer),
            pg_text(span.error) if span.error is not None else None,
            span.usage.input_tokens if span.usage else None,
            span.usage.output_tokens if span.usage else None,
            span.usage.cost if span.usage else None,
        )

    def _get_db_lock(self) -> asyncio.Lock:
        # One connection serves writes and reads; the lock keeps a read from
        # landing inside an in-flight flush transaction.
        if self._db_lock is None:
            self._db_lock = asyncio.Lock()
        return self._db_lock

    async def flush(self) -> None:
        async with self._get_db_lock():
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        # Snapshot rows and open/closed state in one synchronous pass before
        # any await — same reasoning as SQLiteTracer.flush: a consistent view
        # immune to concurrent start_span/end_span, and eviction that depends
        # on state *at snapshot time*.
        pending_ids: set[str] = set()
        open_at_snapshot: set[str] = set()
        rows: list[tuple[Any, ...]] = []
        for sid, span in self._spans.items():
            pending_ids.add(sid)
            if span.ended_at is None:
                open_at_snapshot.add(sid)
            rows.append(self._row(span))
        if not rows:
            return
        logger.debug("postgres tracer flush start (pending=%d)", len(rows))
        try:
            conn = await self._conn.get()
            async with conn.transaction(), conn.cursor() as cur:
                await cur.executemany(self._upsert, rows)
        except (psycopg.Error, OSError) as exc:
            # IO boundary: a database failure must not fail the agent run,
            # and must not be silent. Nothing was evicted, so every span is
            # retried on the next flush (subject to the buffer bound).
            self._failed_flushes += 1
            trace_ids = sorted({row[2] for row in rows})
            shown = ", ".join(trace_ids[:_LOGGED_TRACE_IDS])
            more = len(trace_ids) - _LOGGED_TRACE_IDS
            logger.error(
                "postgres tracer flush failed; %d spans stay buffered (traces: %s%s); "
                "failed flushes so far=%d: %s",
                len(rows), shown, f" +{more} more" if more > 0 else "",
                self._failed_flushes, exc,
            )
            return
        # A span is retained if it was open at snapshot time (its final state
        # still needs a write) or was admitted after the snapshot.
        self._spans = {
            sid: s
            for sid, s in self._spans.items()
            if sid in open_at_snapshot or sid not in pending_ids
        }
        logger.debug(
            "postgres tracer flush done (persisted=%d, retained=%d)",
            len(pending_ids), len(self._spans),
        )

    async def get_trace(self, trace_id: str) -> list[Span]:
        async with self._get_db_lock():
            conn = await self._conn.get()
            cur = await conn.execute(self._select_trace, (self._namespace, trace_id))
            rows = await cur.fetchall()
        return [_row_to_span(row) for row in rows]

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        """List trace roots: top-level AGENT_RUN or PIPELINE_RUN spans.

        Same rule as :meth:`SQLiteTracer.list_traces` — only spans with no
        parent qualify, so a nested ``PIPELINE_RUN`` is excluded. A naive
        ``since`` is read as UTC.
        """
        query = self._select_roots
        params: list[Any] = [
            self._namespace,
            [SpanKind.AGENT_RUN.value, SpanKind.PIPELINE_RUN.value],
        ]
        if since:
            query += " AND started_at >= %s"
            params.append(since if since.tzinfo is not None else since.replace(tzinfo=UTC))
        if name:
            query += " AND name = %s"
            params.append(name)
        query += " ORDER BY started_at DESC LIMIT %s"
        params.append(limit)
        async with self._get_db_lock():
            conn = await self._conn.get()
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
        return [_row_to_span(row) for row in rows]

    async def close(self) -> None:
        try:
            if self._spans:
                await self.flush()
        finally:
            await self._conn.close()


def _row_to_span(row: tuple[Any, ...]) -> Span:
    (
        sid, trace_id, parent_id, kind, name, inp, out,
        started, ended, duration, meta, error,
        u_in, u_out, u_cost,
    ) = row
    usage = None
    if u_in is not None:
        usage = Usage(input_tokens=u_in, output_tokens=u_out or 0, cost=u_cost)
    return Span(
        id=sid,
        trace_id=trace_id,
        kind=SpanKind(kind),
        name=name,
        started_at=started.astimezone(UTC),
        parent_id=parent_id,
        input=inp,
        output=out,
        ended_at=ended.astimezone(UTC) if ended else None,
        duration_ms=duration,
        metadata=meta,
        error=error,
        usage=usage,
    )
