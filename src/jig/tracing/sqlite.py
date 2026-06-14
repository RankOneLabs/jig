from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from jig._sqlite import LazyConnection, json_loads
from jig.core.types import Span, SpanKind, TracingLogger, Usage

logger = logging.getLogger(__name__)


def _default_serializer(obj: Any) -> Any:
    """Fallback serializer for json.dumps — handles datetime, dataclasses, etc."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return repr(obj)


def _safe_json(obj: Any) -> str | None:
    """Serialize to JSON, falling back gracefully for non-serializable objects."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=_default_serializer)
    except (TypeError, ValueError):
        return json.dumps(repr(obj))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    input JSON,
    output JSON,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms REAL,
    metadata JSON,
    error TEXT,
    usage_input_tokens INTEGER,
    usage_output_tokens INTEGER,
    usage_cost REAL
);

CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
"""


class SQLiteTracer(TracingLogger):
    def __init__(self, db_path: str = "jig_traces.db"):
        self._spans: dict[str, Span] = {}
        self._conn = LazyConnection(db_path, _SCHEMA)

    async def _get_db(self) -> aiosqlite.Connection:
        return await self._conn.get()

    def _insert_span_sync(self, span: Span) -> None:
        self._spans[span.id] = span

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        span = Span(
            id=span_id,
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            metadata=metadata,
        )
        self._spans[span_id] = span
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
        span_id = str(uuid.uuid4())
        span = Span(
            id=span_id,
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            parent_id=parent_id,
            input=input,
            metadata=metadata,
        )
        self._spans[span_id] = span
        return span

    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Usage | None = None) -> None:
        span = self._spans.get(span_id)
        if not span:
            return
        span.ended_at = datetime.now()
        span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        span.output = output
        span.error = error
        span.usage = usage

    async def flush(self) -> None:
        db = await self._get_db()
        # Snapshot before iterating: start_span / start_trace mutate
        # self._spans synchronously between our awaits below. Each
        # individual write is atomic on the event loop, so a list
        # snapshot at this moment captures a consistent view — spans
        # added during the flush body are picked up by the next flush.
        # Without the snapshot, iterating .values() while concurrent
        # tasks insert raises RuntimeError: dictionary changed size.
        pending = list(self._spans.values())
        logger.debug("sqlite tracer flush start (pending=%d)", len(pending))
        for span in pending:
            await db.execute(
                """INSERT OR REPLACE INTO spans
                   (id, trace_id, parent_id, kind, name, input, output,
                    started_at, ended_at, duration_ms, metadata, error,
                    usage_input_tokens, usage_output_tokens, usage_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.id,
                    span.trace_id,
                    span.parent_id,
                    span.kind.value,
                    span.name,
                    _safe_json(span.input),
                    _safe_json(span.output),
                    span.started_at.isoformat(),
                    span.ended_at.isoformat() if span.ended_at else None,
                    span.duration_ms,
                    _safe_json(span.metadata),
                    span.error,
                    span.usage.input_tokens if span.usage else None,
                    span.usage.output_tokens if span.usage else None,
                    span.usage.cost if span.usage else None,
                ),
            )
        await db.commit()
        # Drop only the spans we just persisted. Two cases survive:
        #  - Open spans (``ended_at is None``): a subsequent end_span
        #    needs the entry to update ended_at/duration. This is the
        #    mid-run-flush case (e.g. before trajectory grading).
        #  - Concurrent spans added *after* the snapshot — including
        #    spans the writer task ended synchronously while we were
        #    awaiting db.execute. They never reached the DB this round
        #    and must stay in _spans so the next flush picks them up.
        # The dict comprehension is sync (no await inside) so it can't
        # interleave with concurrent inserts.
        flushed_ids = {span.id for span in pending}
        self._spans = {
            sid: s
            for sid, s in self._spans.items()
            if s.ended_at is None or sid not in flushed_ids
        }
        logger.debug(
            "sqlite tracer flush done (persisted=%d, retained=%d)",
            len(flushed_ids), len(self._spans),
        )

    async def get_trace(self, trace_id: str) -> list[Span]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC",
            (trace_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_span(row) for row in rows]

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        db = await self._get_db()
        query = "SELECT * FROM spans WHERE kind = ?"
        params: list[Any] = [SpanKind.AGENT_RUN.value]
        if since:
            query += " AND started_at >= ?"
            params.append(since.isoformat())
        if name:
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_span(row) for row in rows]

    def _row_to_span(self, row: Any) -> Span:
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
            started_at=datetime.fromisoformat(started),
            parent_id=parent_id,
            input=json_loads(inp) if isinstance(inp, str) else inp,
            output=json_loads(out) if isinstance(out, str) else out,
            ended_at=datetime.fromisoformat(ended) if ended else None,
            duration_ms=duration,
            metadata=json_loads(meta) if isinstance(meta, str) else meta,
            error=error,
            usage=usage,
        )

    async def close(self) -> None:
        try:
            # Skip the flush (and its lazy _get_db() open) when nothing is
            # buffered — otherwise close() on an unused tracer would create
            # an empty DB file and run schema setup for no reason.
            if self._spans:
                await self.flush()
        finally:
            await self._conn.close()
