from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from jig._sqlite import LazyConnection, json_loads, parse_aware_utc
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
            started_at=datetime.now(UTC),
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
            started_at=datetime.now(UTC),
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
        span.ended_at = datetime.now(UTC)
        span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        span.output = output
        span.error = error
        span.usage = usage

    async def flush(self) -> None:
        db = await self._get_db()
        # Capture a stable snapshot of every span's fields and open/closed
        # state in one synchronous pass before any await. This gives us:
        #   1. A consistent view immune to dict-mutation-during-iteration
        #      errors (RuntimeError: dictionary changed size).
        #   2. Stable field values for DB writes — span objects are mutable
        #      and end_span() can update them between our awaits below.
        #   3. A record of which spans were open *at snapshot time*, so
        #      eviction does not depend on the span's current state after
        #      the DB writes complete.
        pending_ids: set[str] = set()
        open_at_snapshot: set[str] = set()
        row_data: list[tuple[Any, ...]] = []
        for sid, span in self._spans.items():
            pending_ids.add(sid)
            if span.ended_at is None:
                open_at_snapshot.add(sid)
            row_data.append((
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
            ))
        logger.debug("sqlite tracer flush start (pending=%d)", len(row_data))
        for row in row_data:
            await db.execute(
                """INSERT OR REPLACE INTO spans
                   (id, trace_id, parent_id, kind, name, input, output,
                    started_at, ended_at, duration_ms, metadata, error,
                    usage_input_tokens, usage_output_tokens, usage_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
        await db.commit()
        # Eviction rules — a span is retained if either condition holds:
        #  1. It was open at snapshot time: end_span() may have fired during
        #     our DB awaits, giving it final ended_at/duration/output/error.
        #     The row we just wrote still shows it as open. One more flush
        #     cycle is needed to persist the final state.
        #  2. It was added after the snapshot (not in pending_ids): it was
        #     never written this round and must survive for the next flush.
        # The dict comprehension has no await so it cannot interleave with
        # concurrent inserts.
        self._spans = {
            sid: s
            for sid, s in self._spans.items()
            if sid in open_at_snapshot or sid not in pending_ids
        }
        logger.debug(
            "sqlite tracer flush done (persisted=%d, retained=%d)",
            len(pending_ids), len(self._spans),
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
        """List trace roots: top-level AGENT_RUN or PIPELINE_RUN spans.

        Only spans with ``parent_id IS NULL`` qualify — a nested
        ``PIPELINE_RUN`` (e.g. each item run by ``map_pipeline``) has a
        parent span and is not itself a trace root, so it's excluded.
        """
        db = await self._get_db()
        query = "SELECT * FROM spans WHERE parent_id IS NULL AND kind IN (?, ?)"
        params: list[Any] = [SpanKind.AGENT_RUN.value, SpanKind.PIPELINE_RUN.value]
        if since:
            # Normalize to UTC before comparing against stored UTC strings —
            # a non-UTC-offset aware datetime would otherwise compare
            # incorrectly against lexicographically-sorted TEXT timestamps.
            # A naive input is treated as UTC, matching parse_aware_utc's
            # read-side interpretation of legacy naive rows.
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)
            query += " AND started_at >= ?"
            params.append(since_utc.astimezone(UTC).isoformat())
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
            started_at=parse_aware_utc(started),
            parent_id=parent_id,
            input=json_loads(inp) if isinstance(inp, str) else inp,
            output=json_loads(out) if isinstance(out, str) else out,
            ended_at=parse_aware_utc(ended) if ended else None,
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
