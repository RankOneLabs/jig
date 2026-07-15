"""Tests for SQLiteTracer — flush, serialization, round-trip."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import aiosqlite
import pytest

from jig import PipelineConfig, Step, run_pipeline
from jig.core.types import SpanKind
from jig.tracing import SQLiteTracer


@pytest.fixture
def tracer(tmp_path: Any) -> SQLiteTracer:
    return SQLiteTracer(db_path=str(tmp_path / "test_traces.db"))


@pytest.mark.asyncio
async def test_flush_writes_spans(tracer: SQLiteTracer) -> None:
    """Basic flush: start a trace, end it, flush, read it back."""
    span = tracer.start_trace("test-pipeline", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(span.id, output={"result": 42})

    await tracer.flush()

    traces = await tracer.list_traces()
    assert len(traces) == 1  # list_traces includes root PIPELINE_RUN spans
    assert traces[0].name == "test-pipeline"

    spans = await tracer.get_trace(span.trace_id)
    assert len(spans) == 1
    assert spans[0].name == "test-pipeline"
    assert spans[0].output == {"result": 42}


@pytest.mark.asyncio
async def test_flush_with_dataclass_output(tracer: SQLiteTracer) -> None:
    """Flush handles dataclass outputs that aren't JSON-serializable."""

    @dataclass(frozen=True, slots=True)
    class EvalResult:
        score: float
        reason: str

    span = tracer.start_trace("eval-pipeline", kind=SpanKind.PIPELINE_RUN)
    step = tracer.start_span(span.id, SpanKind.PIPELINE_STEP, "evaluate")
    tracer.end_span(step.id, output=EvalResult(score=0.85, reason="relevant"))
    tracer.end_span(span.id)

    await tracer.flush()

    spans = await tracer.get_trace(span.trace_id)
    step_span = next(s for s in spans if s.name == "evaluate")
    assert step_span.output == {"score": 0.85, "reason": "relevant"}


@pytest.mark.asyncio
async def test_flush_with_nested_dataclass_containing_datetime(tracer: SQLiteTracer) -> None:
    """Flush handles dataclass with datetime fields (the scout bug)."""

    @dataclass(frozen=True, slots=True)
    class Message:
        author: str
        created_at: datetime

    @dataclass(frozen=True, slots=True)
    class EvalResult:
        message: Message
        score: float

    ts = datetime(2026, 3, 20, 12, 0, 0)
    result = EvalResult(message=Message(author="alice", created_at=ts), score=0.9)

    span = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    step = tracer.start_span(span.id, SpanKind.PIPELINE_STEP, "evaluate")
    tracer.end_span(step.id, output=result)
    tracer.end_span(span.id)

    await tracer.flush()

    spans = await tracer.get_trace(span.trace_id)
    step_span = next(s for s in spans if s.name == "evaluate")
    assert step_span.output["score"] == 0.9
    assert step_span.output["message"]["author"] == "alice"
    assert step_span.output["message"]["created_at"] == ts.isoformat()


@pytest.mark.asyncio
async def test_flush_with_non_serializable_object(tracer: SQLiteTracer) -> None:
    """Flush falls back to repr for objects that aren't dataclasses."""

    class CustomObj:
        def __repr__(self) -> str:
            return "<CustomObj>"

    span = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(span.id, output=CustomObj())

    await tracer.flush()

    spans = await tracer.get_trace(span.trace_id)
    assert spans[0].output == "<CustomObj>"


@pytest.mark.asyncio
async def test_close_flushes_remaining_spans(tracer: SQLiteTracer, tmp_path: Any) -> None:
    """close() flushes before closing the DB."""
    span = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(span.id, output="done")

    await tracer.close()

    # Reopen to verify data was persisted — path matches the tracer fixture
    tracer2 = SQLiteTracer(db_path=str(tmp_path / "test_traces.db"))
    spans = await tracer2.get_trace(span.trace_id)
    assert len(spans) == 1
    assert spans[0].output == "done"
    await tracer2.close()


@pytest.mark.asyncio
async def test_pipeline_with_sqlite_tracer(tracer: SQLiteTracer) -> None:
    """End-to-end: run a pipeline with SQLiteTracer, verify spans are persisted."""

    async def add_one(ctx: dict[str, Any]) -> int:
        return ctx["input"] + 1

    async def double(ctx: dict[str, Any]) -> int:
        return ctx["add_one"] * 2

    result = await run_pipeline(
        PipelineConfig(
            name="math",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(name="double", fn=double),
            ],
            tracer=tracer,
        ),
        input=5,
    )

    assert result.step_outputs["add_one"] == 6
    assert result.step_outputs["double"] == 12

    await tracer.flush()

    # Find the trace
    db = await tracer._get_db()
    cursor = await db.execute(
        "SELECT DISTINCT trace_id FROM spans WHERE kind = ?",
        (SpanKind.PIPELINE_RUN.value,),
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1

    spans = await tracer.get_trace(rows[0][0])
    assert len(spans) == 3  # 1 pipeline_run + 2 pipeline_step
    kinds = [s.kind for s in spans]
    assert kinds.count(SpanKind.PIPELINE_RUN) == 1
    assert kinds.count(SpanKind.PIPELINE_STEP) == 2
    await tracer.close()


@pytest.mark.asyncio
async def test_list_traces_includes_agent_run_and_pipeline_run_roots(
    tracer: SQLiteTracer,
) -> None:
    agent_span = tracer.start_trace("agent", kind=SpanKind.AGENT_RUN)
    tracer.end_span(agent_span.id)
    pipeline_span = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(pipeline_span.id)
    await tracer.flush()

    traces = await tracer.list_traces()
    names = {t.name for t in traces}
    assert names == {"agent", "pipeline"}


@pytest.mark.asyncio
async def test_list_traces_since_normalizes_non_utc_offset(tracer: SQLiteTracer) -> None:
    """A since filter with a non-UTC offset must compare correctly against
    UTC-stored timestamps, not be compared as raw mismatched-offset text."""
    span = tracer.start_trace("agent", kind=SpanKind.AGENT_RUN)
    tracer.end_span(span.id)
    await tracer.flush()

    stored_utc = span.started_at.astimezone(UTC)
    # Same instant, expressed in a +05:00 offset — well before the actual
    # UTC-stored timestamp) once correctly normalized.
    since_plus5 = (stored_utc - timedelta(minutes=1)).astimezone(timezone(timedelta(hours=5)))

    traces = await tracer.list_traces(since=since_plus5)
    assert [t.name for t in traces] == ["agent"]

    since_after = (stored_utc + timedelta(minutes=1)).astimezone(timezone(timedelta(hours=5)))
    traces_after = await tracer.list_traces(since=since_after)
    assert traces_after == []


@pytest.mark.asyncio
async def test_list_traces_excludes_nested_pipeline_run(tracer: SQLiteTracer) -> None:
    """A map_pipeline item's PIPELINE_RUN has a parent_id and isn't a root."""
    parent = tracer.start_trace("batch", kind=SpanKind.PIPELINE_RUN)
    nested = tracer.start_span(parent.id, SpanKind.PIPELINE_RUN, "item-0")
    tracer.end_span(nested.id)
    tracer.end_span(parent.id)
    await tracer.flush()

    traces = await tracer.list_traces()
    names = [t.name for t in traces]
    assert names == ["batch"]


@pytest.mark.asyncio
async def test_list_traces_excludes_pipeline_step_and_other_kinds(
    tracer: SQLiteTracer,
) -> None:
    root = tracer.start_trace("agent", kind=SpanKind.AGENT_RUN)
    step = tracer.start_span(root.id, SpanKind.PIPELINE_STEP, "step-0")
    tracer.end_span(step.id)
    tracer.end_span(root.id)
    await tracer.flush()

    traces = await tracer.list_traces()
    assert [t.name for t in traces] == ["agent"]


@pytest.mark.asyncio
async def test_new_spans_get_aware_utc_timestamps(tracer: SQLiteTracer) -> None:
    span = tracer.start_trace("root", kind=SpanKind.AGENT_RUN)
    assert span.started_at.tzinfo is not None
    assert span.started_at.utcoffset() == timedelta(0)

    tracer.end_span(span.id)
    ended = tracer._spans[span.id].ended_at
    assert ended is not None
    assert ended.tzinfo is not None
    assert ended.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_round_tripped_span_timestamps_are_aware(tracer: SQLiteTracer) -> None:
    span = tracer.start_trace("root", kind=SpanKind.AGENT_RUN)
    tracer.end_span(span.id)
    await tracer.flush()

    spans = await tracer.get_trace(span.trace_id)
    assert spans[0].started_at.tzinfo is not None
    assert spans[0].ended_at.tzinfo is not None


@pytest.mark.asyncio
async def test_legacy_naive_span_timestamps_read_as_aware_utc(tmp_path: Any) -> None:
    """Rows written before aware timestamps existed must still round-trip."""
    from jig.tracing.sqlite import _SCHEMA

    db_path = str(tmp_path / "test_traces.db")
    conn = await aiosqlite.connect(db_path)
    try:
        await conn.executescript(_SCHEMA)
        await conn.execute(
            """INSERT INTO spans
               (id, trace_id, parent_id, kind, name, started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("legacy-1", "trace-legacy", None, "agent_run", "legacy",
             "2024-01-01T10:00:00", "2024-01-01T10:00:01"),
        )
        await conn.commit()
    finally:
        await conn.close()

    legacy_tracer = SQLiteTracer(db_path=db_path)
    try:
        spans = await legacy_tracer.get_trace("trace-legacy")
        assert len(spans) == 1
        assert spans[0].started_at.tzinfo is not None
        assert spans[0].started_at == datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        assert spans[0].ended_at == datetime(2024, 1, 1, 10, 0, 1, tzinfo=UTC)
    finally:
        await legacy_tracer.close()


@pytest.mark.asyncio
async def test_flush_safe_under_concurrent_start_span(tracer: SQLiteTracer) -> None:
    """Regression: flush() snapshots ``self._spans`` before iterating so a
    concurrent task adding spans (between our awaits in the flush body)
    doesn't trigger ``RuntimeError: dictionary changed size during
    iteration``.

    Pre-fix repro: this raised under ``concurrency=3`` specialist sweeps —
    one task in flush, the others mid-LLM-call adding ``llm_call`` spans.
    """
    root = tracer.start_trace("root", kind=SpanKind.AGENT_RUN)
    # Seed enough completed spans that flush() takes several await
    # iterations — gives the writer task many chances to interleave.
    for i in range(50):
        s = tracer.start_span(root.id, SpanKind.LLM_CALL, f"pre-{i}")
        tracer.end_span(s.id)

    async def writer() -> None:
        for i in range(50):
            s = tracer.start_span(root.id, SpanKind.LLM_CALL, f"concurrent-{i}")
            tracer.end_span(s.id)
            # Yield so flush's await gets a chance to resume mid-iteration.
            await asyncio.sleep(0)

    # Pre-fix: raises RuntimeError. Post-fix: completes cleanly.
    await asyncio.gather(tracer.flush(), writer())

    # Tail flush picks up whatever the concurrent writer added that the
    # first flush deferred. End the root, drain, and verify everything's
    # durable.
    tracer.end_span(root.id)
    await tracer.flush()
    persisted = await tracer.get_trace(root.trace_id)
    names = {s.name for s in persisted}
    assert "root" in names
    for i in range(50):
        assert f"pre-{i}" in names
        assert f"concurrent-{i}" in names


# ---------------------------------------------------------------------------
# Pipeline failure acceptance tests (SQLiteTracer-backed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_step_failure_immediately_readable(tmp_path: Any) -> None:
    """A failing pipeline step leaves root and step spans readable in SQLiteTracer
    immediately after run_pipeline raises — no manual flush required by the caller.

    This is the acceptance test for the pipeline trace-flush guarantee introduced
    in the lifecycle hardening cohort: run_pipeline must close the root span and
    flush the tracer in all exit paths, including exception propagation.
    """
    db = SQLiteTracer(db_path=str(tmp_path / "fail.db"))

    captured_trace_ids: list[str] = []
    _orig = db.start_trace

    def _capture(*args: Any, **kwargs: Any) -> Any:
        span = _orig(*args, **kwargs)
        captured_trace_ids.append(span.trace_id)
        return span

    db.start_trace = _capture  # type: ignore[method-assign]

    async def boom(ctx: dict[str, Any]) -> str:
        raise ValueError("step exploded")

    try:
        await run_pipeline(
            PipelineConfig(
                name="failing",
                steps=[Step(name="boom", fn=boom)],
                tracer=db,
            ),
            input=1,
        )
    except ValueError:
        pass

    assert captured_trace_ids, "start_trace was never called"
    spans = await db.get_trace(captured_trace_ids[0])

    root = next((s for s in spans if s.kind == SpanKind.PIPELINE_RUN), None)
    assert root is not None, "pipeline root span must be flushed to DB on exception"
    assert root.ended_at is not None, "root span must be closed"
    assert root.error is not None, "root span must record the error"

    step = next((s for s in spans if s.kind == SpanKind.PIPELINE_STEP), None)
    assert step is not None, "failing step span must be flushed to DB"
    assert step.ended_at is not None, "step span must be closed"
    assert step.error is not None, "step span must record the error"
    # Error detail must be non-empty so callers can diagnose without log scraping
    assert "ValueError" in step.error or "step exploded" in step.error

    await db.close()


@pytest.mark.asyncio
async def test_span_end_during_flush_is_retained_and_persists_final_state(
    tracer: SQLiteTracer,
) -> None:
    """A span that ends during flush() must not be permanently frozen as open.

    Pre-fix: flush() evicted spans using current ``span.ended_at`` at eviction
    time. A span open at snapshot time but ended during the DB writes would be
    evicted immediately — the row in SQLite remained open forever because
    ``_spans`` no longer held the entry needed for a follow-up flush.

    Post-fix: eviction uses the open-at-snapshot set captured before any await.
    A span open at snapshot time is retained even if end_span() fires during
    the DB writes, giving it one more flush cycle to persist its final state.
    """
    span = tracer.start_trace("root", kind=SpanKind.AGENT_RUN)
    # span is open at this point — ended_at is None

    # Intercept db.execute to call end_span() immediately after the span row
    # is first written.  This simulates a caller ending the span between
    # flush()'s snapshot (where ended_at was None) and the DB write commit.
    db = await tracer._get_db()
    original_execute = db.execute
    end_injected = False

    async def intercepting_execute(*args: Any, **kwargs: Any) -> Any:
        nonlocal end_injected
        result = await original_execute(*args, **kwargs)
        if not end_injected and args and "INSERT OR REPLACE INTO spans" in str(args[0]):
            # Row was just written with ended_at=NULL (span was open at snapshot).
            # End the span now — simulating concurrent end_span during flush.
            tracer.end_span(span.id, output="ended-mid-flush")
            end_injected = True
        return result

    db.execute = intercepting_execute  # type: ignore[method-assign]
    try:
        await tracer.flush()
    finally:
        db.execute = original_execute  # type: ignore[method-assign]

    assert end_injected, "test setup error: intercepting_execute was never triggered"

    # After first flush: span was open at snapshot time, so it must still
    # be in _spans even though end_span() fired during the DB write.
    # Pre-fix: span would be evicted here (s.ended_at is not None at eviction time).
    assert span.id in tracer._spans, (
        "span open at snapshot time must be retained after flush even if "
        "end_span() fired during the DB writes"
    )

    # Second flush captures the final ended_at / output.
    await tracer.flush()

    # Now evicted — span was ended when this flush snapshotted it.
    assert span.id not in tracer._spans, "ended span must be evicted after second flush"

    # DB must contain the final state, not the stale open-row.
    spans = await tracer.get_trace(span.trace_id)
    assert len(spans) == 1
    assert spans[0].ended_at is not None, "final ended_at must be persisted"
    assert spans[0].output == "ended-mid-flush", "final output must be persisted"


@pytest.mark.asyncio
async def test_graderless_pipeline_success_immediately_readable(tmp_path: Any) -> None:
    """A pipeline with no grader still flushes spans immediately on success.

    Proves that the tracer flush guarantee covers the no-grader path (where there
    is no pre-grade flush call), so callers always see completed spans after
    run_pipeline returns without having to call tracer.flush() themselves.
    """
    db = SQLiteTracer(db_path=str(tmp_path / "nograde.db"))

    async def add_one(ctx: dict[str, Any]) -> int:
        return ctx["input"] + 1

    result = await run_pipeline(
        PipelineConfig(
            name="nograde",
            steps=[Step(name="add_one", fn=add_one)],
            tracer=db,
        ),
        input=10,
    )
    assert result.output == 11

    spans = await db.get_trace(result.trace_id)
    assert len(spans) == 2  # pipeline_run + pipeline_step

    root = next(s for s in spans if s.kind == SpanKind.PIPELINE_RUN)
    assert root.ended_at is not None

    step = next(s for s in spans if s.kind == SpanKind.PIPELINE_STEP)
    assert step.ended_at is not None
    assert step.error is None

    await db.close()
