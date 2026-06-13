"""Tests for SQLiteTracer — flush, serialization, round-trip."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
    assert len(traces) == 0  # list_traces filters by AGENT_RUN

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
async def test_close_flushes_remaining_spans(tracer: SQLiteTracer) -> None:
    """close() flushes before closing the DB."""
    span = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(span.id, output="done")

    await tracer.close()

    # Reopen to verify data was persisted
    tracer2 = SQLiteTracer(db_path=tracer._conn._db_path)
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
