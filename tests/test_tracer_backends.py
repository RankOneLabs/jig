"""TracingLogger contract tests, run against SQLiteTracer and PostgresTracer.

Every case here holds for both backends, so a behaviour that drifts in one
fails here rather than in a consumer that swapped backends.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from jig import PipelineConfig, Step, run_pipeline
from jig.core.types import SpanKind, TracingLogger, Usage
from jig.tracing import SQLiteTracer


@pytest.fixture(params=("sqlite", "postgres"))
async def tracer(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[TracingLogger]:
    backend: TracingLogger
    if request.param == "sqlite":
        backend = SQLiteTracer(db_path=str(tmp_path / "traces.db"))
    else:
        from jig.tracing.postgres import PostgresTracer

        backend = PostgresTracer(
            request.getfixturevalue("pg_dsn"), schema=request.getfixturevalue("pg_schema"),
        )
    try:
        yield backend
    finally:
        await backend.close()  # type: ignore[attr-defined]


async def test_flushed_span_round_trips_every_field(tracer: TracingLogger) -> None:
    root = tracer.start_trace("agent", metadata={"model": "m1", "strategy_name": "s1"})
    step = tracer.start_span(
        root.id, SpanKind.LLM_CALL, "call", input={"prompt": "hi"}, metadata={"turn": 1},
    )
    tracer.end_span(
        step.id, output={"text": "hello"}, error="boom",
        usage=Usage(input_tokens=10, output_tokens=4, cost=0.25),
    )
    tracer.end_span(root.id)
    await tracer.flush()

    spans = await tracer.get_trace(root.trace_id)
    got = next(s for s in spans if s.id == step.id)
    assert (
        got.trace_id, got.parent_id, got.kind, got.name, got.input, got.output,
        got.metadata, got.error, got.usage,
    ) == (
        root.trace_id, root.id, SpanKind.LLM_CALL, "call", {"prompt": "hi"}, {"text": "hello"},
        {"turn": 1}, "boom", Usage(input_tokens=10, output_tokens=4, cost=0.25),
    )


async def test_get_trace_orders_spans_by_start(tracer: TracingLogger) -> None:
    root = tracer.start_trace("agent")
    first = tracer.start_span(root.id, SpanKind.TOOL_CALL, "first")
    second = tracer.start_span(root.id, SpanKind.TOOL_CALL, "second")
    for span in (first, second, root):
        tracer.end_span(span.id)
    await tracer.flush()

    spans = await tracer.get_trace(root.trace_id)
    assert [s.name for s in spans] == ["agent", "first", "second"]


async def test_dataclass_output_serializes_as_object(tracer: TracingLogger) -> None:
    @dataclass(frozen=True, slots=True)
    class EvalResult:
        score: float
        at: datetime

    at = datetime(2026, 1, 2, tzinfo=UTC)
    root = tracer.start_trace("eval", kind=SpanKind.PIPELINE_RUN)
    tracer.end_span(root.id, output=EvalResult(score=0.85, at=at))
    await tracer.flush()

    [span] = await tracer.get_trace(root.trace_id)
    assert span.output == {"score": 0.85, "at": at.isoformat()}


async def test_unserializable_output_falls_back_to_repr(tracer: TracingLogger) -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    root = tracer.start_trace("agent")
    tracer.end_span(root.id, output={"thing": Opaque()})
    await tracer.flush()

    [span] = await tracer.get_trace(root.trace_id)
    assert span.output == {"thing": "<opaque>"}


async def test_list_traces_returns_only_parentless_run_roots(tracer: TracingLogger) -> None:
    agent = tracer.start_trace("agent", kind=SpanKind.AGENT_RUN)
    pipeline = tracer.start_trace("pipeline", kind=SpanKind.PIPELINE_RUN)
    nested = tracer.start_span(pipeline.id, SpanKind.PIPELINE_RUN, "item-0")
    step = tracer.start_span(agent.id, SpanKind.PIPELINE_STEP, "step-0")
    other_root = tracer.start_trace("grading", kind=SpanKind.GRADING)
    for span in (nested, step, agent, pipeline, other_root):
        tracer.end_span(span.id)
    await tracer.flush()

    traces = await tracer.list_traces()
    assert {t.name for t in traces} == {"agent", "pipeline"}


async def test_list_traces_newest_first_and_limited(tracer: TracingLogger) -> None:
    for name in ("one", "two", "three"):
        span = tracer.start_trace(name)
        tracer.end_span(span.id)
        await asyncio.sleep(0.002)
    await tracer.flush()

    traces = await tracer.list_traces(limit=2)
    assert [t.name for t in traces] == ["three", "two"]


async def test_list_traces_filters_by_name(tracer: TracingLogger) -> None:
    for name in ("keep", "drop"):
        span = tracer.start_trace(name)
        tracer.end_span(span.id)
    await tracer.flush()

    traces = await tracer.list_traces(name="keep")
    assert [t.name for t in traces] == ["keep"]


async def test_list_traces_since_compares_instants_across_offsets(tracer: TracingLogger) -> None:
    span = tracer.start_trace("agent")
    tracer.end_span(span.id)
    await tracer.flush()

    plus5 = timezone(timedelta(hours=5))
    before = (span.started_at - timedelta(minutes=1)).astimezone(plus5)
    after = (span.started_at + timedelta(minutes=1)).astimezone(plus5)
    assert (
        [t.name for t in await tracer.list_traces(since=before)],
        await tracer.list_traces(since=after),
    ) == (["agent"], [])


async def test_list_traces_reads_naive_since_as_utc(tracer: TracingLogger) -> None:
    span = tracer.start_trace("agent")
    tracer.end_span(span.id)
    await tracer.flush()

    naive_before = (span.started_at - timedelta(minutes=1)).replace(tzinfo=None)
    traces = await tracer.list_traces(since=naive_before)
    assert [t.name for t in traces] == ["agent"]


async def test_round_tripped_timestamps_are_aware_utc(tracer: TracingLogger) -> None:
    span = tracer.start_trace("agent")
    tracer.end_span(span.id)
    await tracer.flush()

    [got] = await tracer.get_trace(span.trace_id)
    assert (got.started_at, got.ended_at, got.started_at.utcoffset()) == (
        span.started_at, span.ended_at, timedelta(0),
    )


async def test_span_open_at_flush_persists_its_final_state_next_flush(
    tracer: TracingLogger,
) -> None:
    root = tracer.start_trace("agent")
    await tracer.flush()
    tracer.end_span(root.id, output={"done": True})
    await tracer.flush()

    [got] = await tracer.get_trace(root.trace_id)
    assert (got.ended_at is not None, got.output) == (True, {"done": True})


async def test_pipeline_run_is_readable_after_flush(tracer: TracingLogger) -> None:
    async def add_one(ctx: dict[str, Any]) -> int:
        return ctx["input"] + 1

    await run_pipeline(
        PipelineConfig(name="math", steps=[Step(name="add_one", fn=add_one)], tracer=tracer),
        input=5,
    )
    await tracer.flush()

    [root] = await tracer.list_traces()
    kinds = sorted(s.kind.value for s in await tracer.get_trace(root.trace_id))
    assert kinds == sorted([SpanKind.PIPELINE_RUN.value, SpanKind.PIPELINE_STEP.value])
