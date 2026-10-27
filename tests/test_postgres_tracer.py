"""PostgresTracer guarantees the shared contract does not cover: the bounded
buffer, surviving a database outage, queryable jsonb metadata, namespaces,
and values jsonb rejects that SQLite stored verbatim."""
from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator, Callable
from typing import Any

import psycopg
import pytest

from jig import PipelineConfig, Step, run_pipeline
from jig.core.types import SpanKind
from jig.tracing.postgres import PostgresTracer, tracer_schema_sql

# Nothing listens here: connecting fails fast with a refused connection.
DEAD_DSN = "postgresql://jig:jig@127.0.0.1:1/jig?connect_timeout=2"

type TracerFactory = Callable[..., PostgresTracer]


@pytest.fixture
async def make_tracer(pg_dsn: str, pg_schema: str) -> AsyncIterator[TracerFactory]:
    opened: list[PostgresTracer] = []

    def factory(**kwargs: Any) -> PostgresTracer:
        kwargs.setdefault("schema", pg_schema)
        dsn = kwargs.pop("dsn", pg_dsn)
        tracer = PostgresTracer(dsn, **kwargs)
        opened.append(tracer)
        return tracer

    try:
        yield factory
    finally:
        for tracer in opened:
            await tracer.close()


def _sync_query(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


def _sync_execute(dsn: str, sql: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql)


async def _one_finished_span_per_trace(tracer: PostgresTracer, count: int) -> list[str]:
    trace_ids = []
    for i in range(count):
        span = tracer.start_trace(f"t{i}")
        tracer.end_span(span.id)
        trace_ids.append(span.trace_id)
    return trace_ids


# --- bounded buffer ---------------------------------------------------------


async def test_buffer_past_its_bound_counts_every_dropped_span(make_tracer: TracerFactory) -> None:
    tracer = make_tracer(max_buffered_spans=3)
    await _one_finished_span_per_trace(tracer, 5)
    assert tracer.dropped_span_count == 2


async def test_buffer_keeps_the_newest_spans(make_tracer: TracerFactory) -> None:
    tracer = make_tracer(max_buffered_spans=3)
    trace_ids = await _one_finished_span_per_trace(tracer, 5)
    await tracer.flush()

    stored = {t.trace_id for t in await tracer.list_traces()}
    assert stored == set(trace_ids[2:])


async def test_every_drop_is_logged_with_its_trace_id(
    make_tracer: TracerFactory, caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = make_tracer(max_buffered_spans=1)
    caplog.set_level(logging.WARNING, logger="jig.tracing.postgres")
    trace_ids = await _one_finished_span_per_trace(tracer, 3)

    logged = [r.getMessage() for r in caplog.records]
    assert [any(tid in m for m in logged) for tid in trace_ids[:2]] == [True, True]


async def test_eviction_spares_an_open_root_while_finished_spans_remain(
    make_tracer: TracerFactory,
) -> None:
    tracer = make_tracer(max_buffered_spans=3)
    root = tracer.start_trace("long-run")
    for i in range(5):
        child = tracer.start_span(root.id, SpanKind.TOOL_CALL, f"c{i}")
        tracer.end_span(child.id)
    tracer.end_span(root.id)
    await tracer.flush()

    names = {s.name for s in await tracer.get_trace(root.trace_id)}
    assert names == {"long-run", "c3", "c4"}


async def test_pipeline_completes_when_buffer_overflows_with_database_down(
    make_tracer: TracerFactory,
) -> None:
    """The mutation-check case for the bound: fill the buffer past it while
    every flush fails, and the run must still complete with drops counted."""
    tracer = make_tracer(dsn=DEAD_DSN, max_buffered_spans=2)

    async def step(ctx: dict[str, Any]) -> int:
        return ctx["input"] + 1

    result = await run_pipeline(
        PipelineConfig(
            name="overflow",
            steps=[Step(name=f"s{i}", fn=step) for i in range(4)],
            tracer=tracer,
        ),
        input=0,
    )
    assert (result.step_outputs["s0"], tracer.dropped_span_count > 0) == (1, True)


def test_rejects_a_non_positive_bound() -> None:
    with pytest.raises(ValueError, match="max_buffered_spans"):
        PostgresTracer(DEAD_DSN, max_buffered_spans=0)


# --- outage -----------------------------------------------------------------


async def test_flush_against_a_dead_server_does_not_raise(make_tracer: TracerFactory) -> None:
    tracer = make_tracer(dsn=DEAD_DSN)
    tracer.start_trace("agent")
    await tracer.flush()
    assert tracer.failed_flush_count == 1


async def test_failed_flush_keeps_every_span_buffered(make_tracer: TracerFactory) -> None:
    tracer = make_tracer(dsn=DEAD_DSN)
    await _one_finished_span_per_trace(tracer, 3)
    await tracer.flush()
    assert len(tracer._spans) == 3


async def test_failed_flush_is_logged_with_trace_ids(
    make_tracer: TracerFactory, caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = make_tracer(dsn=DEAD_DSN)
    caplog.set_level(logging.ERROR, logger="jig.tracing.postgres")
    [trace_id] = await _one_finished_span_per_trace(tracer, 1)
    await tracer.flush()
    assert any(trace_id in r.getMessage() for r in caplog.records)


async def test_flush_after_a_dropped_connection_persists_the_retained_spans(
    make_tracer: TracerFactory, pg_dsn: str,
) -> None:
    tracer = make_tracer()
    first = tracer.start_trace("before")
    tracer.end_span(first.id)
    await tracer.flush()
    backend_pid = (await tracer._conn.get()).info.backend_pid
    _sync_query(pg_dsn, "SELECT pg_terminate_backend(%s)", (backend_pid,))

    second = tracer.start_trace("after")
    tracer.end_span(second.id)
    await tracer.flush()  # fails on the terminated connection, keeping the span
    await tracer.flush()  # reconnects

    assert (
        tracer.failed_flush_count,
        [t.name for t in await tracer.list_traces(name="after")],
    ) == (1, ["after"])


# --- storage shape ----------------------------------------------------------


async def test_span_metadata_is_queryable_jsonb(
    make_tracer: TracerFactory, pg_dsn: str, pg_schema: str,
) -> None:
    """Consumers attribute runs by parsing metadata (model, strategy name);
    opaque text would leave that attribution null for every row."""
    tracer = make_tracer()
    span = tracer.start_trace("explorer", metadata={"strategy_name": "funding_fade"})
    tracer.end_span(span.id)
    await tracer.flush()

    rows = _sync_query(
        pg_dsn,
        f"SELECT metadata->>'strategy_name' FROM \"{pg_schema}\".spans WHERE id = %s",
        (span.id,),
    )
    assert rows == [("funding_fade",)]


async def test_namespaces_do_not_see_each_others_traces(make_tracer: TracerFactory) -> None:
    trial_a = make_tracer(namespace="trial-a")
    trial_b = make_tracer(namespace="trial-b")
    span = trial_a.start_trace("agent")
    trial_a.end_span(span.id)
    await trial_a.flush()

    assert (await trial_b.list_traces(), await trial_b.get_trace(span.trace_id)) == ([], [])


async def test_non_finite_floats_are_stored_as_null(make_tracer: TracerFactory) -> None:
    tracer = make_tracer()
    span = tracer.start_trace("agent")
    tracer.end_span(span.id, output={"sharpe": math.nan, "calmar": math.inf, "trades": 3})
    await tracer.flush()

    [got] = await tracer.get_trace(span.trace_id)
    assert got.output == {"sharpe": None, "calmar": None, "trades": 3}


async def test_nul_characters_are_replaced_not_rejected(make_tracer: TracerFactory) -> None:
    tracer = make_tracer()
    span = tracer.start_trace("a\x00b")
    tracer.end_span(span.id, output={"text": "x\x00y", "escaped": "\\u0000"}, error="e\x00")
    await tracer.flush()

    [got] = await tracer.get_trace(span.trace_id)
    assert (got.name, got.output, got.error) == (
        "a�b", {"text": "x�y", "escaped": "\\u0000"}, "e�",
    )


async def test_externally_applied_schema_needs_no_runtime_ddl(
    make_tracer: TracerFactory, pg_dsn: str, pg_schema: str,
) -> None:
    _sync_execute(pg_dsn, tracer_schema_sql(pg_schema))
    tracer = make_tracer(create_schema=False)
    span = tracer.start_trace("agent")
    tracer.end_span(span.id)
    await tracer.flush()
    assert [t.name for t in await tracer.list_traces()] == ["agent"]


async def test_schema_ddl_is_idempotent(pg_dsn: str, pg_schema: str) -> None:
    ddl = tracer_schema_sql(pg_schema)
    _sync_execute(pg_dsn, ddl)
    _sync_execute(pg_dsn, ddl)
    tables = _sync_query(
        pg_dsn,
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
        (pg_schema,),
    )
    assert tables == [("spans",)]


@pytest.mark.parametrize("schema", ["Telemetry", "tele-metry", 'x"; DROP TABLE y; --', ""])
def test_rejects_schema_names_that_are_not_plain_identifiers(schema: str) -> None:
    with pytest.raises(ValueError, match="schema name"):
        PostgresTracer(DEAD_DSN, schema=schema)
