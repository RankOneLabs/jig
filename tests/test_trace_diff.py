"""Tests for :func:`jig.trace_diff`."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from jig import Span, SpanKind, Usage, trace_diff
from jig.core.types import TracingLogger
from jig.replay.diff import TraceDiff
from jig.tracing import SQLiteTracer


class _StubTracer(TracingLogger):
    """Minimal tracer that serves a pre-built span list per trace_id.

    FakeTracer in test_core.py hard-codes trace_id='t-0' which makes
    multi-trace tests impossible. This stub accepts preset traces and
    returns them from ``get_trace``.
    """

    def __init__(self, traces: dict[str, list[Span]]):
        self._traces = traces

    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):  # pragma: no cover
        raise NotImplementedError

    def start_span(self, parent_id, kind, name, input=None):  # pragma: no cover
        raise NotImplementedError

    def end_span(self, span_id, output=None, error=None, usage=None):  # pragma: no cover
        raise NotImplementedError

    async def get_trace(self, trace_id: str) -> list[Span]:
        return self._traces.get(trace_id, [])

    async def list_traces(self, since=None, limit=50, name=None):  # pragma: no cover
        return []

    async def flush(self) -> None:
        pass


def _root(trace_id: str, *, output: str = "", error_category: str | None = None, duration_ms: float = 10.0) -> Span:
    out: dict[str, Any] = {"output": output, "scores": None}
    if error_category is not None:
        out["error_category"] = error_category
    return Span(
        id=f"{trace_id}-root",
        trace_id=trace_id,
        kind=SpanKind.AGENT_RUN,
        name="agent",
        started_at=datetime.now(),
        ended_at=datetime.now(),
        duration_ms=duration_ms,
        input=None,
        output=out,
    )


def _tool(trace_id: str, idx: int, name: str, args: dict[str, Any], output: str, error: str | None = None) -> Span:
    return Span(
        id=f"{trace_id}-tool-{idx}",
        trace_id=trace_id,
        kind=SpanKind.TOOL_CALL,
        name=name,
        started_at=datetime.now(),
        parent_id=f"{trace_id}-root",
        input=args,
        output=output,
        error=error,
    )


def _grading(trace_id: str, scores: list[dict[str, Any]]) -> Span:
    return Span(
        id=f"{trace_id}-grade",
        trace_id=trace_id,
        kind=SpanKind.GRADING,
        name="grade",
        started_at=datetime.now(),
        parent_id=f"{trace_id}-root",
        output={"scores": scores},
    )


def _llm(trace_id: str, idx: int, cost: float) -> Span:
    return Span(
        id=f"{trace_id}-llm-{idx}",
        trace_id=trace_id,
        kind=SpanKind.LLM_CALL,
        name="completion",
        started_at=datetime.now(),
        parent_id=f"{trace_id}-root",
        usage=Usage(input_tokens=5, output_tokens=5, cost=cost),
    )


@pytest.mark.asyncio
async def test_identical_traces_are_empty_diff():
    spans = [
        _root("a", output="done"),
        _tool("a", 0, "echo", {"text": "x"}, "x"),
    ]
    spans_b = [
        _root("b", output="done"),
        _tool("b", 0, "echo", {"text": "x"}, "x"),
    ]
    tracer = _StubTracer({"a": spans, "b": spans_b})

    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.identical is True
    assert diff.tool_divergence == []
    assert diff.output_diff is None


@pytest.mark.asyncio
async def test_args_divergence_classified():
    tracer = _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "y"}, "y")],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert len(diff.tool_divergence) == 1
    d = diff.tool_divergence[0]
    assert d.divergence == "args"
    assert d.a.args == {"text": "x"}
    assert d.b.args == {"text": "y"}


@pytest.mark.asyncio
async def test_only_b_trailing_tool():
    tracer = _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [
            _root("b"),
            _tool("b", 0, "echo", {"text": "x"}, "x"),
            _tool("b", 1, "echo", {"text": "extra"}, "extra"),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert len(diff.tool_divergence) == 1
    assert diff.tool_divergence[0].divergence == "only_b"
    assert diff.tool_divergence[0].a is None
    assert diff.tool_divergence[0].b.name == "echo"


@pytest.mark.asyncio
async def test_only_a_missing_from_b():
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _tool("a", 0, "echo", {"text": "x"}, "x"),
            _tool("a", 1, "echo", {"text": "y"}, "y"),
        ],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "x"}, "x")],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert len(diff.tool_divergence) == 1
    assert diff.tool_divergence[0].divergence == "only_a"


@pytest.mark.asyncio
async def test_output_diff_populated_when_finals_differ():
    tracer = _StubTracer({
        "a": [_root("a", output="alpha")],
        "b": [_root("b", output="beta")],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.output_diff == ("alpha", "beta")


@pytest.mark.asyncio
async def test_error_category_change_populated():
    tracer = _StubTracer({
        "a": [_root("a", error_category="max_llm_calls")],
        "b": [_root("b", error_category=None)],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.error_category_change == ("max_llm_calls", None)


@pytest.mark.asyncio
async def test_score_deltas_computed_per_dimension():
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _grading("a", [{"dimension": "quality", "value": 0.6}]),
        ],
        "b": [
            _root("b"),
            _grading("b", [{"dimension": "quality", "value": 0.8}]),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.score_deltas == {"quality": pytest.approx(0.2)}


@pytest.mark.asyncio
async def test_score_deltas_surface_added_and_dropped_dimensions():
    """A dimension present in only one trace must show up in the diff —
    otherwise a grader change (adding/removing a rubric entry) would
    silently look identical and `identical` would wrongly return True."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _grading("a", [
                {"dimension": "quality", "value": 0.7},
                {"dimension": "accuracy", "value": 0.5},
            ]),
        ],
        "b": [
            _root("b"),
            _grading("b", [
                {"dimension": "quality", "value": 0.7},
                {"dimension": "relevance", "value": 0.9},
            ]),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    # accuracy was dropped (present in A, missing in B → negative delta)
    # relevance was added (missing in A, present in B → positive delta)
    # quality identical → omitted
    assert "quality" not in diff.score_deltas
    assert diff.score_deltas["accuracy"] == pytest.approx(-0.5)
    assert diff.score_deltas["relevance"] == pytest.approx(0.9)
    assert diff.identical is False


@pytest.mark.asyncio
async def test_cost_and_latency_delta():
    tracer = _StubTracer({
        "a": [
            _root("a", duration_ms=100.0),
            _llm("a", 0, cost=0.01),
            _llm("a", 1, cost=0.02),
        ],
        "b": [
            _root("b", duration_ms=250.0),
            _llm("b", 0, cost=0.05),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.cost_delta == pytest.approx(0.02)
    assert diff.latency_ms_delta == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_submit_output_spans_ignored_in_diff():
    """``submit_output`` is runner-internal; diff must not surface it."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            Span(
                id="a-so",
                trace_id="a",
                kind=SpanKind.TOOL_CALL,
                name="submit_output",
                started_at=datetime.now(),
                input={"result": "x"},
                output="{}",
            ),
        ],
        "b": [
            _root("b"),
            Span(
                id="b-so",
                trace_id="b",
                kind=SpanKind.TOOL_CALL,
                name="submit_output",
                started_at=datetime.now(),
                input={"result": "y"},
                output="{}",
            ),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.tool_divergence == []


# --- SQLite integration ---


@pytest.mark.asyncio
async def test_trace_diff_against_sqlite_backend(tmp_path: Any):
    """Sanity check the SQLite-backed read path (not just the stub)."""
    tracer = SQLiteTracer(db_path=str(tmp_path / "d.db"))

    def write(name: str, tool_arg: str, final: str) -> str:
        root = tracer.start_trace(name)
        tool = tracer.start_span(root.id, SpanKind.TOOL_CALL, "echo", {"text": tool_arg})
        tracer.end_span(tool.id, tool_arg)
        tracer.end_span(root.id, {"output": final, "scores": None})
        return root.trace_id

    t_a = write("A", "same", "done_a")
    t_b = write("B", "same", "done_b")
    await tracer.flush()

    diff: TraceDiff = await trace_diff(t_a, t_b, tracer=tracer)
    assert diff.tool_divergence == []
    assert diff.output_diff == ("done_a", "done_b")
