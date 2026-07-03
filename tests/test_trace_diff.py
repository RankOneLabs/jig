"""Tests for :func:`jig.trace_diff`."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from jig import AgentConfig, CompletionParams, LLMResponse, Score, ScoreSource, Span, SpanKind, Usage, run_agent, trace_diff
from jig.core.types import Grader, LLMClient, TracingLogger
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.tools import ToolRegistry
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

    def start_span(self, parent_id, kind, name, input=None, metadata=None):  # noqa: A002  # pragma: no cover
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
async def test_fully_identical_requires_cost_and_latency_match():
    """``identical`` tolerates cost/latency drift; ``fully_identical`` doesn't.

    Same-behavior traces with different spend should report
    ``identical=True`` (behavioral match, the model-swap use case) but
    ``fully_identical=False`` (stricter full-equality sibling).
    """
    tracer = _StubTracer({
        "a": [_root("a", output="done", duration_ms=10.0), _llm("a", 0, cost=0.01)],
        "b": [_root("b", output="done", duration_ms=50.0), _llm("b", 0, cost=0.05)],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.identical is True
    assert diff.fully_identical is False
    assert diff.cost_delta == pytest.approx(0.04)
    assert diff.latency_ms_delta == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_fully_identical_true_on_byte_for_byte_match():
    tracer = _StubTracer({
        "a": [_root("a", output="done", duration_ms=10.0)],
        "b": [_root("b", output="done", duration_ms=10.0)],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.identical is True
    assert diff.fully_identical is True


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
async def test_missing_trace_ids_raise():
    tracer = _StubTracer({})
    with pytest.raises(ValueError, match="not found or has no spans"):
        await trace_diff("missing-a", "missing-b", tracer=tracer)


@pytest.mark.asyncio
async def test_trace_with_no_agent_run_root_raises():
    # Trace has spans but none is the AGENT_RUN root
    tracer = _StubTracer({
        "a": [_root("a")],
        "b": [_tool("b", 0, "echo", {"text": "x"}, "x")],  # no root!
    })
    with pytest.raises(ValueError, match="no AGENT_RUN root span"):
        await trace_diff("a", "b", tracer=tracer)


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


@pytest.mark.asyncio
async def test_canonical_grading_span_writer_contract(tmp_path: Any):
    """Production-style grading span (canonical {dimension,value} shape written
    directly via the tracer) is correctly parsed by trace_diff."""
    tracer = SQLiteTracer(db_path=str(tmp_path / "canon.db"))

    def write_with_grade(name: str, quality: float) -> str:
        root = tracer.start_trace(name, kind=SpanKind.AGENT_RUN)
        grade = tracer.start_span(root.id, SpanKind.GRADING, "auto_grade")
        tracer.end_span(grade.id, output={
            "scores": [{"dimension": "quality", "value": quality}],
            "feedback_result_id": f"fb-{name}",
        })
        tracer.end_span(root.id, {"output": "done", "scores": None})
        return root.trace_id

    t_a = write_with_grade("A", 0.6)
    t_b = write_with_grade("B", 0.9)
    await tracer.flush()

    diff: TraceDiff = await trace_diff(t_a, t_b, tracer=tracer)
    assert diff.score_deltas == {"quality": pytest.approx(0.3)}
    assert diff.score_details["quality"] == (pytest.approx(0.6), pytest.approx(0.9))
    assert diff.identical is False


# --- Legacy score format support ---


@pytest.mark.asyncio
async def test_legacy_dim_val_scores_are_read():
    """Legacy {dim, val} entries must be accepted alongside canonical {dimension, value}."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _grading("a", [{"dim": "quality", "val": 0.5}]),
        ],
        "b": [
            _root("b"),
            _grading("b", [{"dim": "quality", "val": 0.9}]),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.score_deltas == {"quality": pytest.approx(0.4)}
    assert diff.identical is False


@pytest.mark.asyncio
async def test_canonical_and_legacy_scores_mixed_across_traces():
    """Canonical shape in one trace and legacy shape in another must both be parsed."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _grading("a", [{"dimension": "relevance", "value": 0.6}]),
        ],
        "b": [
            _root("b"),
            _grading("b", [{"dim": "relevance", "val": 0.8}]),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.score_deltas == {"relevance": pytest.approx(0.2)}
    assert diff.identical is False


@pytest.mark.asyncio
async def test_score_details_exposes_per_dimension_old_and_new_values():
    """score_details must record (a_avg, b_avg) for each dimension in the union."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _grading("a", [
                {"dimension": "quality", "value": 0.6},
                {"dimension": "accuracy", "value": 0.8},
            ]),
        ],
        "b": [
            _root("b"),
            _grading("b", [
                {"dimension": "quality", "value": 0.9},
                {"dimension": "relevance", "value": 0.7},
            ]),
        ],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    # quality: changed
    assert diff.score_details["quality"] == (pytest.approx(0.6), pytest.approx(0.9))
    # accuracy: dropped (only in A)
    assert diff.score_details["accuracy"] == (pytest.approx(0.8), None)
    # relevance: added (only in B)
    assert diff.score_details["relevance"] == (None, pytest.approx(0.7))


@pytest.mark.asyncio
async def test_score_details_identical_dimension_has_matching_values():
    """A dimension with the same score in both traces must appear in score_details
    but not in score_deltas."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.7}])],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.7}])],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert "quality" not in diff.score_deltas
    assert diff.score_details["quality"] == (pytest.approx(0.7), pytest.approx(0.7))
    assert diff.identical is True


@pytest.mark.asyncio
async def test_score_delta_zero_value_does_not_lose_dimension():
    """A score of 0.0 must not be confused with an absent dimension (0.0 is falsy)."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.0}])],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.5}])],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert diff.score_deltas == {"quality": pytest.approx(0.5)}
    assert diff.score_details["quality"] == (pytest.approx(0.0), pytest.approx(0.5))


@pytest.mark.asyncio
async def test_added_dimension_with_zero_score_makes_identical_false():
    """A dimension added in B with score 0.0 must still appear in score_deltas and
    make identical=False — the numeric delta is 0.0 but the rubric changed."""
    tracer = _StubTracer({
        "a": [_root("a")],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.0}])],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert "quality" in diff.score_deltas
    assert diff.score_deltas["quality"] == pytest.approx(0.0)
    assert diff.score_details["quality"] == (None, pytest.approx(0.0))
    assert diff.identical is False


@pytest.mark.asyncio
async def test_dropped_dimension_with_zero_score_makes_identical_false():
    """A dimension dropped from A where its score was 0.0 must still appear in
    score_deltas and make identical=False."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.0}])],
        "b": [_root("b")],
    })
    diff = await trace_diff("a", "b", tracer=tracer)
    assert "quality" in diff.score_deltas
    assert diff.score_deltas["quality"] == pytest.approx(0.0)
    assert diff.score_details["quality"] == (pytest.approx(0.0), None)
    assert diff.identical is False


# --- Real runner grading spans ---


async def _fake_embed_diff(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


class _FixedLLM(LLMClient):
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, params: CompletionParams) -> LLMResponse:
        return LLMResponse(
            content=self._content,
            tool_calls=None,
            usage=Usage(10, 5, cost=0.0),
            latency_ms=1,
            model="fixed",
        )


class _FixedGrader(Grader):
    def __init__(self, value: float) -> None:
        self._value = value

    async def grade(self, input: Any, output: Any, context: dict[str, Any] | None = None) -> list[Score]:
        return [Score(dimension="quality", value=self._value, source=ScoreSource.HEURISTIC)]


@pytest.mark.asyncio
async def test_score_deltas_from_real_runner_grading_spans(tmp_path: Any):
    """trace_diff reads score_deltas from grading spans produced by run_agent."""
    tracer = SQLiteTracer(db_path=str(tmp_path / "runner.db"))

    feedback = SQLiteFeedbackLoop(db_path=str(tmp_path / "fb.db"))
    feedback._embed = _fake_embed_diff  # type: ignore[method-assign]

    def _config(name: str, value: float) -> AgentConfig:
        return AgentConfig(
            name=name,
            description="trace diff test",
            system_prompt="You are a test agent.",
            llm=_FixedLLM("answer"),
            feedback=feedback,
            tracer=tracer,
            tools=ToolRegistry(),
            grader=_FixedGrader(value),
        )

    result_a = await run_agent(_config("agent-a", 0.9), "question")
    result_b = await run_agent(_config("agent-b", 0.3), "question")
    await tracer.flush()

    diff = await trace_diff(result_a.trace_id, result_b.trace_id, tracer=tracer)

    assert "quality" in diff.score_deltas
    assert diff.score_deltas["quality"] == pytest.approx(-0.6, abs=1e-6)
    assert diff.identical is False
