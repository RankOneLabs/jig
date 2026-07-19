"""Tests for :func:`jig.trace_diff`."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pytest

from jig import AgentConfig, CompletionParams, LLMResponse, Score, ScoreSource, Span, SpanKind, Usage, run_agent, trace_diff
from jig.core.types import Grader, LLMClient, TracingLogger
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.replay.align import AlignedPair, IdentityAligner, ToolEvent, UnmatchedEvent
from jig.replay.diff import TraceDiff
from jig.tools import ToolRegistry
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


@pytest.fixture(
    params=[
        pytest.param(None, id="identity-fields-none"),
        pytest.param({}, id="identity-fields-empty"),
    ]
)
def legacy_identity_fields(
    request: pytest.FixtureRequest,
) -> dict[str, list[str]] | None:
    """Exercise every legacy trace-diff expectation through both falsey paths."""
    return request.param


# --- Legacy-equivalence scaffolding ---
#
# ``_legacy_fields`` extracts every field a caller could observe from a
# TraceDiff before the ToolDiff schema grew tier/index_a/index_b, so it
# can pin behavioral equivalence across the alignment-integration change
# without demanding byte-identical dataclasses. ``_LEGACY_FIXTURES``
# mirrors every pre-existing StubTracer-based fixture shape in this file
# so each can be replayed under different alignment configurations.


def _legacy_fields(diff: TraceDiff) -> dict[str, Any]:
    return {
        "trace_ids": (diff.trace_a_id, diff.trace_b_id),
        "divergence_count": len(diff.tool_divergence),
        "report_indices": [d.index for d in diff.tool_divergence],
        "kinds": [d.divergence for d in diff.tool_divergence],
        "payloads": [(d.a, d.b) for d in diff.tool_divergence],
        "output_diff": diff.output_diff,
        "error_category_change": diff.error_category_change,
        "score_deltas": diff.score_deltas,
        "score_details": diff.score_details,
        "cost_delta": diff.cost_delta,
        "latency_ms_delta": diff.latency_ms_delta,
        "identical": diff.identical,
        "fully_identical": diff.fully_identical,
    }


def _fixture_identical() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a", output="done"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [_root("b", output="done"), _tool("b", 0, "echo", {"text": "x"}, "x")],
    }), "a", "b"


def _fixture_args_divergence() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "y"}, "y")],
    }), "a", "b"


def _fixture_only_b_trailing() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [
            _root("b"),
            _tool("b", 0, "echo", {"text": "x"}, "x"),
            _tool("b", 1, "echo", {"text": "extra"}, "extra"),
        ],
    }), "a", "b"


def _fixture_only_a_missing() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [
            _root("a"),
            _tool("a", 0, "echo", {"text": "x"}, "x"),
            _tool("a", 1, "echo", {"text": "y"}, "y"),
        ],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "x"}, "x")],
    }), "a", "b"


def _fixture_output_diff() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a", output="alpha")],
        "b": [_root("b", output="beta")],
    }), "a", "b"


def _fixture_error_category_change() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a", error_category="max_llm_calls")],
        "b": [_root("b", error_category=None)],
    }), "a", "b"


def _fixture_score_deltas() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.6}])],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.8}])],
    }), "a", "b"


def _fixture_score_deltas_add_drop() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a"), _grading("a", [
            {"dimension": "quality", "value": 0.7},
            {"dimension": "accuracy", "value": 0.5},
        ])],
        "b": [_root("b"), _grading("b", [
            {"dimension": "quality", "value": 0.7},
            {"dimension": "relevance", "value": 0.9},
        ])],
    }), "a", "b"


def _fixture_cost_latency_drift() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [_root("a", output="done", duration_ms=10.0), _llm("a", 0, cost=0.01)],
        "b": [_root("b", output="done", duration_ms=50.0), _llm("b", 0, cost=0.05)],
    }), "a", "b"


def _fixture_cost_latency_delta() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [
            _root("a", duration_ms=100.0),
            _llm("a", 0, cost=0.01),
            _llm("a", 1, cost=0.02),
        ],
        "b": [
            _root("b", duration_ms=250.0),
            _llm("b", 0, cost=0.05),
        ],
    }), "a", "b"


def _fixture_submit_output_ignored() -> tuple[_StubTracer, str, str]:
    return _StubTracer({
        "a": [
            _root("a"),
            Span(
                id="a-so", trace_id="a", kind=SpanKind.TOOL_CALL, name="submit_output",
                started_at=datetime.now(), input={"result": "x"}, output="{}",
            ),
        ],
        "b": [
            _root("b"),
            Span(
                id="b-so", trace_id="b", kind=SpanKind.TOOL_CALL, name="submit_output",
                started_at=datetime.now(), input={"result": "y"}, output="{}",
            ),
        ],
    }), "a", "b"


_LEGACY_FIXTURES: list[tuple[str, Callable[[], tuple[_StubTracer, str, str]]]] = [
    ("identical", _fixture_identical),
    ("args_divergence", _fixture_args_divergence),
    ("only_b_trailing", _fixture_only_b_trailing),
    ("only_a_missing", _fixture_only_a_missing),
    ("output_diff", _fixture_output_diff),
    ("error_category_change", _fixture_error_category_change),
    ("score_deltas", _fixture_score_deltas),
    ("score_deltas_add_drop", _fixture_score_deltas_add_drop),
    ("cost_latency_drift", _fixture_cost_latency_drift),
    ("cost_latency_delta", _fixture_cost_latency_delta),
    ("submit_output_ignored", _fixture_submit_output_ignored),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,build", _LEGACY_FIXTURES, ids=[n for n, _ in _LEGACY_FIXTURES])
async def test_legacy_fields_helper_is_stable(name, build):
    """Guard for the legacy-equivalence harness: the field-extraction
    helper and fixture table must themselves be deterministic before
    they're used to pin identity_fields=None vs. {} equivalence."""
    tracer, id_a, id_b = build()
    diff1 = await trace_diff(id_a, id_b, tracer=tracer)
    diff2 = await trace_diff(id_a, id_b, tracer=tracer)
    assert _legacy_fields(diff1) == _legacy_fields(diff2)


@pytest.mark.asyncio
async def test_identical_traces_are_empty_diff(legacy_identity_fields):
    spans = [
        _root("a", output="done"),
        _tool("a", 0, "echo", {"text": "x"}, "x"),
    ]
    spans_b = [
        _root("b", output="done"),
        _tool("b", 0, "echo", {"text": "x"}, "x"),
    ]
    tracer = _StubTracer({"a": spans, "b": spans_b})

    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.identical is True
    assert diff.tool_divergence == []
    assert diff.output_diff is None


@pytest.mark.asyncio
async def test_args_divergence_classified(legacy_identity_fields):
    tracer = _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "y"}, "y")],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert len(diff.tool_divergence) == 1
    d = diff.tool_divergence[0]
    assert d.divergence == "args"
    assert d.a.args == {"text": "x"}
    assert d.b.args == {"text": "y"}


@pytest.mark.asyncio
async def test_only_b_trailing_tool(legacy_identity_fields):
    tracer = _StubTracer({
        "a": [_root("a"), _tool("a", 0, "echo", {"text": "x"}, "x")],
        "b": [
            _root("b"),
            _tool("b", 0, "echo", {"text": "x"}, "x"),
            _tool("b", 1, "echo", {"text": "extra"}, "extra"),
        ],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert len(diff.tool_divergence) == 1
    assert diff.tool_divergence[0].divergence == "only_b"
    assert diff.tool_divergence[0].a is None
    assert diff.tool_divergence[0].b.name == "echo"


@pytest.mark.asyncio
async def test_only_a_missing_from_b(legacy_identity_fields):
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _tool("a", 0, "echo", {"text": "x"}, "x"),
            _tool("a", 1, "echo", {"text": "y"}, "y"),
        ],
        "b": [_root("b"), _tool("b", 0, "echo", {"text": "x"}, "x")],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert len(diff.tool_divergence) == 1
    assert diff.tool_divergence[0].divergence == "only_a"


@pytest.mark.asyncio
async def test_output_diff_populated_when_finals_differ(legacy_identity_fields):
    tracer = _StubTracer({
        "a": [_root("a", output="alpha")],
        "b": [_root("b", output="beta")],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.output_diff == ("alpha", "beta")


@pytest.mark.asyncio
async def test_error_category_change_populated(legacy_identity_fields):
    tracer = _StubTracer({
        "a": [_root("a", error_category="max_llm_calls")],
        "b": [_root("b", error_category=None)],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.error_category_change == ("max_llm_calls", None)


@pytest.mark.asyncio
async def test_score_deltas_computed_per_dimension(legacy_identity_fields):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.score_deltas == {"quality": pytest.approx(0.2)}


@pytest.mark.asyncio
async def test_score_deltas_surface_added_and_dropped_dimensions(
    legacy_identity_fields,
):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    # accuracy was dropped (present in A, missing in B → negative delta)
    # relevance was added (missing in A, present in B → positive delta)
    # quality identical → omitted
    assert "quality" not in diff.score_deltas
    assert diff.score_deltas["accuracy"] == pytest.approx(-0.5)
    assert diff.score_deltas["relevance"] == pytest.approx(0.9)
    assert diff.identical is False


@pytest.mark.asyncio
async def test_fully_identical_requires_cost_and_latency_match(
    legacy_identity_fields,
):
    """``identical`` tolerates cost/latency drift; ``fully_identical`` doesn't.

    Same-behavior traces with different spend should report
    ``identical=True`` (behavioral match, the model-swap use case) but
    ``fully_identical=False`` (stricter full-equality sibling).
    """
    tracer = _StubTracer({
        "a": [_root("a", output="done", duration_ms=10.0), _llm("a", 0, cost=0.01)],
        "b": [_root("b", output="done", duration_ms=50.0), _llm("b", 0, cost=0.05)],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.identical is True
    assert diff.fully_identical is False
    assert diff.cost_delta == pytest.approx(0.04)
    assert diff.latency_ms_delta == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_fully_identical_true_on_byte_for_byte_match(
    legacy_identity_fields,
):
    tracer = _StubTracer({
        "a": [_root("a", output="done", duration_ms=10.0)],
        "b": [_root("b", output="done", duration_ms=10.0)],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.identical is True
    assert diff.fully_identical is True


@pytest.mark.asyncio
async def test_cost_and_latency_delta(legacy_identity_fields):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.cost_delta == pytest.approx(0.02)
    assert diff.latency_ms_delta == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_submit_output_spans_ignored_in_diff(legacy_identity_fields):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.tool_divergence == []


# --- SQLite integration ---


@pytest.mark.asyncio
async def test_missing_trace_ids_raise(legacy_identity_fields):
    tracer = _StubTracer({})
    with pytest.raises(ValueError, match="not found or has no spans"):
        await trace_diff(
            "missing-a",
            "missing-b",
            tracer=tracer,
            identity_fields=legacy_identity_fields,
        )


@pytest.mark.asyncio
async def test_trace_with_no_agent_run_root_raises(legacy_identity_fields):
    # Trace has spans but none is the AGENT_RUN root
    tracer = _StubTracer({
        "a": [_root("a")],
        "b": [_tool("b", 0, "echo", {"text": "x"}, "x")],  # no root!
    })
    with pytest.raises(ValueError, match="no AGENT_RUN root span"):
        await trace_diff(
            "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
        )


@pytest.mark.asyncio
async def test_trace_diff_against_sqlite_backend(
    tmp_path: Any, legacy_identity_fields
):
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

    diff: TraceDiff = await trace_diff(
        t_a, t_b, tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.tool_divergence == []
    assert diff.output_diff == ("done_a", "done_b")


@pytest.mark.asyncio
async def test_canonical_grading_span_writer_contract(
    tmp_path: Any, legacy_identity_fields
):
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

    diff: TraceDiff = await trace_diff(
        t_a, t_b, tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.score_deltas == {"quality": pytest.approx(0.3)}
    assert diff.score_details["quality"] == (pytest.approx(0.6), pytest.approx(0.9))
    assert diff.identical is False


# --- Legacy score format support ---


@pytest.mark.asyncio
async def test_legacy_dim_val_scores_are_read(legacy_identity_fields):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.score_deltas == {"quality": pytest.approx(0.4)}
    assert diff.identical is False


@pytest.mark.asyncio
async def test_canonical_and_legacy_scores_mixed_across_traces(
    legacy_identity_fields,
):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.score_deltas == {"relevance": pytest.approx(0.2)}
    assert diff.identical is False


@pytest.mark.asyncio
async def test_score_details_exposes_per_dimension_old_and_new_values(
    legacy_identity_fields,
):
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
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    # quality: changed
    assert diff.score_details["quality"] == (pytest.approx(0.6), pytest.approx(0.9))
    # accuracy: dropped (only in A)
    assert diff.score_details["accuracy"] == (pytest.approx(0.8), None)
    # relevance: added (only in B)
    assert diff.score_details["relevance"] == (None, pytest.approx(0.7))


@pytest.mark.asyncio
async def test_score_details_identical_dimension_has_matching_values(
    legacy_identity_fields,
):
    """A dimension with the same score in both traces must appear in score_details
    but not in score_deltas."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.7}])],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.7}])],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert "quality" not in diff.score_deltas
    assert diff.score_details["quality"] == (pytest.approx(0.7), pytest.approx(0.7))
    assert diff.identical is True


@pytest.mark.asyncio
async def test_score_delta_zero_value_does_not_lose_dimension(
    legacy_identity_fields,
):
    """A score of 0.0 must not be confused with an absent dimension (0.0 is falsy)."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.0}])],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.5}])],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert diff.score_deltas == {"quality": pytest.approx(0.5)}
    assert diff.score_details["quality"] == (pytest.approx(0.0), pytest.approx(0.5))


@pytest.mark.asyncio
async def test_added_dimension_with_zero_score_makes_identical_false(
    legacy_identity_fields,
):
    """A dimension added in B with score 0.0 must still appear in score_deltas and
    make identical=False — the numeric delta is 0.0 but the rubric changed."""
    tracer = _StubTracer({
        "a": [_root("a")],
        "b": [_root("b"), _grading("b", [{"dimension": "quality", "value": 0.0}])],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
    assert "quality" in diff.score_deltas
    assert diff.score_deltas["quality"] == pytest.approx(0.0)
    assert diff.score_details["quality"] == (None, pytest.approx(0.0))
    assert diff.identical is False


@pytest.mark.asyncio
async def test_dropped_dimension_with_zero_score_makes_identical_false(
    legacy_identity_fields,
):
    """A dimension dropped from A where its score was 0.0 must still appear in
    score_deltas and make identical=False."""
    tracer = _StubTracer({
        "a": [_root("a"), _grading("a", [{"dimension": "quality", "value": 0.0}])],
        "b": [_root("b")],
    })
    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=legacy_identity_fields
    )
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
async def test_score_deltas_from_real_runner_grading_spans(
    tmp_path: Any, legacy_identity_fields
):
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

    diff = await trace_diff(
        result_a.trace_id,
        result_b.trace_id,
        tracer=tracer,
        identity_fields=legacy_identity_fields,
    )

    assert "quality" in diff.score_deltas
    assert diff.score_deltas["quality"] == pytest.approx(-0.6, abs=1e-6)
    assert diff.identical is False


# --- Identity alignment integration ---


def _mixed_tier_events() -> tuple[list[ToolEvent], list[ToolEvent]]:
    return [
        ToolEvent("write", {"id": 1, "val": "old"}, "wrote", None),
        ToolEvent("search", {"q": "A"}, "resultA", None),
        ToolEvent("delete", {"id": 9}, "deleted", None),
        ToolEvent("search", {"q": "B"}, "resultB", None),
        ToolEvent("notify", {"msg": "done"}, "ok", None),
        ToolEvent("write", {"id": 2, "val": "fresh"}, "wrote2", None),
    ], [
        ToolEvent("delete", {"id": 9}, "deleted", None),
        ToolEvent("write", {"id": 1, "val": "new"}, "wrote", None),
        ToolEvent("search", {"q": "A"}, "resultA", None),
        ToolEvent("search", {"q": "B"}, "resultB", None),
        ToolEvent("audit", {"msg": "inserted"}, "logged", None),
        ToolEvent("notify", {"msg": "done"}, "ok", None),
    ]


def _mixed_tier_tracer(
) -> tuple[_StubTracer, str, str, list[ToolEvent], list[ToolEvent]]:
    a_events, b_events = _mixed_tier_events()

    def spans(trace_id: str, events: list[ToolEvent]) -> list[Span]:
        result = [_root(trace_id)]
        for index, event in enumerate(events):
            assert isinstance(event.args, dict)
            assert isinstance(event.output, str)
            result.append(
                _tool(
                    trace_id,
                    index,
                    event.name,
                    event.args,
                    event.output,
                    event.error,
                )
            )
        return result

    tracer = _StubTracer({"a": spans("a", a_events), "b": spans("b", b_events)})
    return tracer, "a", "b", a_events, b_events


@pytest.mark.asyncio
async def test_mixed_tier_alignment_pins_full_partition():
    """A single fixture with two declared tools, a repeated undeclared
    search, a unique-both undeclared notify, an insertion, a deletion,
    a same-entity argument change, and reordered identity matches —
    pinning the intermediate identity/anchor/ordinal partition and the
    resulting public divergence list without conflating the tiers."""
    tracer, id_a, id_b, a_events, b_events = _mixed_tier_tracer()
    identity_fields = {"write": ["id"], "delete": ["id"]}

    alignment = IdentityAligner().align(
        a_events, b_events, identity_fields=identity_fields
    )
    alignment.validate(a_events, b_events, identity_fields=identity_fields)
    assert alignment.pairs == [
        AlignedPair(0, 1, "identity"),
        AlignedPair(1, 2, "ordinal"),
        AlignedPair(2, 0, "identity"),
        AlignedPair(3, 3, "ordinal"),
        AlignedPair(4, 5, "anchor"),
    ]
    assert alignment.only_a == [UnmatchedEvent(5, "identity")]
    assert alignment.only_b == [UnmatchedEvent(4, "ordinal")]

    diff = await trace_diff(id_a, id_b, tracer=tracer, identity_fields=identity_fields)

    assert len(diff.tool_divergence) == 3

    d0 = diff.tool_divergence[0]
    assert d0.index == 0
    assert d0.divergence == "args"
    assert d0.tier == "identity"
    assert d0.index_a == 0
    assert d0.index_b == 1
    assert d0.a.args == {"id": 1, "val": "old"}
    assert d0.b.args == {"id": 1, "val": "new"}

    d1 = diff.tool_divergence[1]
    assert d1.index == 1
    assert d1.divergence == "only_a"
    assert d1.tier == "identity"
    assert d1.index_a == 5
    assert d1.index_b is None
    assert d1.a.name == "write"
    assert d1.b is None

    d2 = diff.tool_divergence[2]
    assert d2.index == 2
    assert d2.divergence == "only_b"
    assert d2.tier == "ordinal"
    assert d2.index_a is None
    assert d2.index_b == 4
    assert d2.b.name == "audit"
    assert d2.a is None


def _insertion_cascade_tracer() -> tuple[_StubTracer, str, str]:
    a_spans = [
        _root("a"),
        _tool("a", 0, "toolX", {"id": 1}, "outX"),
        _tool("a", 1, "toolY", {"q": "y"}, "outY"),
        _tool("a", 2, "toolZ", {"q": "z"}, "outZ"),
    ]
    b_spans = [
        _root("b"),
        _tool("b", 0, "new_tool", {"q": "new"}, "outNew"),
        _tool("b", 1, "toolX", {"id": 1}, "outX"),
        _tool("b", 2, "toolY", {"q": "y"}, "outY"),
        _tool("b", 3, "toolZ", {"q": "z"}, "outZ"),
    ]
    return _StubTracer({"a": a_spans, "b": b_spans}), "a", "b"


@pytest.mark.asyncio
async def test_insertion_cascade_prevented_when_fully_anchored():
    """When every unchanged call is identity-keyed or a unique-both
    anchor, a single new insertion produces exactly one ordinal-tier
    only_b and no paired divergences — cascade prevention only in the
    domain where identities/anchors can claim every unchanged call."""
    tracer, id_a, id_b = _insertion_cascade_tracer()

    diff = await trace_diff(id_a, id_b, tracer=tracer, identity_fields={"toolX": ["id"]})

    assert len(diff.tool_divergence) == 1
    d = diff.tool_divergence[0]
    assert d.index == 0
    assert d.divergence == "only_b"
    assert d.tier == "ordinal"
    assert d.index_a is None
    assert d.index_b == 0
    assert d.b.name == "new_tool"


def _ambiguity_tracer() -> tuple[_StubTracer, str, str]:
    a_spans = [
        _root("a"),
        _tool("a", 0, "search", {"q": "q1"}, "r1"),
        _tool("a", 1, "search", {"q": "q2"}, "r2"),
    ]
    b_spans = [
        _root("b"),
        _tool("b", 0, "search", {"q": "new"}, "rnew"),
        _tool("b", 1, "search", {"q": "q1"}, "r1"),
        _tool("b", 2, "search", {"q": "q2"}, "r2"),
    ]
    return _StubTracer({"a": a_spans, "b": b_spans}), "a", "b"


@pytest.mark.asyncio
async def test_ambiguous_repeated_unkeyed_calls_stay_positional():
    """Repeated unkeyed calls have no anchor and remain positionally
    ambiguous within their segment even with IdentityAligner enabled —
    guards against later fuzzy matching being mistaken for a bug fix."""
    tracer, id_a, id_b = _ambiguity_tracer()

    diff = await trace_diff(id_a, id_b, tracer=tracer, identity_fields={"other_tool": ["id"]})

    assert len(diff.tool_divergence) == 3

    d0 = diff.tool_divergence[0]
    assert d0.divergence == "args"
    assert d0.tier == "ordinal"
    assert d0.index_a == 0
    assert d0.index_b == 0
    assert d0.a.args == {"q": "q1"}
    assert d0.b.args == {"q": "new"}

    d1 = diff.tool_divergence[1]
    assert d1.divergence == "args"
    assert d1.tier == "ordinal"
    assert d1.index_a == 1
    assert d1.index_b == 1
    assert d1.a.args == {"q": "q2"}
    assert d1.b.args == {"q": "q1"}

    d2 = diff.tool_divergence[2]
    assert d2.divergence == "only_b"
    assert d2.tier == "ordinal"
    assert d2.index_a is None
    assert d2.index_b == 2
    assert d2.b.args == {"q": "q2"}


def _reordered_identity_tracer() -> tuple[_StubTracer, str, str]:
    a_spans = [
        _root("a", output="done"),
        _tool("a", 0, "write", {"id": 1}, "wrote1"),
        _tool("a", 1, "write", {"id": 2}, "wrote2"),
    ]
    b_spans = [
        _root("b", output="done"),
        _tool("b", 0, "write", {"id": 2}, "wrote2"),
        _tool("b", 1, "write", {"id": 1}, "wrote1"),
    ]
    return _StubTracer({"a": a_spans, "b": b_spans}), "a", "b"


@pytest.mark.asyncio
async def test_reordered_identity_calls_are_not_divergent():
    """Central semantic change: equal keyed calls in opposite orders
    with otherwise-equal traces produce zero tool divergences and
    identical=True, directly disproving exact sequence equality as an
    implication of `identical` — see the order-insensitivity docs on
    TraceDiff.identical."""
    tracer, id_a, id_b = _reordered_identity_tracer()

    diff = await trace_diff(id_a, id_b, tracer=tracer, identity_fields={"write": ["id"]})

    assert diff.tool_divergence == []
    assert diff.identical is True


def _a_centric_ordering_tracer() -> tuple[_StubTracer, str, str]:
    a_spans = [
        _root("a"),
        _tool("a", 0, "write", {"id": 1, "val": "old"}, "w1"),
        _tool("a", 1, "delete", {"id": 9}, "d9"),
        _tool("a", 2, "write", {"id": 2, "val": "old2"}, "w2"),
    ]
    b_spans = [
        _root("b"),
        _tool("b", 0, "write", {"id": 1, "val": "new"}, "w1"),
        _tool("b", 1, "write", {"id": 2, "val": "new2"}, "w2"),
        _tool("b", 2, "notify", {"msg": "extra"}, "logged"),
    ]
    return _StubTracer({"a": a_spans, "b": b_spans}), "a", "b"


@pytest.mark.asyncio
async def test_a_centric_report_order_interleaves_only_a():
    """A divergent pair, an interleaved only_a, a later divergent pair,
    and a trailing only_b — pins A-centric sorting independently of the
    mixed-tier fixture's particular event arrangement."""
    tracer, id_a, id_b = _a_centric_ordering_tracer()
    identity_fields = {"write": ["id"], "delete": ["id"]}

    diff = await trace_diff(id_a, id_b, tracer=tracer, identity_fields=identity_fields)

    assert len(diff.tool_divergence) == 4

    d0 = diff.tool_divergence[0]
    assert (d0.index, d0.divergence, d0.tier, d0.index_a, d0.index_b) == (0, "args", "identity", 0, 0)

    d1 = diff.tool_divergence[1]
    assert (d1.index, d1.divergence, d1.tier, d1.index_a, d1.index_b) == (1, "only_a", "identity", 1, None)
    assert d1.a.name == "delete"

    d2 = diff.tool_divergence[2]
    assert (d2.index, d2.divergence, d2.tier, d2.index_a, d2.index_b) == (2, "args", "identity", 2, 1)

    d3 = diff.tool_divergence[3]
    assert (d3.index, d3.divergence, d3.tier, d3.index_a, d3.index_b) == (3, "only_b", "ordinal", None, 2)
    assert d3.b.name == "notify"


@pytest.mark.asyncio
async def test_submit_output_excluded_under_identity_alignment():
    """The existing submit_output-exclusion contract must hold with
    identity alignment enabled too — excluded spans must not leak into
    identities, anchors, source indices, or divergences."""
    tracer = _StubTracer({
        "a": [
            _root("a"),
            _tool("a", 0, "write", {"id": 1}, "wrote"),
            Span(
                id="a-so", trace_id="a", kind=SpanKind.TOOL_CALL, name="submit_output",
                started_at=datetime.now(), input={"result": "x"}, output="{}",
            ),
        ],
        "b": [
            _root("b"),
            _tool("b", 0, "write", {"id": 1}, "wrote"),
            Span(
                id="b-so", trace_id="b", kind=SpanKind.TOOL_CALL, name="submit_output",
                started_at=datetime.now(), input={"result": "y"}, output="{}",
            ),
        ],
    })

    diff = await trace_diff("a", "b", tracer=tracer, identity_fields={"write": ["id"]})

    assert diff.tool_divergence == []
    assert diff.identical is True


# --- identity_fields validation ---


class _ExplodingTracer(TracingLogger):
    """Tracer that records/fails on access, to prove validation runs
    before any tracer read."""

    def __init__(self) -> None:
        self.get_trace_calls = 0

    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):  # pragma: no cover
        raise NotImplementedError

    def start_span(self, parent_id, kind, name, input=None, metadata=None):  # noqa: A002  # pragma: no cover
        raise NotImplementedError

    def end_span(self, span_id, output=None, error=None, usage=None):  # pragma: no cover
        raise NotImplementedError

    async def get_trace(self, trace_id: str) -> list[Span]:
        self.get_trace_calls += 1
        raise AssertionError("get_trace must not be called when identity_fields is invalid")

    async def list_traces(self, since=None, limit=50, name=None):  # pragma: no cover
        return []

    async def flush(self) -> None:  # pragma: no cover
        pass


class _MutatingIdentityFieldsTracer(_StubTracer):
    """Mutate caller-owned identity configuration during the first trace read."""

    def __init__(
        self,
        traces: dict[str, list[Span]],
        identity_fields: dict[str, list[str]],
    ) -> None:
        super().__init__(traces)
        self.identity_fields = identity_fields
        self.original_paths = identity_fields["write"]
        self.get_trace_calls = 0

    async def get_trace(self, trace_id: str) -> list[Span]:
        self.get_trace_calls += 1
        spans = await super().get_trace(trace_id)
        if self.get_trace_calls == 1:
            self.identity_fields.clear()
            self.original_paths[:] = ["missing"]
        return spans


_INVALID_IDENTITY_FIELDS: list[tuple[str, Any]] = [
    ("not_a_dict", ["write", "id"]),
    ("non_string_tool_name", {123: ["id"]}),
    ("non_list_declaration", {"write": "id"}),
    ("empty_list", {"write": []}),
    ("non_string_path", {"write": [123]}),
    ("empty_path", {"write": [""]}),
    ("leading_empty_dot_segment", {"write": [".id"]}),
    ("trailing_empty_dot_segment", {"write": ["id."]}),
    ("interior_empty_dot_segment", {"write": ["work..id"]}),
    ("duplicate_path", {"write": ["id", "id"]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,bad_fields", _INVALID_IDENTITY_FIELDS, ids=[n for n, _ in _INVALID_IDENTITY_FIELDS]
)
async def test_invalid_identity_fields_raises_before_any_tracer_read(name, bad_fields):
    tracer = _ExplodingTracer()

    with pytest.raises(ValueError, match="identity_fields"):
        await trace_diff("a", "b", tracer=tracer, identity_fields=bad_fields)

    assert tracer.get_trace_calls == 0


@pytest.mark.asyncio
async def test_identity_fields_are_snapshotted_before_trace_reads():
    identity_fields = {"write": ["id"]}
    tracer = _MutatingIdentityFieldsTracer(
        {
            "a": [
                _root("a", output="done"),
                _tool("a", 0, "write", {"id": 1}, "wrote1"),
                _tool("a", 1, "write", {"id": 2}, "wrote2"),
            ],
            "b": [
                _root("b", output="done"),
                _tool("b", 0, "write", {"id": 2}, "wrote2"),
                _tool("b", 1, "write", {"id": 1}, "wrote1"),
            ],
        },
        identity_fields,
    )

    diff = await trace_diff(
        "a", "b", tracer=tracer, identity_fields=identity_fields
    )

    assert identity_fields == {}
    assert tracer.original_paths == ["missing"]
    assert diff.tool_divergence == []
    assert diff.identical is True


# --- Full legacy-equivalence harness ---


@pytest.mark.asyncio
@pytest.mark.parametrize("name,build", _LEGACY_FIXTURES, ids=[n for n, _ in _LEGACY_FIXTURES])
async def test_legacy_equivalence_none_vs_empty_identity_fields(name, build):
    """identity_fields=None and identity_fields={} must produce
    field-for-field identical behavior across every pre-existing
    trace-diff fixture shape — serialized byte equality is impossible
    after the ToolDiff schema extension, so this pins every
    legacy-observable field plus the new ordinal metadata and correct
    source indices."""
    tracer, id_a, id_b = build()

    diff_none = await trace_diff(id_a, id_b, tracer=tracer, identity_fields=None)
    diff_empty = await trace_diff(id_a, id_b, tracer=tracer, identity_fields={})

    assert _legacy_fields(diff_none) == _legacy_fields(diff_empty)

    for diff in (diff_none, diff_empty):
        assert [d.index for d in diff.tool_divergence] == list(
            range(len(diff.tool_divergence))
        )
        for d in diff.tool_divergence:
            assert d.tier == "ordinal"
            if d.divergence == "only_a":
                assert isinstance(d.index_a, int)
                assert d.index_b is None
            elif d.divergence == "only_b":
                assert d.index_a is None
                assert isinstance(d.index_b, int)
            else:
                assert isinstance(d.index_a, int)
                assert isinstance(d.index_b, int)
                assert d.index_a == d.index_b
