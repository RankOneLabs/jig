"""Tests for TrajectoryGrader and the helper assertions.

Includes an end-to-end test through ``run_agent`` with a real
``SQLiteTracer`` to verify the mid-run flush makes spans visible
to the grader (the buffered-tracer regression flagged in the spec).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from jig import (
    AgentConfig,
    CompletionParams,
    LLMResponse,
    Span,
    SpanKind,
    Tool,
    ToolCall,
    ToolDefinition,
    TrajectoryAssertion,
    TrajectoryGrader,
    Usage,
    run_agent,
    step_budget,
    tool_called,
    tool_sequence,
)
from jig.core.types import (
    EvalCase,
    FeedbackLoop,
    LLMClient,
    MemoryEntry,
    MemoryStore,
    Message,
    Retriever,
    Score,
    ScoredResult,
    ScoreSource,
    TracingLogger,
)
from jig.feedback import (
    CompositeGrader,
    HeuristicGrader,
    Check,
)
from jig.tools import ToolRegistry
from jig.tracing import SQLiteTracer


# --- Fakes (minimal, mirrors test_core.py) ---


class FakeLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, params: CompletionParams) -> LLMResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class FakeMemory(MemoryStore, Retriever):
    def __init__(self) -> None:
        self.stored: list[tuple[str, dict]] = []
        self.sessions: dict[str, list[Message]] = {}

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        self.stored.append((content, metadata or {}))
        return f"mem-{len(self.stored)}"

    async def get(self, id: str) -> MemoryEntry | None:
        return None

    async def all(self) -> list[MemoryEntry]:
        return []

    async def delete(self, id: str) -> None:
        pass

    async def retrieve(
        self, query: str, k: int = 5, context: dict[str, Any] | None = None
    ) -> list[MemoryEntry]:
        return []

    async def get_session(self, session_id: str) -> list[Message]:
        return self.sessions.get(session_id, [])

    async def add_to_session(self, session_id: str, message: Message) -> None:
        self.sessions.setdefault(session_id, []).append(message)

    async def clear(
        self, session_id: str | None = None, before: datetime | None = None
    ) -> None:
        pass


class FakeFeedback(FeedbackLoop):
    def __init__(self) -> None:
        self.scored: list[tuple[str, list[Score]]] = []

    async def store_result(self, content, input_text, metadata=None):
        return "r-0"

    async def score(self, result_id: str, scores: list[Score]) -> None:
        self.scored.append((result_id, scores))

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        return []

    async def query(self, q):
        return []

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        return []


class EchoTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echoes input",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return args.get("text", "")


def _span(kind: SpanKind, name: str) -> Span:
    return Span(
        id=f"s-{name}",
        trace_id="t-0",
        kind=kind,
        name=name,
        started_at=datetime.now(),
    )


class StubTracer(TracingLogger):
    """Returns a fixed span list from get_trace; ignores writes.

    Used for unit tests of the helpers and grader logic where we
    want to control the span list directly.
    """

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def start_trace(
        self, name: str, metadata=None, kind: SpanKind = SpanKind.AGENT_RUN
    ) -> Span:
        return _span(kind, name)

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input=None, metadata=None) -> Span:
        return _span(kind, name)

    def end_span(self, span_id, output=None, error=None, usage=None) -> None:
        return None

    async def get_trace(self, trace_id: str) -> list[Span]:
        return list(self._spans)

    async def list_traces(self, since=None, limit: int = 50, name=None) -> list[Span]:
        return []


# --- Helper-function tests ---


async def test_tool_called_assertion_passes_and_fails():
    spans = [
        _span(SpanKind.TOOL_CALL, "echo"),
        _span(SpanKind.LLM_CALL, "complete"),
    ]
    assert tool_called("echo")(spans) == 1.0
    assert tool_called("missing")(spans) == 0.0
    # Wrong kind should not match
    assert tool_called("complete")(spans) == 0.0


async def test_tool_sequence_partial_match():
    spans = [
        _span(SpanKind.TOOL_CALL, "a"),
        _span(SpanKind.TOOL_CALL, "b"),
        _span(SpanKind.TOOL_CALL, "x"),
    ]
    # Full prefix match
    assert tool_sequence(["a", "b", "x"])(spans) == 1.0
    # Two of three match before divergence
    assert tool_sequence(["a", "b", "c"])(spans) == pytest.approx(2 / 3)
    # First diverges immediately
    assert tool_sequence(["z"])(spans) == 0.0
    # Empty expectation is vacuously satisfied
    assert tool_sequence([])(spans) == 1.0


async def test_step_budget_under_and_over():
    spans = [
        _span(SpanKind.LLM_CALL, "c1"),
        _span(SpanKind.LLM_CALL, "c2"),
        _span(SpanKind.TOOL_CALL, "echo"),
    ]
    assert step_budget(2)(spans) == 1.0
    assert step_budget(3)(spans) == 1.0
    assert step_budget(1)(spans) == 0.0


# --- Grader logic tests ---


async def test_grader_handles_missing_trace_id_gracefully():
    grader = TrajectoryGrader(
        StubTracer([]),
        [TrajectoryAssertion(name="called_echo", check=tool_called("echo"))],
    )
    scores = await grader.grade("input", "output", context=None)
    assert len(scores) == 1
    assert scores[0].dimension == "called_echo"
    assert scores[0].value == 0.0
    assert scores[0].source == ScoreSource.HEURISTIC

    scores = await grader.grade("input", "output", context={})
    assert scores[0].value == 0.0


async def test_grader_handles_non_string_trace_id():
    """A non-string trace_id (e.g. accidentally passing a Span obj)
    should fail soft to 0.0 rather than crashing the sweep.
    """
    grader = TrajectoryGrader(
        StubTracer([]),
        [TrajectoryAssertion(name="x", check=tool_called("echo"))],
    )
    scores = await grader.grade("i", "o", context={"trace_id": 12345})
    assert scores[0].value == 0.0


async def test_grader_handles_tracer_get_trace_exception():
    """Tracer.get_trace raising (e.g. StdoutTracer's
    NotImplementedError, closed DB) should fail soft for every
    assertion rather than propagate.
    """

    class BrokenTracer(StubTracer):
        async def get_trace(self, trace_id: str) -> list[Span]:
            raise NotImplementedError("not supported")

    grader = TrajectoryGrader(
        BrokenTracer([]),
        [
            TrajectoryAssertion(name="a", check=tool_called("echo")),
            TrajectoryAssertion(name="b", check=step_budget(5)),
        ],
    )
    scores = await grader.grade("i", "o", context={"trace_id": "t-0"})
    assert {s.dimension: s.value for s in scores} == {"a": 0.0, "b": 0.0}
    # All scores should still be HEURISTIC source — fail-soft doesn't
    # change provenance.
    assert all(s.source == ScoreSource.HEURISTIC for s in scores)


async def test_grader_handles_assertion_exception():
    def raises(spans: list[Span]) -> float:
        raise RuntimeError("boom")

    spans = [_span(SpanKind.TOOL_CALL, "echo")]
    grader = TrajectoryGrader(
        StubTracer(spans),
        [
            TrajectoryAssertion(name="ok", check=tool_called("echo")),
            TrajectoryAssertion(name="bad", check=raises),
        ],
    )
    scores = await grader.grade("i", "o", context={"trace_id": "t-0"})
    assert {s.dimension: s.value for s in scores} == {"ok": 1.0, "bad": 0.0}


async def test_grader_clamps_check_value_to_unit_interval():
    spans = [_span(SpanKind.TOOL_CALL, "echo")]
    grader = TrajectoryGrader(
        StubTracer(spans),
        [
            TrajectoryAssertion(name="too_high", check=lambda s: 5.0),
            TrajectoryAssertion(name="too_low", check=lambda s: -1.0),
        ],
    )
    scores = await grader.grade("i", "o", context={"trace_id": "t-0"})
    by_dim = {s.dimension: s.value for s in scores}
    assert by_dim == {"too_high": 1.0, "too_low": 0.0}


# --- Integration test (the spec-critical one) ---


@pytest.mark.asyncio
async def test_grader_in_run_agent_pipeline_with_sqlite_tracer(tmp_path):
    """End-to-end: TrajectoryGrader sees tool-call spans through a
    buffered SQLiteTracer thanks to the runner's mid-run flush.

    Without the flush, get_trace returns an empty list and the
    assertion silently scores 0.0. Regressing the flush would flip
    this test red.
    """
    tracer = SQLiteTracer(db_path=str(tmp_path / "trace.db"))

    # LLM: turn 1 calls echo, turn 2 returns final text.
    llm = FakeLLM([
        LLMResponse(
            content="thinking",
            tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "hi"})],
            usage=Usage(5, 5),
            latency_ms=10,
            model="fake",
        ),
        LLMResponse(
            content="done",
            tool_calls=None,
            usage=Usage(5, 5),
            latency_ms=10,
            model="fake",
        ),
    ])

    grader = TrajectoryGrader(
        tracer,
        [
            TrajectoryAssertion(name="called_echo", check=tool_called("echo")),
            TrajectoryAssertion(name="under_3_steps", check=step_budget(3)),
            TrajectoryAssertion(
                name="echo_then_finish",
                check=tool_sequence(["echo"]),
            ),
        ],
    )

    feedback = FakeFeedback()
    result = await run_agent(
        AgentConfig(
            name="trajectory_test",
            description="exercise trajectory grading",
            system_prompt="You are a test agent.",
            llm=llm,
            store=FakeMemory(),
            retriever=None,
            feedback=feedback,
            tracer=tracer,
            tools=ToolRegistry([EchoTool()]),
            grader=grader,
        ),
        "say hi",
    )

    assert result.output == "done"
    assert result.scores is not None
    by_dim = {s.dimension: s.value for s in result.scores}
    assert by_dim == {
        "called_echo": 1.0,
        "under_3_steps": 1.0,
        "echo_then_finish": 1.0,
    }

    # The feedback loop should also have received the trajectory scores
    assert len(feedback.scored) == 1
    persisted_dims = {s.dimension for s in feedback.scored[0][1]}
    assert persisted_dims == {"called_echo", "under_3_steps", "echo_then_finish"}

    # And the trace itself should be properly closed (root span ended)
    # — verifies the unended-span retention in flush() didn't break
    # the finalize path.
    await tracer.close()
    spans = await tracer.get_trace(result.trace_id)
    root = next(s for s in spans if s.kind == SpanKind.AGENT_RUN)
    assert root.ended_at is not None
    assert root.duration_ms is not None


@pytest.mark.asyncio
async def test_composite_with_trajectory_and_heuristic(tmp_path):
    """A CompositeGrader running a TrajectoryGrader alongside a
    HeuristicGrader should produce scores from both — the trajectory
    grader still sees spans through the runner's flush.
    """
    tracer = SQLiteTracer(db_path=str(tmp_path / "trace.db"))

    llm = FakeLLM([
        LLMResponse(
            content="answer with [1] citation",
            tool_calls=None,
            usage=Usage(5, 5),
            latency_ms=10,
            model="fake",
        ),
    ])

    grader = CompositeGrader([
        HeuristicGrader([Check(name="cites_source", pattern=r"\[\d+\]")]),
        TrajectoryGrader(
            tracer,
            [TrajectoryAssertion(name="under_2_steps", check=step_budget(2))],
        ),
    ])

    result = await run_agent(
        AgentConfig(
            name="composite_test",
            description="composite grader",
            system_prompt="You are a test agent.",
            llm=llm,
            store=FakeMemory(),
            retriever=None,
            feedback=FakeFeedback(),
            tracer=tracer,
            tools=ToolRegistry(),
            grader=grader,
        ),
        "give me an answer",
    )

    by_dim = {s.dimension: s.value for s in (result.scores or [])}
    assert by_dim == {"cites_source": 1.0, "under_2_steps": 1.0}
