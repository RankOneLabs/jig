from __future__ import annotations

from datetime import datetime
from typing import Any

from jig import (
    PipelineConfig,
    Score,
    ScoreSource,
    SpanKind,
    Step,
    map_pipeline,
    run_pipeline,
)
from jig.core.types import EvalCase, FeedbackLoop, Grader, ScoredResult, Span, TracingLogger


# --- Fakes ---


class FakeTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"span-{self._counter}"

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        sid = self._next_id()
        tid = f"trace-{sid}"
        s = Span(
            id=sid,
            trace_id=tid,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            metadata=metadata,
        )
        self.spans.append(s)
        return s

    def start_span(
        self, parent_id: str, kind: SpanKind, name: str, input: Any = None
    ) -> Span:
        parent = next((s for s in self.spans if s.id == parent_id), None)
        trace_id = parent.trace_id if parent else parent_id
        sid = self._next_id()
        s = Span(
            id=sid,
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            parent_id=parent_id,
            input=input,
        )
        self.spans.append(s)
        return s

    def end_span(
        self, span_id: str, output: Any = None, error: str | None = None, usage: Any = None
    ) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.ended_at = datetime.now()
                s.output = output
                s.error = error
                s.usage = usage

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]


class FakeFeedback(FeedbackLoop):
    def __init__(self) -> None:
        self.scored: list[tuple[str, list[Score]]] = []

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

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        return []


class FixedGrader(Grader):
    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    async def grade(
        self, input: Any, output: Any, context: dict[str, Any] | None = None
    ) -> list[Score]:
        return [Score(dimension="quality", value=self._value, source=ScoreSource.HEURISTIC)]


# --- Step functions ---


async def add_one(ctx: dict[str, Any]) -> int:
    return ctx["input"] + 1


async def double(ctx: dict[str, Any]) -> int:
    return ctx["add_one"] * 2


async def to_string(ctx: dict[str, Any]) -> str:
    return str(ctx["double"])


# --- Tests ---


async def test_linear_pipeline():
    """3 steps, verify step_outputs and trace spans."""
    tracer = FakeTracer()
    result = await run_pipeline(
        PipelineConfig(
            name="linear",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(name="double", fn=double),
                Step(name="to_string", fn=to_string),
            ],
            tracer=tracer,
        ),
        input=5,
    )
    assert result.output == "12"
    assert result.step_outputs == {"add_one": 6, "double": 12, "to_string": "12"}
    assert not result.short_circuited
    assert result.error_step is None

    # Verify trace: 1 pipeline_run + 3 pipeline_step
    kinds = [s.kind for s in tracer.spans]
    assert kinds.count(SpanKind.PIPELINE_RUN) == 1
    assert kinds.count(SpanKind.PIPELINE_STEP) == 3


async def test_short_circuit_on_err():
    """Step 2 returns error, step 3 never runs."""

    async def failing_step(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"error": "something broke"}

    async def should_not_run(ctx: dict[str, Any]) -> str:
        raise AssertionError("This step should not execute")

    tracer = FakeTracer()
    result = await run_pipeline(
        PipelineConfig(
            name="short_circuit",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(name="fail", fn=failing_step),
                Step(name="unreachable", fn=should_not_run),
            ],
            tracer=tracer,
            is_err=lambda r: isinstance(r, dict) and "error" in r,
            extract_err=lambda r: r["error"],
        ),
        input=5,
    )
    assert result.short_circuited
    assert result.error_step == "fail"
    assert "unreachable" not in result.step_outputs
    assert result.step_outputs["add_one"] == 6
    assert result.step_outputs["fail"] == {"error": "something broke"}


async def test_skip_when_conditional():
    """Step skipped via skip_when, verify not in step_outputs."""
    tracer = FakeTracer()
    result = await run_pipeline(
        PipelineConfig(
            name="skip_test",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(
                    name="double",
                    fn=double,
                    skip_when=lambda ctx: ctx["add_one"] > 100,
                ),
                Step(name="to_string", fn=to_string),
            ],
            tracer=tracer,
        ),
        input=5,
    )
    # add_one returns 6, which is not > 100, so double runs
    assert "double" in result.step_outputs

    # Now with input that triggers skip
    tracer2 = FakeTracer()

    async def fallback_to_string(ctx: dict[str, Any]) -> str:
        # double was skipped, so use add_one directly
        return str(ctx["add_one"])

    result2 = await run_pipeline(
        PipelineConfig(
            name="skip_test",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(
                    name="double",
                    fn=double,
                    skip_when=lambda ctx: ctx["add_one"] > 5,
                ),
                Step(name="to_string", fn=fallback_to_string),
            ],
            tracer=tracer2,
        ),
        input=200,
    )
    assert "double" not in result2.step_outputs
    assert result2.output == "201"

    # Verify the skipped step has a span with "skipped" output
    skipped = [s for s in tracer2.spans if s.name == "double"]
    assert len(skipped) == 1
    assert skipped[0].output == "skipped"


async def test_per_step_grading():
    """Step has grader, verify step_scores populated."""
    tracer = FakeTracer()
    grader = FixedGrader(0.9)
    result = await run_pipeline(
        PipelineConfig(
            name="graded",
            steps=[
                Step(name="add_one", fn=add_one, grader=grader),
                Step(name="double", fn=double),
            ],
            tracer=tracer,
        ),
        input=5,
    )
    assert "add_one" in result.step_scores
    assert result.step_scores["add_one"][0].value == 0.9
    assert "double" not in result.step_scores

    # Verify grading span
    grading_spans = [s for s in tracer.spans if s.kind == SpanKind.GRADING]
    assert len(grading_spans) == 1
    assert grading_spans[0].name == "grade_add_one"


async def test_pipeline_level_grading():
    """Config has grader, verify scores populated."""
    tracer = FakeTracer()
    result = await run_pipeline(
        PipelineConfig(
            name="pipeline_graded",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(name="double", fn=double),
            ],
            tracer=tracer,
            grader=FixedGrader(0.85),
        ),
        input=5,
    )
    assert result.scores is not None
    assert result.scores[0].value == 0.85

    grading_spans = [s for s in tracer.spans if s.kind == SpanKind.GRADING]
    assert len(grading_spans) == 1
    assert grading_spans[0].name == "pipeline_grade"


async def test_feedback_integration():
    """Graded result stored via feedback.score()."""
    tracer = FakeTracer()
    feedback = FakeFeedback()
    grader = FixedGrader(0.95)

    result = await run_pipeline(
        PipelineConfig(
            name="feedback_test",
            steps=[
                Step(name="add_one", fn=add_one, grader=grader),
            ],
            tracer=tracer,
            feedback=feedback,
        ),
        input=5,
    )
    assert len(feedback.scored) == 1
    result_id, scores = feedback.scored[0]
    assert result.trace_id in result_id
    assert "add_one" in result_id
    assert scores[0].value == 0.95


async def test_map_pipeline():
    """3 items, verify 3 PipelineResults, parent trace."""
    tracer = FakeTracer()
    result = await map_pipeline(
        PipelineConfig(
            name="map_test",
            steps=[
                Step(name="add_one", fn=add_one),
                Step(name="double", fn=double),
            ],
            tracer=tracer,
        ),
        items=[1, 2, 3],
    )
    assert len(result.results) == 3
    assert result.results[0].output == 4   # (1+1)*2
    assert result.results[1].output == 6   # (2+1)*2
    assert result.results[2].output == 8   # (3+1)*2

    # All child pipelines share the parent trace_id
    parent_trace_id = result.trace_id
    for r in result.results:
        assert r.trace_id == parent_trace_id

    # Parent is PIPELINE_RUN, children are nested PIPELINE_RUN spans
    run_spans = [s for s in tracer.spans if s.kind == SpanKind.PIPELINE_RUN]
    assert len(run_spans) == 4  # 1 parent + 3 children


async def test_nested_pipeline():
    """Step calls run_pipeline internally, verify nested spans."""
    tracer = FakeTracer()

    async def inner_pipeline_step(ctx: dict[str, Any]) -> int:
        inner_result = await run_pipeline(
            PipelineConfig(
                name="inner",
                steps=[
                    Step(name="add_one", fn=add_one),
                ],
                tracer=ctx["_tracer"],
            ),
            input=ctx["input"],
            _parent_span_id=ctx["_span_id"],
        )
        return inner_result.output

    result = await run_pipeline(
        PipelineConfig(
            name="outer",
            steps=[
                Step(name="nested", fn=inner_pipeline_step),
            ],
            tracer=tracer,
        ),
        input=10,
    )
    assert result.output == 11  # inner add_one: 10+1

    # Verify nesting: outer PIPELINE_RUN > nested PIPELINE_STEP + inner PIPELINE_RUN
    run_spans = [s for s in tracer.spans if s.kind == SpanKind.PIPELINE_RUN]
    assert len(run_spans) == 2  # outer + inner

    step_spans = [s for s in tracer.spans if s.kind == SpanKind.PIPELINE_STEP]
    assert len(step_spans) == 2  # outer's "nested" step + inner's "add_one" step

    # Inner pipeline run is a child of outer's root
    inner_run = [s for s in run_spans if s.name == "inner"][0]
    assert inner_run.parent_id is not None
