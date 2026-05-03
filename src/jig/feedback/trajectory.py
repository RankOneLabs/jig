"""Trajectory grading — score the recorded span list, not the final output.

`Grader[T]` receives the agent's final output. Agent behavior often lives
in the trajectory: which tools fired in what order, did memory get
queried, how many steps. ``TrajectoryGrader`` consumes the span list via
the supplied tracer and turns assertions over that list into ``Score``
objects.

The framework's grade-call sites (``run_agent`` auto-grade,
``run_pipeline`` per-step / pipeline / batch) flush the tracer and pass
``trace_id`` in the grader context so this grader can read the run's
spans via ``tracer.get_trace`` mid-run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jig.core.types import (
    Grader,
    Score,
    ScoreSource,
    Span,
    SpanKind,
    TracingLogger,
)


@dataclass
class TrajectoryAssertion:
    """A single named check against the recorded trajectory.

    ``name`` becomes the Score dimension. ``check`` returns a float in
    ``[0.0, 1.0]`` given the ordered span list. Use 0/1 for pass/fail
    or fractional for partial credit (e.g. "called 3 of 4 expected
    tools").
    """

    name: str
    check: Callable[[list[Span]], float]


class TrajectoryGrader(Grader[Any]):
    """Score the recorded trajectory rather than the final output.

    Reads ``context["trace_id"]`` (set by ``run_agent`` and the
    pipeline grade-call sites) and pulls the span list via the
    supplied tracer. Each assertion produces one ``Score`` with
    ``source=HEURISTIC``. The ``output`` parameter is unused — pair
    with a ``CompositeGrader`` if you also want output-level grading.

    When ``trace_id`` is missing from the context (e.g. a caller
    invokes ``grade()`` directly without it), every assertion returns
    0.0 rather than raising — this matches the existing grader
    convention of failing soft so a single misconfigured grader
    doesn't crash a sweep.
    """

    def __init__(
        self,
        tracer: TracingLogger,
        assertions: list[TrajectoryAssertion],
    ):
        self._tracer = tracer
        self._assertions = assertions

    async def grade(
        self,
        input: Any,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        if not context or "trace_id" not in context:
            return [
                Score(
                    dimension=a.name,
                    value=0.0,
                    source=ScoreSource.HEURISTIC,
                )
                for a in self._assertions
            ]
        spans = await self._tracer.get_trace(context["trace_id"])
        scores: list[Score] = []
        for a in self._assertions:
            try:
                value = a.check(spans)
            except Exception:
                value = 0.0
            scores.append(
                Score(
                    dimension=a.name,
                    value=max(0.0, min(1.0, float(value))),
                    source=ScoreSource.HEURISTIC,
                )
            )
        return scores


def tool_called(name: str) -> Callable[[list[Span]], float]:
    """Pass if a TOOL_CALL span with this name appears in the trace."""

    def check(spans: list[Span]) -> float:
        return 1.0 if any(
            s.kind == SpanKind.TOOL_CALL and s.name == name
            for s in spans
        ) else 0.0

    return check


def tool_sequence(names: list[str]) -> Callable[[list[Span]], float]:
    """Fractional: longest matching prefix of expected tool order.

    ``[a, b, c]`` against actual ``[a, b, x]`` scores 2/3. Empty
    expectation list scores 1.0 (vacuously satisfied).
    """

    def check(spans: list[Span]) -> float:
        actual = [s.name for s in spans if s.kind == SpanKind.TOOL_CALL]
        if not names:
            return 1.0
        match = 0
        for expected, observed in zip(names, actual):
            if expected != observed:
                break
            match += 1
        return match / len(names)

    return check


def step_budget(max_steps: int) -> Callable[[list[Span]], float]:
    """Pass if the agent stayed under the step budget.

    Counts ``LLM_CALL`` spans — one per agent turn. Returns 1.0 when
    the count is ``<= max_steps``, 0.0 otherwise.
    """

    def check(spans: list[Span]) -> float:
        steps = sum(1 for s in spans if s.kind == SpanKind.LLM_CALL)
        return 1.0 if steps <= max_steps else 0.0

    return check
