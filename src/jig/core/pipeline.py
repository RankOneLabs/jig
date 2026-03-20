from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Awaitable

from jig.core.types import FeedbackLoop, Grader, Score, SpanKind, TracingLogger


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    fn: Callable[[dict[str, Any]], Awaitable[Any]]
    grader: Grader | None = None
    skip_when: Callable[[dict[str, Any]], bool] | None = None


@dataclass
class PipelineConfig:
    name: str
    steps: Sequence[Step]
    tracer: TracingLogger

    grader: Grader | None = None
    feedback: FeedbackLoop | None = None
    is_err: Callable[[Any], bool] | None = None
    extract_err: Callable[[Any], str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PipelineResult:
    output: Any
    trace_id: str
    step_outputs: dict[str, Any]
    scores: list[Score] | None
    step_scores: dict[str, list[Score]]
    duration_ms: float
    short_circuited: bool
    error_step: str | None


@dataclass
class MapResult:
    results: list[PipelineResult]
    trace_id: str
    duration_ms: float
    scores: list[Score] | None


async def run_pipeline(
    config: PipelineConfig,
    input: Any,
    context: dict[str, Any] | None = None,
    _parent_span_id: str | None = None,
) -> PipelineResult:
    start = time.time()

    # 1. Start trace or child span
    if _parent_span_id:
        root = config.tracer.start_span(
            _parent_span_id, SpanKind.PIPELINE_RUN, config.name, input=input
        )
        trace_id = root.trace_id
    else:
        root = config.tracer.start_trace(
            config.name, metadata=config.metadata, kind=SpanKind.PIPELINE_RUN
        )
        trace_id = root.trace_id

    # 2. Init context
    ctx: dict[str, Any] = {
        "input": input,
        "_tracer": config.tracer,
        "_span_id": root.id,
        **(context or {}),
    }

    step_outputs: dict[str, Any] = {}
    step_scores: dict[str, list[Score]] = {}
    short_circuited = False
    error_step: str | None = None
    error_detail: str | None = None
    last_output: Any = input

    # 3. Execute steps
    try:
        for step in config.steps:
            # Check skip_when
            if step.skip_when and step.skip_when(ctx):
                skip_span = config.tracer.start_span(
                    root.id, SpanKind.PIPELINE_STEP, step.name
                )
                config.tracer.end_span(skip_span.id, output="skipped")
                continue

            step_span = config.tracer.start_span(
                root.id, SpanKind.PIPELINE_STEP, step.name
            )

            try:
                result = await step.fn(ctx)
            except Exception as exc:
                config.tracer.end_span(step_span.id, error=str(exc))
                raise

            # Store in context and outputs
            ctx[step.name] = result
            step_outputs[step.name] = result
            last_output = result

            # Check for error
            if config.is_err and config.is_err(result):
                error_detail = (
                    config.extract_err(result) if config.extract_err else str(result)
                )
                config.tracer.end_span(step_span.id, error=error_detail)
                short_circuited = True
                error_step = step.name
                break

            config.tracer.end_span(step_span.id, output=result)

            # Per-step grading
            if step.grader:
                grade_span = config.tracer.start_span(
                    root.id, SpanKind.GRADING, f"grade_{step.name}"
                )
                try:
                    scores = await step.grader.grade(input, result, ctx)
                    step_scores[step.name] = scores
                    config.tracer.end_span(
                        grade_span.id,
                        output=[{"dim": s.dimension, "val": s.value} for s in scores],
                    )
                except Exception as exc:
                    config.tracer.end_span(grade_span.id, error=str(exc))
                    raise

                # Feedback integration
                if config.feedback and scores:
                    await config.feedback.score(
                        f"{trace_id}:{step.name}", scores
                    )

        # 4. Pipeline-level grading
        pipeline_scores: list[Score] | None = None
        if config.grader and not short_circuited:
            grade_span = config.tracer.start_span(
                root.id, SpanKind.GRADING, "pipeline_grade"
            )
            try:
                pipeline_scores = await config.grader.grade(input, last_output)
                config.tracer.end_span(
                    grade_span.id,
                    output=[
                        {"dim": s.dimension, "val": s.value} for s in pipeline_scores
                    ],
                )
            except Exception as exc:
                config.tracer.end_span(grade_span.id, error=str(exc))
                raise

    except Exception:
        # Ensure root span is always closed on unhandled exceptions
        config.tracer.end_span(root.id, error="unhandled exception")
        raise

    # 5. Close trace
    duration = (time.time() - start) * 1000
    config.tracer.end_span(
        root.id,
        output=last_output,
        error=error_detail if short_circuited else None,
    )

    return PipelineResult(
        output=last_output,
        trace_id=trace_id,
        step_outputs=step_outputs,
        scores=pipeline_scores,
        step_scores=step_scores,
        duration_ms=duration,
        short_circuited=short_circuited,
        error_step=error_step,
    )


async def map_pipeline(
    config: PipelineConfig,
    items: Sequence[Any],
    context: dict[str, Any] | None = None,
    batch_grader: Grader | None = None,
) -> MapResult:
    start = time.time()

    # Parent trace for the batch
    parent = config.tracer.start_trace(
        f"{config.name}_batch",
        metadata={"item_count": len(items), **(config.metadata or {})},
        kind=SpanKind.PIPELINE_RUN,
    )

    results: list[PipelineResult] = []
    for item in items:
        result = await run_pipeline(
            config, item, context=context, _parent_span_id=parent.id
        )
        results.append(result)

    # Batch grading
    batch_scores: list[Score] | None = None
    if batch_grader:
        grade_span = config.tracer.start_span(
            parent.id, SpanKind.GRADING, "batch_grade"
        )
        all_outputs = [r.output for r in results]
        batch_scores = await batch_grader.grade(
            [item for item in items], all_outputs
        )
        config.tracer.end_span(
            grade_span.id,
            output=[{"dim": s.dimension, "val": s.value} for s in batch_scores],
        )

    duration = (time.time() - start) * 1000
    config.tracer.end_span(parent.id, output={"item_count": len(results)})

    return MapResult(
        results=results,
        trace_id=parent.trace_id,
        duration_ms=duration,
        scores=batch_scores,
    )
