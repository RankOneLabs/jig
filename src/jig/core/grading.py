"""Shared fail-soft grading policy for run_agent, pipelines, and batches.

Grading and feedback persistence are intermediate stages. Expected failures are
returned as typed results: they never erase a successful worker output, and they
never disappear into logs or tracing alone. Cancellation, ``KeyboardInterrupt``,
and other process-control ``BaseException`` values continue to propagate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from jig.core.types import FeedbackLoop, Grader, Score, SpanKind, TracingLogger
from jig.feedback.validation import validate_scores

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StageError:
    """Serializable identity of an expected intermediate-stage exception."""

    stage: Literal["grade", "validate", "feedback_store", "feedback_score"]
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class FeedbackStored:
    result_id: str


@dataclass(frozen=True, slots=True)
class FeedbackSkipped:
    reason: Literal["not_configured", "no_scores"]


@dataclass(frozen=True, slots=True)
class FeedbackFailed:
    error: StageError
    # store_result may succeed before score fails. Preserve that partial
    # success instead of pretending no feedback row was created.
    result_id: str | None = None


type FeedbackResult = FeedbackStored | FeedbackSkipped | FeedbackFailed


@dataclass(frozen=True, slots=True)
class GradingSucceeded:
    scores: list[Score]
    feedback: FeedbackResult


@dataclass(frozen=True, slots=True)
class GradingFailed:
    error: StageError


type GradingResult = GradingSucceeded | GradingFailed


def _stage_error(
    stage: Literal["grade", "validate", "feedback_store", "feedback_score"],
    exc: Exception,
) -> StageError:
    return StageError(stage=stage, type=type(exc).__name__, message=str(exc))


def _record_grading_failure(
    *,
    tracer: TracingLogger,
    span_id: str,
    error: StageError,
) -> GradingFailed:
    logger.exception("grading failed (non-fatal, execution output preserved)")
    tracer.end_span(
        span_id,
        output={
            "scores": [],
            "grading_error": {
                "stage": error.stage,
                "type": error.type,
                "message": error.message,
            },
        },
        error=f"{error.type}: {error.message}",
    )
    return GradingFailed(error=error)


async def grade_and_record(
    *,
    tracer: TracingLogger,
    parent_span_id: str,
    span_name: str,
    grader: Grader[Any],
    grade_input: Any,
    grade_output: Any,
    grade_context: dict[str, Any] | None = None,
    feedback: FeedbackLoop | None = None,
    feedback_content: str | None = None,
    feedback_input_text: str | None = None,
    feedback_metadata: dict[str, Any] | None = None,
) -> GradingResult:
    """Grade and optionally persist feedback, returning every stage outcome.

    A grader exception or invalid score result returns :class:`GradingFailed`.
    Feedback persistence happens only after successful grading and is represented
    independently inside :class:`GradingSucceeded`, including partial success
    when ``store_result`` succeeded but ``score`` failed.
    """
    grade_span = tracer.start_span(parent_span_id, SpanKind.GRADING, span_name)

    try:
        scores = await grader.grade(grade_input, grade_output, grade_context)
    except Exception as exc:
        return _record_grading_failure(
            tracer=tracer,
            span_id=grade_span.id,
            error=_stage_error("grade", exc),
        )

    try:
        if not isinstance(scores, list):
            raise TypeError(
                f"grader returned {type(scores).__name__}, expected list[Score]"
            )
        if scores:
            validate_scores(scores)
    except Exception as exc:
        return _record_grading_failure(
            tracer=tracer,
            span_id=grade_span.id,
            error=_stage_error("validate", exc),
        )

    feedback_result: FeedbackResult
    feedback_error: StageError | None = None
    feedback_result_id: str | None = None
    if feedback is None:
        feedback_result = FeedbackSkipped(reason="not_configured")
    elif not scores:
        feedback_result = FeedbackSkipped(reason="no_scores")
    else:
        try:
            feedback_result_id = await feedback.store_result(
                feedback_content if feedback_content is not None else "",
                feedback_input_text if feedback_input_text is not None else "",
                feedback_metadata,
            )
        except Exception as exc:
            logger.exception(
                "feedback persistence failed after successful grading (non-fatal)"
            )
            feedback_error = _stage_error("feedback_store", exc)
            feedback_result = FeedbackFailed(error=feedback_error)
        else:
            try:
                await feedback.score(feedback_result_id, scores)
            except Exception as exc:
                logger.exception(
                    "feedback persistence failed after successful grading (non-fatal)"
                )
                feedback_error = _stage_error("feedback_score", exc)
                feedback_result = FeedbackFailed(
                    error=feedback_error,
                    result_id=feedback_result_id,
                )
            else:
                feedback_result = FeedbackStored(result_id=feedback_result_id)

    span_output: dict[str, Any] = {
        "scores": [{"dimension": s.dimension, "value": s.value} for s in scores],
    }
    if isinstance(feedback_result, FeedbackStored):
        span_output["feedback_result_id"] = feedback_result.result_id
    if feedback_error is not None:
        span_output["feedback_error"] = {
            "stage": feedback_error.stage,
            "type": feedback_error.type,
            "message": feedback_error.message,
        }
    tracer.end_span(grade_span.id, output=span_output)

    return GradingSucceeded(scores=scores, feedback=feedback_result)


__all__ = [
    "FeedbackFailed",
    "FeedbackResult",
    "FeedbackSkipped",
    "FeedbackStored",
    "GradingFailed",
    "GradingResult",
    "GradingSucceeded",
    "StageError",
    "grade_and_record",
]
