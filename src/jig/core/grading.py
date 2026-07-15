"""Shared fail-soft grading policy for run_agent, pipelines, and batches.

Grading (and the feedback persistence that follows a successful grade) must
never turn a successful execution into a failed one. This module is the one
place that policy is implemented, so run_agent, per-step pipeline grading,
pipeline-level grading, and map_pipeline batch grading can't drift from each
other.

Only ``Exception`` is caught — never ``BaseException``. Cancellation,
``KeyboardInterrupt``, and process-control signals are not grading failures
and must keep propagating.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jig.core.types import FeedbackLoop, Grader, Score, SpanKind, TracingLogger
from jig.feedback.validation import validate_scores

logger = logging.getLogger(__name__)


@dataclass
class GradingOutcome:
    """Result of :func:`grade_and_record`.

    ``scores`` is ``None`` when grading itself failed (grader raised, or
    returned scores that failed validation) — distinct from a real empty
    list, which means the grader ran successfully and assigned no scores.
    ``feedback_result_id`` is ``None`` whenever feedback persistence did not
    complete, whether because grading failed, there was nothing to store, or
    storage itself failed.
    """

    scores: list[Score] | None
    feedback_result_id: str | None


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
) -> GradingOutcome:
    """Grade, then (if configured) persist the result — fail-soft throughout.

    Starts and closes one GRADING span. A grader exception or invalid score
    output logs the failure, records ``grading_error`` (type + message) as
    the span output, sets the span error, and returns ``scores=None`` —
    execution output the caller already produced is untouched.

    A feedback persistence failure (``store_result``/``score`` raising)
    after a successful grade does not discard the scores already computed:
    it's recorded as ``feedback_error`` on the span (which is not marked as
    an error span), ``feedback_result_id`` is omitted, and the real scores
    are still returned.
    """
    grade_span = tracer.start_span(parent_span_id, SpanKind.GRADING, span_name)

    try:
        scores = await grader.grade(grade_input, grade_output, grade_context)
        if not isinstance(scores, list):
            raise TypeError(
                f"grader returned {type(scores).__name__}, expected list[Score]"
            )
        if scores:
            validate_scores(scores)
    except Exception as exc:
        logger.exception("grading failed (non-fatal, execution output preserved)")
        tracer.end_span(
            grade_span.id,
            output={
                "scores": [],
                "grading_error": {"type": type(exc).__name__, "message": str(exc)},
            },
            error=f"{type(exc).__name__}: {exc}",
        )
        return GradingOutcome(scores=None, feedback_result_id=None)

    feedback_result_id: str | None = None
    feedback_error: dict[str, str] | None = None
    if feedback is not None and scores:
        try:
            feedback_result_id = await feedback.store_result(
                feedback_content if feedback_content is not None else "",
                feedback_input_text if feedback_input_text is not None else "",
                feedback_metadata,
            )
            await feedback.score(feedback_result_id, scores)
        except Exception as exc:
            logger.exception(
                "feedback persistence failed after successful grading (non-fatal)"
            )
            feedback_result_id = None
            feedback_error = {"type": type(exc).__name__, "message": str(exc)}

    span_output: dict[str, Any] = {
        "scores": [{"dimension": s.dimension, "value": s.value} for s in scores],
    }
    if feedback_result_id is not None:
        span_output["feedback_result_id"] = feedback_result_id
    if feedback_error is not None:
        span_output["feedback_error"] = feedback_error
    tracer.end_span(grade_span.id, output=span_output)

    return GradingOutcome(scores=scores, feedback_result_id=feedback_result_id)
