from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any

from jig.core.types import Span, SpanKind, TracingLogger, Usage

logger = logging.getLogger(__name__)


def _format_span_exception(exc: BaseException) -> str:
    detail = str(exc)
    if not detail:
        detail = "exception raised without message"
    return f"{type(exc).__name__}: {detail}"


@dataclass
class SpanGuard:
    """Handle yielded by :func:`span_guard`.

    ``finish`` is the normal way to close the span with output/usage before
    leaving the guarded block. If a caller does not finish explicitly, the
    context manager closes the span on a successful exit with no output, or
    with exception metadata on a failed exit.
    """

    tracer: TracingLogger
    span: Span
    _finished: bool = False

    @property
    def id(self) -> str:
        return self.span.id

    @property
    def trace_id(self) -> str:
        return self.span.trace_id

    def finish(
        self,
        output: Any = None,
        *,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.tracer.end_span(self.span.id, output=output, error=error, usage=usage)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished or self.span.ended_at is not None


@contextmanager
def span_guard(
    tracer: TracingLogger,
    parent_id: str,
    kind: SpanKind,
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
):
    """Start a span and close it on success or propagated exception.

    Callers may use ``guard.finish(output=..., usage=...)`` to attach output
    metadata before exiting the block. If they do not, the guard still closes
    the span on the success path so opened spans cannot be left dangling by a
    missing explicit ``end_span`` call.
    """
    span = tracer.start_span(parent_id, kind, name, input=input, metadata=metadata)
    guard = SpanGuard(tracer=tracer, span=span)
    try:
        yield guard
    except BaseException as exc:
        if not guard.finished:
            try:
                guard.finish(error=_format_span_exception(exc))
            except Exception:
                logger.exception("tracer.end_span failed while closing failed span")
        raise
    else:
        if not guard.finished:
            guard.finish()
