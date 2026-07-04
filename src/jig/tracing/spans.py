from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from jig.core.types import SpanKind, TracingLogger


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
    """Start a span and close it with error details if an exception propagates.

    On success the caller is responsible for calling tracer.end_span with the
    output value.  On exception the guard calls end_span with an error string
    of the form "{ExcType}: {exc}" and re-raises so no child span is left open
    on pre-output failures.
    """
    span = tracer.start_span(parent_id, kind, name, input=input, metadata=metadata)
    try:
        yield span
    except BaseException as exc:
        tracer.end_span(span.id, error=f"{type(exc).__name__}: {exc}")
        raise
