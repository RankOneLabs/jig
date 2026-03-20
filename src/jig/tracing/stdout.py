from __future__ import annotations

import sys
import uuid
from datetime import datetime
from typing import Any

from jig.core.types import Span, SpanKind, TracingLogger

_COLORS = {
    SpanKind.AGENT_RUN: "\033[1;36m",      # bold cyan
    SpanKind.PIPELINE_RUN: "\033[1;36m",   # bold cyan
    SpanKind.PIPELINE_STEP: "\033[1;37m",  # bold white
    SpanKind.LLM_CALL: "\033[1;33m",      # bold yellow
    SpanKind.TOOL_CALL: "\033[1;32m",     # bold green
    SpanKind.MEMORY_QUERY: "\033[1;35m",  # bold magenta
    SpanKind.GRADING: "\033[1;34m",       # bold blue
}
_RESET = "\033[0m"
_RED = "\033[1;31m"


class StdoutTracer(TracingLogger):
    def __init__(self, color: bool = True):
        self._color = color and sys.stdout.isatty()
        self._spans: dict[str, Span] = {}
        self._depth: dict[str, int] = {}

    def _c(self, kind: SpanKind) -> str:
        return _COLORS.get(kind, "") if self._color else ""

    def _r(self) -> str:
        return _RESET if self._color else ""

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        span = Span(
            id=span_id,
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            metadata=metadata,
        )
        self._spans[span_id] = span
        self._depth[span_id] = 0
        print(f"{self._c(kind)}[{kind.value}] {name} started{self._r()}")
        return span

    def start_span(
        self, parent_id: str, kind: SpanKind, name: str, input: Any = None
    ) -> Span:
        parent = self._spans.get(parent_id)
        trace_id = parent.trace_id if parent else parent_id
        span_id = str(uuid.uuid4())
        depth = self._depth.get(parent_id, 0) + 1
        span = Span(
            id=span_id,
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            parent_id=parent_id,
            input=input,
        )
        self._spans[span_id] = span
        self._depth[span_id] = depth
        indent = "  " * depth
        print(f"{indent}{self._c(kind)}[{kind.value}] {name}{self._r()}")
        return span

    def end_span(self, span_id: str, output: Any = None, error: str | None = None) -> None:
        span = self._spans.get(span_id)
        if not span:
            return
        span.ended_at = datetime.now()
        span.duration_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        span.output = output
        span.error = error

        depth = self._depth.get(span_id, 0)
        indent = "  " * depth

        if error:
            red = _RED if self._color else ""
            print(f"{indent}{red}[{span.kind.value}] {span.name} ERROR: {error}{self._r()}")
        else:
            c = self._c(span.kind)
            print(f"{indent}{c}[{span.kind.value}] {span.name} {span.duration_ms:.0f}ms{self._r()}")

        if span.usage:
            print(
                f"{indent}  tokens: {span.usage.input_tokens}→{span.usage.output_tokens}"
                + (f" ${span.usage.cost:.4f}" if span.usage.cost else "")
            )

    async def get_trace(self, trace_id: str) -> list[Span]:
        raise NotImplementedError("StdoutTracer does not support get_trace")

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        raise NotImplementedError("StdoutTracer does not support list_traces")
