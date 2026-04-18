"""Structured diff between two recorded traces.

Pairs TOOL_CALL spans by ordinal position, reports the first field
that diverges (name > args > output > error), and rolls up final-output,
cost, latency, grader-score, and error-category deltas.

``submit_output`` spans are intentionally skipped — they're runner-
internal bookkeeping, not agent-observable tool calls, and including
them clutters the diff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Literal

from jig.core.types import Span, SpanKind, TracingLogger

_SUBMIT_OUTPUT_TOOL = "submit_output"


@dataclass
class ToolEvent:
    name: str
    args: Any
    output: str | None
    error: str | None


ToolDivergenceKind = Literal[
    "name", "args", "output", "error", "only_a", "only_b",
]


@dataclass
class ToolDiff:
    """A single divergence in the ordered TOOL_CALL stream."""

    index: int
    divergence: ToolDivergenceKind
    a: ToolEvent | None
    b: ToolEvent | None


@dataclass
class TraceDiff:
    trace_a_id: str
    trace_b_id: str
    tool_divergence: list[ToolDiff] = field(default_factory=list)
    output_diff: tuple[str, str] | None = None
    error_category_change: tuple[str | None, str | None] | None = None
    score_deltas: dict[str, float] = field(default_factory=dict)
    cost_delta: float = 0.0
    latency_ms_delta: float = 0.0

    @property
    def identical(self) -> bool:
        """True when the two traces show no meaningful divergence."""
        return (
            not self.tool_divergence
            and self.output_diff is None
            and self.error_category_change is None
            and not self.score_deltas
        )


def _tool_spans(spans: list[Span]) -> list[Span]:
    return [
        s for s in spans
        if s.kind == SpanKind.TOOL_CALL and s.name != _SUBMIT_OUTPUT_TOOL
    ]


def _to_event(span: Span) -> ToolEvent:
    output = span.output if isinstance(span.output, str) else (
        None if span.output is None else str(span.output)
    )
    return ToolEvent(
        name=span.name,
        args=span.input,
        output=output,
        error=span.error,
    )


def _classify(a: ToolEvent, b: ToolEvent) -> ToolDivergenceKind | None:
    """Return the first divergence kind, or None if ``a`` and ``b`` match."""
    if a.name != b.name:
        return "name"
    if a.args != b.args:
        return "args"
    if a.output != b.output:
        return "output"
    if a.error != b.error:
        return "error"
    return None


def _root(spans: list[Span]) -> Span | None:
    return next(
        (s for s in spans if s.parent_id is None and s.kind == SpanKind.AGENT_RUN),
        None,
    )


def _final_output_preview(root: Span) -> str:
    if isinstance(root.output, dict):
        value = root.output.get("output")
        return value if isinstance(value, str) else ""
    return ""


def _error_category(root: Span) -> str | None:
    if isinstance(root.output, dict):
        value = root.output.get("error_category")
        return value if isinstance(value, str) else None
    return None


def _avg_scores(spans: list[Span]) -> dict[str, float]:
    """Extract per-dimension average scores from GRADING spans.

    Grading spans record their scores on the output dict; we compute a
    mean per dimension name so a diff over two traces can call out
    regressed/improved dimensions directly.
    """
    buckets: dict[str, list[float]] = {}
    for s in spans:
        if s.kind != SpanKind.GRADING or not isinstance(s.output, dict):
            continue
        scores = s.output.get("scores")
        if not isinstance(scores, list):
            continue
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            dim = entry.get("dimension")
            val = entry.get("value")
            if not isinstance(dim, str):
                continue
            if isinstance(val, (int, float)):
                buckets.setdefault(dim, []).append(float(val))
    return {dim: sum(v) / len(v) for dim, v in buckets.items() if v}


def _trace_totals(spans: list[Span]) -> tuple[float, float]:
    """Return (total_cost, root_duration_ms)."""
    cost = 0.0
    for s in spans:
        if s.usage is not None and s.usage.cost is not None:
            cost += float(s.usage.cost)
    root = _root(spans)
    duration = float(root.duration_ms) if root and root.duration_ms else 0.0
    return cost, duration


async def trace_diff(
    trace_a_id: str,
    trace_b_id: str,
    *,
    tracer: TracingLogger,
) -> TraceDiff:
    """Diff two recorded traces via the supplied tracer.

    Both traces must already be flushed to whatever backend ``tracer``
    reads — typically :class:`SQLiteTracer`. The :class:`TraceDiff`
    returned is frame-agnostic; serialize it to JSON if you need
    dashboards.
    """
    a_spans = await tracer.get_trace(trace_a_id)
    b_spans = await tracer.get_trace(trace_b_id)

    a_tools = _tool_spans(a_spans)
    b_tools = _tool_spans(b_spans)

    tool_divergence: list[ToolDiff] = []
    for idx, (a_span, b_span) in enumerate(zip_longest(a_tools, b_tools)):
        if a_span is None:
            tool_divergence.append(ToolDiff(
                index=idx,
                divergence="only_b",
                a=None,
                b=_to_event(b_span),
            ))
            continue
        if b_span is None:
            tool_divergence.append(ToolDiff(
                index=idx,
                divergence="only_a",
                a=_to_event(a_span),
                b=None,
            ))
            continue
        a_event = _to_event(a_span)
        b_event = _to_event(b_span)
        kind = _classify(a_event, b_event)
        if kind is not None:
            tool_divergence.append(ToolDiff(
                index=idx, divergence=kind, a=a_event, b=b_event,
            ))

    a_root = _root(a_spans)
    b_root = _root(b_spans)

    output_diff: tuple[str, str] | None = None
    error_category_change: tuple[str | None, str | None] | None = None
    if a_root and b_root:
        a_out = _final_output_preview(a_root)
        b_out = _final_output_preview(b_root)
        if a_out != b_out:
            output_diff = (a_out, b_out)
        a_err = _error_category(a_root)
        b_err = _error_category(b_root)
        if a_err != b_err:
            error_category_change = (a_err, b_err)

    a_scores = _avg_scores(a_spans)
    b_scores = _avg_scores(b_spans)
    score_deltas: dict[str, float] = {}
    # Iterate the union so a grader dimension that exists in only one
    # trace still shows up — a dropped or added dimension is a real
    # regression the diff must surface. Missing side contributes 0.0
    # (so an added dim appears as a positive delta, a dropped one
    # negative).
    for dim in set(a_scores) | set(b_scores):
        delta = b_scores.get(dim, 0.0) - a_scores.get(dim, 0.0)
        if delta != 0:
            score_deltas[dim] = delta

    a_cost, a_duration = _trace_totals(a_spans)
    b_cost, b_duration = _trace_totals(b_spans)

    return TraceDiff(
        trace_a_id=trace_a_id,
        trace_b_id=trace_b_id,
        tool_divergence=tool_divergence,
        output_diff=output_diff,
        error_category_change=error_category_change,
        score_deltas=score_deltas,
        cost_delta=b_cost - a_cost,
        latency_ms_delta=b_duration - a_duration,
    )
