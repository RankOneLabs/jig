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
from typing import Literal

from jig.core.runner import SUBMIT_OUTPUT_TOOL as _SUBMIT_OUTPUT_TOOL
from jig.core.types import Span, SpanKind, TracingLogger
from jig.replay.align import (
    Aligner,
    Alignment,
    AlignmentTier,
    IdentityAligner,
    OrdinalAligner,
    ToolEvent,
)

ToolDivergenceKind = Literal[
    "name", "args", "output", "error", "only_a", "only_b",
]


@dataclass
class ToolDiff:
    """A single divergence in the aligned TOOL_CALL stream.

    ``index`` is the A-centric report ordinal (see :class:`TraceDiff`),
    not a source position. ``index_a`` / ``index_b`` are the event's
    position in the filtered (post submit_output) trace-A / trace-B
    tool-call lists — a paired divergence has both, ``only_a`` has only
    ``index_a``, and ``only_b`` has only ``index_b``. ``tier`` records
    the alignment strength that produced the pairing or unmatched
    status; see :mod:`jig.replay.align` for what each tier asserts.
    """

    index: int
    divergence: ToolDivergenceKind
    a: ToolEvent | None
    b: ToolEvent | None
    tier: AlignmentTier | None = None
    index_a: int | None = None
    index_b: int | None = None


@dataclass
class TraceDiff:
    trace_a_id: str
    trace_b_id: str
    tool_divergence: list[ToolDiff] = field(default_factory=list)
    output_diff: tuple[str, str] | None = None
    error_category_change: tuple[str | None, str | None] | None = None
    score_deltas: dict[str, float] = field(default_factory=dict)
    # Per-dimension (a_avg, b_avg) — None on a side means that dimension
    # was absent from that trace (added or dropped). Use this to recover
    # old/new aggregate values when score_deltas only gives the difference.
    score_details: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    cost_delta: float = 0.0
    latency_ms_delta: float = 0.0

    @property
    def identical(self) -> bool:
        """True when the two traces show no *behavioral* divergence.

        "Behavioral" means tool-call sequence, final output, terminal
        error category, and grader scores all match. ``cost_delta`` and
        ``latency_ms_delta`` are deliberately excluded — swapping models
        is the main use case, and a different model will almost always
        have different token prices and latency, so folding those in
        would make ``identical`` always False for the exact workflow
        this property exists to support. Use :attr:`fully_identical`
        when you want to detect any change at all.
        """
        return (
            not self.tool_divergence
            and self.output_diff is None
            and self.error_category_change is None
            and not self.score_deltas
        )

    @property
    def fully_identical(self) -> bool:
        """True when :attr:`identical` holds *and* cost/latency also match.

        Stricter sibling of :attr:`identical`. Useful when you're not
        swapping models — e.g. verifying a deterministic replay of the
        same config reproduces the recording down to spend.
        """
        return (
            self.identical
            and self.cost_delta == 0.0
            and self.latency_ms_delta == 0.0
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


def _validate_identity_fields(identity_fields: dict[str, list[str]] | None) -> None:
    """Reject a malformed raw ``identity_fields`` mapping before any I/O.

    Raw dicts bypass :meth:`ToolDefinition.__post_init__`, so this
    re-checks the same shape constraints synchronously: every tool name
    must be a string, every declaration a non-empty list of non-empty
    strings with no empty dot segment and no duplicate path.
    """
    if identity_fields is None:
        return
    if not isinstance(identity_fields, dict):
        raise ValueError(
            f"identity_fields must be a dict[str, list[str]] or None, "
            f"got {identity_fields!r}"
        )
    for tool_name, paths in identity_fields.items():
        if not isinstance(tool_name, str):
            raise ValueError(
                f"identity_fields has a non-string tool name: {tool_name!r}"
            )
        if not isinstance(paths, list):
            raise ValueError(
                f"identity_fields[{tool_name!r}] must be a list[str], "
                f"got {paths!r}"
            )
        if len(paths) == 0:
            raise ValueError(
                f"identity_fields[{tool_name!r}] must not be an empty list "
                f"(omit the tool to declare no identity)"
            )
        seen: set[str] = set()
        for path in paths:
            if not isinstance(path, str):
                raise ValueError(
                    f"identity_fields[{tool_name!r}] has a non-string "
                    f"path: {path!r}"
                )
            if path == "":
                raise ValueError(
                    f"identity_fields[{tool_name!r}] has an empty path"
                )
            if any(segment == "" for segment in path.split(".")):
                raise ValueError(
                    f"identity_fields[{tool_name!r}] path {path!r} has an "
                    f"empty dot segment"
                )
            if path in seen:
                raise ValueError(
                    f"identity_fields[{tool_name!r}] path {path!r} is "
                    f"duplicated"
                )
            seen.add(path)


def _build_tool_divergence(
    a_events: list[ToolEvent],
    b_events: list[ToolEvent],
    alignment: Alignment,
) -> list[ToolDiff]:
    """Classify each aligned pair, convert unmatched events, and order
    the result A-centrically.

    Divergent pairs and ``only_a`` entries are sorted together by
    ``index_a``, then every ``only_b`` entry is appended sorted by
    ``index_b``; ``ToolDiff.index`` is finally assigned as consecutive
    report ordinals starting at zero.
    """
    a_centric: list[ToolDiff] = []
    for pair in alignment.pairs:
        a_event = a_events[pair.index_a]
        b_event = b_events[pair.index_b]
        kind = _classify(a_event, b_event)
        if kind is not None:
            a_centric.append(ToolDiff(
                index=0,
                divergence=kind,
                a=a_event,
                b=b_event,
                tier=pair.tier,
                index_a=pair.index_a,
                index_b=pair.index_b,
            ))
    for unmatched in alignment.only_a:
        a_centric.append(ToolDiff(
            index=0,
            divergence="only_a",
            a=a_events[unmatched.index],
            b=None,
            tier=unmatched.tier,
            index_a=unmatched.index,
            index_b=None,
        ))
    a_centric.sort(key=lambda d: d.index_a)  # type: ignore[arg-type, return-value]

    only_b_diffs = [
        ToolDiff(
            index=0,
            divergence="only_b",
            a=None,
            b=b_events[unmatched.index],
            tier=unmatched.tier,
            index_a=None,
            index_b=unmatched.index,
        )
        for unmatched in sorted(alignment.only_b, key=lambda u: u.index)
    ]

    tool_divergence = a_centric + only_b_diffs
    for report_index, diff_entry in enumerate(tool_divergence):
        diff_entry.index = report_index
    return tool_divergence


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

    Accepts both the canonical shape ``{dimension, value}`` emitted by
    production writers and the legacy shape ``{dim, val}`` found in
    historical fixtures. Canonical keys take precedence when both are
    present in the same entry.
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
            # Canonical keys take precedence; fall back to legacy aliases.
            dim = entry.get("dimension") if "dimension" in entry else entry.get("dim")
            val = entry.get("value") if "value" in entry else entry.get("val")
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
    identity_fields: dict[str, list[str]] | None = None,
) -> TraceDiff:
    """Diff two recorded traces via the supplied tracer.

    Both traces must already be flushed to whatever backend ``tracer``
    reads — typically :class:`SQLiteTracer`. The :class:`TraceDiff`
    returned is frame-agnostic; serialize it to JSON if you need
    dashboards.

    ``identity_fields`` is an optional ``{tool_name: [path, ...]}`` map
    of the same shape as :attr:`ToolDefinition.identity_fields`,
    applied symmetrically and retroactively to both traces — it must be
    valid for both, and there is no support for comparing histories
    recorded under incompatible identity contracts. A falsey value
    (``None`` or ``{}``) preserves exact legacy ordinal (zip-by-
    position) pairing; a non-empty mapping opts the whole diff into the
    three-tier identity-aware aligner. A tool omitted from a non-empty
    mapping is identity-less and may still be paired by patience
    anchors or segment-ordinal fallback, which can differ from legacy
    whole-list ordinal pairing. The mapping is validated synchronously,
    before either trace is read: a non-string tool name, a non-list or
    empty declaration, a non-string or empty path, a path with an empty
    dot segment, or a duplicate path all raise ``ValueError``.
    """
    _validate_identity_fields(identity_fields)

    a_spans = await tracer.get_trace(trace_a_id)
    b_spans = await tracer.get_trace(trace_b_id)

    # Reject missing / malformed traces up front. Silently returning an
    # empty diff (the previous behavior) made an ``identical`` report
    # out of two unknown trace IDs, which is misleading for debugging.
    if not a_spans:
        raise ValueError(f"Trace {trace_a_id!r} not found or has no spans")
    if not b_spans:
        raise ValueError(f"Trace {trace_b_id!r} not found or has no spans")
    if _root(a_spans) is None:
        raise ValueError(
            f"Trace {trace_a_id!r} has no AGENT_RUN root span"
        )
    if _root(b_spans) is None:
        raise ValueError(
            f"Trace {trace_b_id!r} has no AGENT_RUN root span"
        )

    a_tools = _tool_spans(a_spans)
    b_tools = _tool_spans(b_spans)
    a_events = [_to_event(s) for s in a_tools]
    b_events = [_to_event(s) for s in b_tools]

    aligner: Aligner = IdentityAligner() if identity_fields else OrdinalAligner()
    alignment = aligner.align(a_events, b_events, identity_fields=identity_fields)
    tool_divergence = _build_tool_divergence(a_events, b_events, alignment)

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
    score_details: dict[str, tuple[float | None, float | None]] = {}
    # Iterate the union so a grader dimension that exists in only one
    # trace still shows up — a dropped or added dimension is a real
    # regression the diff must surface. Missing side contributes 0.0
    # to the numeric delta, but asymmetric presence (one side None) is
    # always a rubric change even when the present side scores 0.0.
    for dim in set(a_scores) | set(b_scores):
        a_val: float | None = a_scores.get(dim)
        b_val: float | None = b_scores.get(dim)
        delta = (b_val if b_val is not None else 0.0) - (a_val if a_val is not None else 0.0)
        asymmetric = (a_val is None) != (b_val is None)
        if delta != 0 or asymmetric:
            score_deltas[dim] = delta
        score_details[dim] = (a_val, b_val)

    a_cost, a_duration = _trace_totals(a_spans)
    b_cost, b_duration = _trace_totals(b_spans)

    return TraceDiff(
        trace_a_id=trace_a_id,
        trace_b_id=trace_b_id,
        tool_divergence=tool_divergence,
        output_diff=output_diff,
        error_category_change=error_category_change,
        score_deltas=score_deltas,
        score_details=score_details,
        cost_delta=b_cost - a_cost,
        latency_ms_delta=b_duration - a_duration,
    )
