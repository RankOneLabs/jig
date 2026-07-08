from __future__ import annotations

from jig.core.types import SpanKind, Usage
from jig.tracing.stdout import StdoutTracer


def test_stdout_tracer_prints_zero_cost_and_releases_finished_span(capsys) -> None:
    tracer = StdoutTracer(color=False)
    span = tracer.start_trace("agent", kind=SpanKind.AGENT_RUN)

    tracer.end_span(
        span.id,
        output="done",
        usage=Usage(input_tokens=3, output_tokens=2, cost=0.0),
    )

    output = capsys.readouterr().out
    assert "$0.0000" in output
    assert span.id not in tracer._spans
    assert span.id not in tracer._depth


def test_stdout_tracer_ignores_unknown_span() -> None:
    tracer = StdoutTracer(color=False)

    tracer.end_span("missing", output="done")

    assert tracer._spans == {}
    assert tracer._depth == {}
