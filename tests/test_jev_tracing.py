"""Jev trace kind and usage conversion round trips."""

import ast
from pathlib import Path

import pytest

from jig.core.types import SpanKind
from jig.jev.models import JevResult, JevUsage
from jig.jev.tracing import to_jig_usage
from jig.tracing import SQLiteTracer


def test_provider_call_kind_and_unpriced_usage():
    assert SpanKind.PROVIDER_CALL.value == "provider_call"
    assert SpanKind("provider_call") is SpanKind.PROVIDER_CALL

    usage = to_jig_usage(JevUsage(input_tokens=17, output_tokens=29))
    assert usage.input_tokens == 17
    assert usage.output_tokens == 29
    assert usage.cost is None


@pytest.mark.asyncio
async def test_provider_call_round_trip(tmp_path):
    result = JevResult("jev-resolved", {}, JevUsage(17, 29), 24.5, "call-1", "request-1", 1)
    tracer = SQLiteTracer(db_path=str(tmp_path / "trace.db"))
    try:
        root = tracer.start_trace("run")
        span = tracer.start_span(
            root.id,
            SpanKind.PROVIDER_CALL,
            "jev.call",
            metadata={
                "call_id": result.call_id,
                "provider_request_id": result.provider_request_id,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "status": "ok",
            },
        )
        tracer.end_span(span.id, usage=to_jig_usage(result.usage))
        tracer.end_span(root.id)
        await tracer.flush()

        recorded = next(s for s in await tracer.get_trace(root.trace_id) if s.id == span.id)
        assert recorded.kind is SpanKind.PROVIDER_CALL
        assert recorded.usage is not None
        assert recorded.usage.input_tokens == 17
        assert recorded.usage.output_tokens == 29
        assert recorded.usage.cost is None
        assert recorded.metadata == {
            "call_id": "call-1",
            "provider_request_id": "request-1",
            "model": "jev-resolved",
            "latency_ms": 24.5,
            "status": "ok",
        }
    finally:
        await tracer.close()


def test_jev_models_do_not_import_jig_core():
    models = Path(__file__).resolve().parents[1] / "src" / "jig" / "jev" / "models.py"
    module = ast.parse(models.read_text())
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("jig.core") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("jig.core")
