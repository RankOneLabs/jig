"""Tests for :class:`jig.tracing.FederatedTracer` + :class:`RollupClient`."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from jig import SpanKind
from jig.tracing import FederatedTracer, RollupClient, RollupUnreachableError, SQLiteTracer


@pytest.fixture
async def local(tmp_path: Path):
    tracer = SQLiteTracer(db_path=str(tmp_path / "local.db"))
    yield tracer
    await tracer.close()


def _worker_span(
    sid: str,
    *,
    trace_id: str,
    parent_id: str,
    kind: str = "task_run",
    name: str = "task:inference",
    started_at: str = "2026-04-18T00:00:05",
    source_machine: str = "mcbain",
) -> dict:
    """Build a dict in the shape the rollup returns for a worker span."""
    return {
        "id": sid,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "kind": kind,
        "name": name,
        "started_at": started_at,
        "ended_at": "2026-04-18T00:00:06",
        "duration_ms": 1000.0,
        "input": None,
        "output": '{"status": "ok"}',
        "metadata": None,
        "error": None,
        "usage_input_tokens": 42,
        "usage_output_tokens": 9,
        "usage_cost": None,
        "source_machine": source_machine,
    }


def _rollup_transport(
    spans_by_trace: dict[str, list[dict]] | None = None,
    *,
    raise_connect_error: bool = False,
    http_status: int = 200,
) -> httpx.MockTransport:
    """Build an httpx.MockTransport that serves the rollup response shape."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if raise_connect_error:
            raise httpx.ConnectError("rollup down", request=request)
        trace_id = request.url.path.rsplit("/", 1)[-1]
        spans = (spans_by_trace or {}).get(trace_id, [])
        return httpx.Response(
            http_status,
            json={"trace_id": trace_id, "spans": spans, "sources": []},
        )

    return httpx.MockTransport(handler)


# --- RollupClient ---


async def test_rollup_client_parses_worker_rows_into_spans():
    transport = _rollup_transport({
        "t-1": [_worker_span("w-1", trace_id="t-1", parent_id="local-root")],
    })
    async with httpx.AsyncClient(transport=transport) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        spans = await client.get_trace("t-1")
    assert len(spans) == 1
    span = spans[0]
    assert span.id == "w-1"
    assert span.parent_id == "local-root"
    assert span.trace_id == "t-1"
    assert span.duration_ms == 1000.0
    assert span.usage is not None
    assert span.usage.input_tokens == 42
    # task_run (worker-origin kind) falls back to TOOL_CALL in jig's enum.
    assert span.kind == SpanKind.TOOL_CALL


async def test_rollup_client_raises_unreachable_on_connect_error():
    transport = _rollup_transport(raise_connect_error=True)
    async with httpx.AsyncClient(transport=transport) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        with pytest.raises(RollupUnreachableError, match="unreachable"):
            await client.get_trace("t-1")


async def test_rollup_client_raises_unreachable_on_http_error():
    transport = _rollup_transport(http_status=500)
    async with httpx.AsyncClient(transport=transport) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        with pytest.raises(RollupUnreachableError, match="HTTP 500"):
            await client.get_trace("t-1")


async def test_rollup_client_raises_unreachable_on_malformed_json():
    """Invalid JSON from the rollup funnels through the unreachable path.

    Otherwise a ValueError would bubble past FederatedTracer's
    fallback-to-local branch and break reads any time the rollup
    misbehaves.
    """
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        with pytest.raises(RollupUnreachableError, match="malformed JSON"):
            await client.get_trace("t-1")


async def test_rollup_client_raises_unreachable_on_non_object_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "list"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        with pytest.raises(RollupUnreachableError, match="non-object"):
            await client.get_trace("t-1")


async def test_rollup_client_raises_unreachable_on_non_list_spans():
    """Non-list ``spans`` (e.g., dict) must funnel through the fallback.

    Silently iterating a dict yields keys, which would skip every row
    and pretend the trace was empty — worst of both worlds. The
    unreachable error routes us through FederatedTracer's local-only
    warning path instead.
    """
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"spans": {"wrong": "shape"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        with pytest.raises(RollupUnreachableError, match="expected list"):
            await client.get_trace("t-1")


async def test_rollup_client_tolerates_corrupt_ended_at():
    """A bad ``ended_at`` on one row must not break the whole read."""
    bad_row = _worker_span("w-bad", trace_id="t-1", parent_id="p")
    bad_row["ended_at"] = "not-a-datetime"
    good_row = _worker_span("w-good", trace_id="t-1", parent_id="p")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "trace_id": "t-1", "spans": [bad_row, good_row], "sources": [],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        spans = await client.get_trace("t-1")

    # Both rows returned; bad row's ended_at is None (treated as open)
    # while good row parses normally.
    by_id = {s.id: s for s in spans}
    assert by_id["w-bad"].ended_at is None
    assert by_id["w-good"].ended_at is not None


async def test_rollup_client_skips_malformed_rows():
    transport = _rollup_transport({
        "t-1": [
            {"missing_fields": True},  # skip
            _worker_span("w-1", trace_id="t-1", parent_id="p"),  # keep
        ],
    })
    async with httpx.AsyncClient(transport=transport) as http:
        client = RollupClient(base_url="http://willie:8902", http=http)
        spans = await client.get_trace("t-1")
    assert [s.id for s in spans] == ["w-1"]


# --- FederatedTracer ---


async def _seed_local(local: SQLiteTracer) -> tuple[str, str]:
    """Write a caller-side root + child span to ``local`` and flush.

    Returns (trace_id, root_span_id) so tests can reference them.
    """
    root = local.start_trace("agent")
    child = local.start_span(root.id, SpanKind.TOOL_CALL, "echo", {"text": "hi"})
    local.end_span(child.id, output="hi")
    local.end_span(root.id, output={"output": "done", "scores": None})
    await local.flush()
    return root.trace_id, root.id


async def test_federated_returns_local_only_when_no_rollup(local):
    trace_id, root_id = await _seed_local(local)
    federated = FederatedTracer(local=local, rollup=None)
    spans = await federated.get_trace(trace_id)
    assert len(spans) == 2
    assert root_id in {s.id for s in spans}


async def test_federated_unions_rollup_spans_into_local_trace(local):
    trace_id, root_id = await _seed_local(local)
    transport = _rollup_transport({
        trace_id: [
            _worker_span(
                "w-1", trace_id=trace_id, parent_id=root_id,
                name="task:inference",
                started_at="2026-04-18T00:00:10",
            ),
        ],
    })
    async with httpx.AsyncClient(transport=transport) as http:
        rollup = RollupClient(base_url="http://willie:8902", http=http)
        federated = FederatedTracer(local=local, rollup=rollup)
        spans = await federated.get_trace(trace_id)
        await rollup.close()

    ids = [s.id for s in spans]
    assert "w-1" in ids
    assert root_id in ids
    # Spans sorted by started_at — local spans were written first.
    starts = [s.started_at for s in spans]
    assert starts == sorted(starts)

    # Worker span reparents under the local root (proving phase 9's
    # propagation invariant end-to-end).
    worker = next(s for s in spans if s.id == "w-1")
    assert worker.parent_id == root_id


async def test_federated_falls_back_to_local_on_rollup_failure(local, caplog):
    trace_id, root_id = await _seed_local(local)
    transport = _rollup_transport(raise_connect_error=True)
    async with httpx.AsyncClient(transport=transport) as http:
        rollup = RollupClient(base_url="http://willie:8902", http=http)
        federated = FederatedTracer(local=local, rollup=rollup)
        with caplog.at_level("WARNING"):
            spans = await federated.get_trace(trace_id)
        await rollup.close()
    # Local spans returned; rollup failure logged but didn't raise.
    assert len(spans) == 2
    assert root_id in {s.id for s in spans}
    assert any("rollup unavailable" in r.message for r in caplog.records)


async def test_federated_list_traces_delegates_to_local(local):
    """Root AGENT_RUN spans are caller-side only; no rollup needed."""
    await _seed_local(local)
    federated = FederatedTracer(local=local, rollup=None)
    roots = await federated.list_traces()
    assert len(roots) == 1
    assert roots[0].kind == SpanKind.AGENT_RUN


async def test_federated_dedups_when_local_and_rollup_overlap(local):
    """Local wins on duplicate ids — worker metadata-only row is dropped."""
    trace_id, root_id = await _seed_local(local)
    # Rollup returns a span with the same id as the local root. That
    # shouldn't happen in practice (ids are uuid4) but the dedup must
    # be deterministic.
    transport = _rollup_transport({
        trace_id: [
            {
                "id": root_id,
                "trace_id": trace_id,
                "parent_id": "some-other-parent",
                "kind": "task_run",
                "name": "task:collision",
                "started_at": "2099-01-01T00:00:00",
                "ended_at": None,
                "duration_ms": None,
                "input": None,
                "output": None,
                "metadata": None,
                "error": None,
                "usage_input_tokens": None,
                "usage_output_tokens": None,
                "usage_cost": None,
                "source_machine": "mcbain",
            },
        ],
    })
    async with httpx.AsyncClient(transport=transport) as http:
        rollup = RollupClient(base_url="http://willie:8902", http=http)
        federated = FederatedTracer(local=local, rollup=rollup)
        spans = await federated.get_trace(trace_id)
        await rollup.close()

    # Local root kept (AGENT_RUN kind), not overwritten by worker's task_run.
    root = next(s for s in spans if s.id == root_id)
    assert root.kind == SpanKind.AGENT_RUN
    assert root.name == "agent"
