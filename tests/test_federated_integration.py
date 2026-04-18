"""End-to-end: jig-side federated view unifies local + rollup spans.

Closes phase 9 from the jig side. Simulates the full shape of a
cross-machine trace: jig writes AGENT_RUN + caller-tool spans to its
local SQLite; a mocked rollup returns worker TASK_RUN + LLM_CALL spans
whose ``parent_id`` points back into the jig-side span tree.

:meth:`FederatedTracer.get_trace` must return the union with parent
links intact so :func:`jig.trace_diff` and :func:`jig.replay` can
operate on the combined view.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from jig import SpanKind, trace_diff
from jig.tracing import FederatedTracer, RollupClient, SQLiteTracer


def _worker_row(
    sid: str,
    *,
    trace_id: str,
    parent_id: str,
    kind: str,
    name: str,
    started_at: str,
    source_machine: str = "mcbain",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    return {
        "id": sid,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "kind": kind,
        "name": name,
        "started_at": started_at,
        "ended_at": None,
        "duration_ms": 5.0,
        "input": None,
        "output": None,
        "metadata": None,
        "error": None,
        "usage_input_tokens": input_tokens,
        "usage_output_tokens": output_tokens,
        "usage_cost": None,
        "source_machine": source_machine,
    }


@pytest.fixture
async def local(tmp_path: Path):
    t = SQLiteTracer(db_path=str(tmp_path / "local.db"))
    yield t
    await t.close()


async def _seed_dispatched_agent_run(local: SQLiteTracer) -> tuple[str, str, str]:
    """Write the jig-side half of a dispatched agent run.

    Returns ``(trace_id, root_span_id, dispatch_tool_span_id)`` so the
    test can attach worker spans under ``dispatch_tool_span_id`` —
    simulating smithers having picked up the dispatched tool call.
    """
    root = local.start_trace("agent")
    dispatch = local.start_span(
        root.id,
        SpanKind.TOOL_CALL,
        "run_backtest",  # a dispatch=True tool
        {"ticker": "AAPL"},
    )
    # The caller-side tool span stays open until the worker finishes;
    # in real dispatch the tracer ends it when the Job completes. For
    # the test we end it with a dispatched-marker output.
    local.end_span(dispatch.id, output="[dispatched]")
    local.end_span(root.id, output={"output": "done", "scores": None})
    await local.flush()
    return root.trace_id, root.id, dispatch.id


async def test_federated_view_chains_caller_task_and_child_spans(local):
    """Worker-origin parent chain must survive into jig's merged trace.

    Shape the test exercises:

        agent (AGENT_RUN, local)
          └── run_backtest (TOOL_CALL, local)          ← caller-side
                └── task:function (TASK_RUN, worker)   ← rollup
                      └── fn:run_backtest (TOOL_CALL, worker)
    """
    trace_id, root_id, dispatch_id = await _seed_dispatched_agent_run(local)

    # The rollup returns a task span reparented on the dispatch tool
    # span, and a function span reparented on the task span. Span ids
    # are synthetic — the worker would have generated uuid4 values in
    # production.
    task_span_id = "worker-task-1"
    worker_rows = [
        _worker_row(
            task_span_id,
            trace_id=trace_id,
            parent_id=dispatch_id,
            kind="task_run",
            name="task:function",
            started_at="2026-04-18T00:00:10",
        ),
        _worker_row(
            "worker-fn-1",
            trace_id=trace_id,
            parent_id=task_span_id,
            kind="tool_call",
            name="fn:run_backtest",
            started_at="2026-04-18T00:00:11",
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "trace_id": trace_id,
            "spans": worker_rows,
            "sources": [{"machine": "mcbain", "status": "ok", "count": 2}],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rollup = RollupClient(base_url="http://willie:8902", http=http)
        federated = FederatedTracer(local=local, rollup=rollup)

        spans = await federated.get_trace(trace_id)
        await rollup.close()

    # Build id → span map for linkage assertions.
    by_id = {s.id: s for s in spans}

    # Every span we wrote or received is in the merged view.
    assert {root_id, dispatch_id, task_span_id, "worker-fn-1"}.issubset(by_id)

    # Walk the parent chain up from the deepest worker span to the
    # caller-side root, proving end-to-end propagation.
    fn_span = by_id["worker-fn-1"]
    task_span = by_id[fn_span.parent_id]
    caller_tool = by_id[task_span.parent_id]
    root = by_id[caller_tool.parent_id]
    assert task_span.id == task_span_id
    assert caller_tool.id == dispatch_id
    assert root.id == root_id
    assert root.parent_id is None  # true agent root

    # Spans sorted by started_at, so the caller-side ones come first.
    starts = [s.started_at for s in spans]
    assert starts == sorted(starts)


async def test_trace_diff_operates_on_federated_view(local):
    """Concrete payoff: jig.trace_diff over federated traces works.

    Two traces differ only in worker-side fn span output. The diff
    should surface that divergence even though the differing spans
    live on the remote rollup, not local SQLite.
    """
    # Build trace A
    trace_id_a, _root_a, disp_a = await _seed_dispatched_agent_run(local)
    # Build trace B (separate root span in local)
    trace_id_b, _root_b, disp_b = await _seed_dispatched_agent_run(local)

    def _worker_tool(
        trace_id: str,
        parent: str,
        *,
        output: str,
        started_at: str,
    ) -> dict:
        row = _worker_row(
            f"w-{trace_id}-fn",
            trace_id=trace_id,
            parent_id=parent,
            kind="tool_call",
            name="fn:run_backtest",
            started_at=started_at,
        )
        row["output"] = output
        return row

    worker_by_trace = {
        trace_id_a: [
            _worker_tool(
                trace_id_a, disp_a,
                output='"$4200 profit"',
                started_at="2026-04-18T00:00:10",
            ),
        ],
        trace_id_b: [
            _worker_tool(
                trace_id_b, disp_b,
                output='"$200 profit"',
                started_at="2026-04-18T00:00:10",
            ),
        ],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        tid = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={
            "trace_id": tid,
            "spans": worker_by_trace.get(tid, []),
            "sources": [],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rollup = RollupClient(base_url="http://willie:8902", http=http)
        federated = FederatedTracer(local=local, rollup=rollup)

        diff = await trace_diff(trace_id_a, trace_id_b, tracer=federated)
        await rollup.close()

    # Both traces have identical caller-tool spans (same args/output
    # "[dispatched]") and identical final agent output, so the *only*
    # divergence is on the worker-side fn:run_backtest output — which
    # the federated view surfaced to trace_diff.
    assert diff.identical is False
    assert len(diff.tool_divergence) >= 1
    # Find the divergence on the worker's fn span. Its output will
    # differ between the two traces.
    fn_divergences = [
        d for d in diff.tool_divergence
        if d.a is not None and d.a.name == "fn:run_backtest"
    ]
    assert fn_divergences, diff.tool_divergence
    d = fn_divergences[0]
    assert d.a.output != d.b.output
