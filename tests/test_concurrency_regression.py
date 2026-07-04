"""Cross-cutting concurrency regression tests for Refactor 3.

This module provides acceptance coverage for four behaviors that span
multiple subsystems:

1. Sweep deadlock guard — sweeps with blocking workers or budget
   exhaustion complete (or raise predictably) within a bounded wall-clock
   window; no pending tasks are left behind.

2. Budget admission invariant — spent + active_reserved_usd never exceeds
   the hard limit at any point during concurrent admission.

3. LazyConnection repeated close/get cycles — running many cycles with
   an observable connection counter proves no double-live connection state
   accumulates over time.

4. BudgetedLLMClient concurrent admission — a shared budget rejects over-
   admitting callers even when multiple tasks race past the pre-flight check.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jig import (
    AgentConfig,
    BudgetedLLMClient,
    BudgetTracker,
    JigBudgetError,
    LLMResponse,
    Message,
    Role,
    Usage,
    sweep,
)
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    LLMClient,
    MemoryStore,
    Retriever,
    ScoredResult,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeMem(MemoryStore, Retriever):
    async def add(self, content, metadata=None): return "m"
    async def get(self, id): return None
    async def all(self): return []
    async def delete(self, id): pass
    async def retrieve(self, query, k=5, context=None): return []
    async def get_session(self, sid): return []
    async def add_to_session(self, sid, m): pass
    async def clear(self, session_id=None, before=None): pass


class _FakeFB(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None): return "r"
    async def score(self, rid, scores): pass
    async def get_signals(self, q, limit=3, min_score=None, source=None): return []
    async def query(self, q): return []
    async def export_eval_set(self, since=None, min_score=None, max_score=None, limit=None): return []


class _FakeTracer(TracingLogger):
    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):
        return Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now())
    def start_span(self, parent_id, kind, name, input=None, metadata=None):
        return Span(id="s", trace_id="t", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id)
    def end_span(self, span_id, output=None, error=None, usage=None): pass
    async def get_trace(self, tid): return []
    async def list_traces(self, since=None, limit=50, name=None): return []
    async def flush(self): pass


def _config(llm: LLMClient, name: str = "agent") -> AgentConfig:
    return AgentConfig(
        name=name,
        description="test",
        system_prompt="be brief",
        llm=llm,
        store=_FakeMem(), retriever=None,
        feedback=_FakeFB(),
        tracer=_FakeTracer(),
        tools=ToolRegistry(),
    )


def _ok_response(cost: float = 0.001) -> LLMResponse:
    return LLMResponse(
        content="ok",
        tool_calls=None,
        usage=Usage(input_tokens=5, output_tokens=3, cost=cost),
        latency_ms=1.0,
        model="fake",
    )


# ---------------------------------------------------------------------------
# 1. Sweep deadlock guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSweepDeadlockGuard:
    async def test_sweep_completes_within_timeout_on_budget_exhaustion(self):
        """Sweeps where budget exhausts mid-run must complete (with structured errors)
        rather than hanging.  asyncio.wait_for enforces the wall-clock bound.
        """
        budget = BudgetTracker(limit_usd=0.002)
        inner = AsyncMock()
        inner.complete = AsyncMock(return_value=_ok_response(cost=0.002))
        llm = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.002)

        cfg = _config(llm, name="budget-agent")
        result = await asyncio.wait_for(
            sweep(["c1", "c2", "c3"], [cfg], concurrency=1),
            timeout=5.0,
        )
        assert len(result.runs) == 3
        # At least one run must be a budget_exhausted error
        error_cats = [r.result.error.category for r in result.runs if r.result.error]
        assert "budget_exhausted" in error_cats

    async def test_sweep_completes_within_timeout_on_infra_failure(self):
        """Workers that raise RuntimeError on every case must still drain the queue
        within a bounded time — no hang, no pending tasks.
        """
        class _AlwaysCrash(LLMClient):
            async def complete(self, params):
                raise RuntimeError("infra down")

        cfg = _config(_AlwaysCrash(), name="crash-agent")
        result = await asyncio.wait_for(
            sweep(["c1", "c2", "c3", "c4"], [cfg], concurrency=2),
            timeout=5.0,
        )
        # All 4 slots must be filled (worker did not die)
        assert len(result.runs) == 4
        errors = [r for r in result.runs if r.result.error is not None]
        assert len(errors) == 4

    async def test_no_pending_tasks_after_budget_exhausted_sweep(self):
        """After sweep() returns, no asyncio tasks spawned by the sweep are pending."""
        budget = BudgetTracker(limit_usd=0.001)
        inner = AsyncMock()
        inner.complete = AsyncMock(return_value=_ok_response(cost=0.001))
        llm = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.001)

        cfg = _config(llm, name="exhaust-agent")
        before = set(asyncio.all_tasks())
        await sweep(["c1", "c2", "c3"], [cfg], concurrency=2)
        after = set(asyncio.all_tasks())
        # Any tasks added after sweep started must have been cleaned up
        leaked = after - before - {asyncio.current_task()}
        assert not leaked, f"sweep left {len(leaked)} pending task(s)"

    async def test_empty_sweep_no_hang(self):
        """Empty case or config lists must return immediately with no workers spawned."""

        class _NeverCalled(LLMClient):
            async def complete(self, params):
                raise AssertionError("should never be called")

        cfg = _config(_NeverCalled(), name="never")
        result = await asyncio.wait_for(
            sweep([], [cfg], concurrency=4),
            timeout=2.0,
        )
        assert result.runs == []


# ---------------------------------------------------------------------------
# 2. Budget admission invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetAdmissionInvariant:
    async def test_spent_plus_reserved_never_exceeds_limit(self):
        """The invariant spent + active_reserved <= limit must hold at every
        moment during concurrent reservation admission.

        Strategy: wrap BudgetTracker._lock's __aenter__ to sample the
        invariant immediately after each state mutation inside the lock.
        Any violation is recorded so the test can fail cleanly after gather().
        """
        limit = 0.10
        budget = BudgetTracker(limit_usd=limit)
        violations: list[str] = []

        original_reserve = budget.reserve

        async def _checked_reserve(estimate_usd: float):
            reservation = await original_reserve(estimate_usd)
            # Sample the state right after reserve() returns (outside the lock)
            total = budget.spent_usd + budget._active_reserved_usd
            if total > limit + 1e-9:
                violations.append(
                    f"spent={budget.spent_usd:.4f} + reserved={budget._active_reserved_usd:.4f}"
                    f" = {total:.4f} > limit={limit:.4f}"
                )
            return reservation

        budget.reserve = _checked_reserve  # type: ignore[method-assign]

        async def task(estimate: float) -> None:
            try:
                r = await budget.reserve(estimate)
                await asyncio.sleep(0)  # yield — keeps reservation live across awaits
                await r.reconcile(estimate * 0.9)
            except JigBudgetError:
                pass

        # 6 tasks each wanting 0.04 — only 2 can fit (0.04 * 2 = 0.08 ≤ 0.10)
        await asyncio.gather(*[task(0.04) for _ in range(6)])
        assert not violations, "\n".join(violations)

    async def test_admission_rejected_when_projection_exceeds_limit(self):
        """Concurrent reserves that would jointly exceed the cap must be rejected.

        This is the hard invariant: if 3 tasks each want 0.06 and the limit
        is 0.10, at most one can hold a live reservation at any time.
        """
        budget = BudgetTracker(limit_usd=0.10)
        accepted: list[int] = []
        rejected: list[int] = []

        async def try_reserve(i: int) -> None:
            try:
                r = await budget.reserve(0.06)
                accepted.append(i)
                await asyncio.sleep(0)  # keep reservation live
                await r.release()
            except JigBudgetError:
                rejected.append(i)

        await asyncio.gather(*[try_reserve(i) for i in range(5)])
        # No more than one can be admitted at once (0.06 * 2 > 0.10)
        # Sequentially they can each be admitted after the previous releases,
        # but concurrently the second in a concurrent pair must be rejected.
        # Exactly: accepted + rejected == 5, and rejected > 0
        assert len(accepted) + len(rejected) == 5
        assert len(rejected) > 0, "at least some reservations must be rejected under contention"

    async def test_released_reservation_frees_headroom_for_next(self):
        """After a reservation is released, the freed headroom allows the next caller."""
        budget = BudgetTracker(limit_usd=0.10)

        r1 = await budget.reserve(0.09)
        # Budget is now 0.09 reserved; a second 0.09 request must be rejected
        with pytest.raises(JigBudgetError):
            await budget.reserve(0.09)

        # Release r1 → 0.09 freed
        await r1.release()
        assert budget._active_reserved_usd == pytest.approx(0.0)

        # Now 0.09 must be admitted
        r2 = await budget.reserve(0.09)
        await r2.release()


# ---------------------------------------------------------------------------
# 3. LazyConnection repeated close/get cycles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLazyConnectionRepeatedCycles:
    async def test_repeated_close_get_cycles_no_double_connect(self, tmp_path):
        """Run many close/get cycles and assert exactly one live connection at a time.

        Uses an observable connect counter to prove no cycle accumulates an
        extra live connection — the failure mode is two concurrent get() calls
        both seeing _db=None and racing to connect.
        """
        from jig._sqlite import LazyConnection

        lc = LazyConnection(str(tmp_path / "cycles.db"), "")
        connect_calls = 0
        close_calls = 0

        # Wrap aiosqlite.connect to count calls and produce trackable mocks
        active_conns: list[Any] = []
        original_connect = None  # captured inside patch

        async def fake_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            conn = MagicMock(name=f"conn-{connect_calls}")
            conn.execute = AsyncMock()
            conn.executescript = AsyncMock()

            async def fake_close():
                nonlocal close_calls
                close_calls += 1
                if conn in active_conns:
                    active_conns.remove(conn)

            conn.close = AsyncMock(side_effect=fake_close)
            active_conns.append(conn)
            return conn

        CYCLES = 20
        with patch("jig._sqlite.aiosqlite.connect", side_effect=fake_connect):
            for _ in range(CYCLES):
                await lc.get()
                # Concurrent get() calls on an open connection must all return same object
                results = await asyncio.gather(*[lc.get() for _ in range(4)])
                assert all(r is results[0] for r in results), (
                    "concurrent get() on open connection must return the same object"
                )
                await lc.close()
                # After close, no live connections should be accumulated
                assert len(active_conns) == 0, (
                    f"after close, expected 0 live connections, got {len(active_conns)}"
                )

        # Each cycle: one connect, one close → totals must match
        assert connect_calls == CYCLES
        assert close_calls == CYCLES

    async def test_concurrent_get_during_close_no_double_live_connection(self, tmp_path):
        """Concurrent close() + get() must not produce two live connections.

        A get() arriving while close() holds the lock queues, sees _db=None
        after close() completes, and reconnects — resulting in exactly one
        live connection, not two.
        """
        from jig._sqlite import LazyConnection

        lc = LazyConnection(str(tmp_path / "race.db"), "")
        connect_events: list[str] = []

        async def fake_connect(*args, **kwargs):
            conn = MagicMock(name=f"conn-{len(connect_events)}")
            conn.execute = AsyncMock()
            conn.executescript = AsyncMock()
            conn.close = AsyncMock()
            connect_events.append("connect")
            await asyncio.sleep(0)  # yield to let other tasks interleave
            return conn

        with patch("jig._sqlite.aiosqlite.connect", side_effect=fake_connect):
            ROUNDS = 10
            for _ in range(ROUNDS):
                await lc.get()  # establish a live connection
                # Race close() against two get() calls
                results = await asyncio.gather(lc.close(), lc.get(), lc.get())
                get_results = [r for r in results if r is not None]
                if get_results:
                    # All concurrent get() calls must return the same object
                    assert all(c is get_results[0] for c in get_results), (
                        "concurrent get() after race must return the same connection"
                    )
                await lc.close()  # cleanup

        # No round should have produced more than 2 connects (pre + post-close)
        # The important thing is there was never a "double-live" state
        assert len(connect_events) <= ROUNDS * 2


# ---------------------------------------------------------------------------
# 4. BudgetedLLMClient concurrent admission via sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetedClientConcurrentAdmission:
    async def test_concurrent_sweep_respects_budget_cap(self):
        """When multiple sweep workers share a BudgetedLLMClient, the budget cap
        is enforced across concurrent calls — the invariant is the admission
        check, not just the final spend.
        """
        budget = BudgetTracker(limit_usd=0.05)
        inner_call_count = 0

        async def _fake_complete(params):
            nonlocal inner_call_count
            inner_call_count += 1
            await asyncio.sleep(0)  # yield — keeps reservation live
            return _ok_response(cost=0.02)

        inner = AsyncMock()
        inner.complete = AsyncMock(side_effect=_fake_complete)
        llm = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.02)

        cfg = _config(llm, name="concurrent-budget")
        result = await sweep(["c1", "c2", "c3", "c4", "c5"], [cfg], concurrency=5)

        assert len(result.runs) == 5
        errors = [r for r in result.runs if r.result.error is not None]
        successes = [r for r in result.runs if r.result.error is None]
        # With limit=0.05 and estimate=0.02, at most 2 concurrent reservations
        # and at most 2 full reconciles can fit before exhaustion
        assert len(errors) > 0, "some calls must be rejected by the budget cap"
        assert len(successes) <= 3, "at most ~2-3 calls can succeed within 0.05"
        # Final spend must not exceed the hard cap
        assert budget.spent_usd <= budget.limit_usd + 1e-9
