"""Tests for BudgetTracker and BudgetedLLMClient."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from jig import BudgetedLLMClient, BudgetTracker, JigBudgetError
from jig.core.types import CompletionParams, LLMResponse, Message, Role, Usage


class TestBudgetTracker:
    def test_records_cost(self):
        budget = BudgetTracker(limit_usd=1.0)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.05))
        assert budget.spent_usd == 0.05
        assert budget.remaining_usd == 0.95

    def test_exceeds_limit_raises(self):
        budget = BudgetTracker(limit_usd=0.10)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.08))
        with pytest.raises(JigBudgetError) as exc_info:
            budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.05))
        assert exc_info.value.spent_usd == pytest.approx(0.13)
        assert exc_info.value.limit_usd == 0.10

    def test_none_cost_ignored(self):
        budget = BudgetTracker(limit_usd=1.0)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=None))
        assert budget.spent_usd == 0.0

    def test_zero_cost_recorded_but_harmless(self):
        # Ollama usages carry cost=0.0; budget should tally without raising
        budget = BudgetTracker(limit_usd=1.0)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.0))
        assert budget.spent_usd == 0.0

    def test_remaining_clamped_to_zero(self):
        budget = BudgetTracker(limit_usd=1.0)
        # Push spent past limit without triggering check
        budget.spent_usd = 2.0
        assert budget.remaining_usd == 0.0

    def test_reset(self):
        budget = BudgetTracker(limit_usd=1.0)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.5))
        budget.reset()
        assert budget.spent_usd == 0.0

    @pytest.mark.asyncio
    async def test_reset_clears_active_reservations(self):
        budget = BudgetTracker(limit_usd=1.0)
        await budget.reserve(0.75)
        budget.record(Usage(input_tokens=100, output_tokens=50, cost=0.2))

        budget.reset()

        assert budget.spent_usd == 0.0
        assert budget._active_reserved_usd == 0.0
        reservation = await budget.reserve(1.0)
        await reservation.release()

    def test_invalid_limit_raises(self):
        with pytest.raises(ValueError):
            BudgetTracker(limit_usd=0)
        with pytest.raises(ValueError):
            BudgetTracker(limit_usd=-1.0)

    def test_check_raises_when_over(self):
        budget = BudgetTracker(limit_usd=1.0)
        budget.spent_usd = 2.0
        with pytest.raises(JigBudgetError):
            budget.check()

    def test_negative_cost_rejected(self):
        """Negative cost is always a bug — reject rather than allow bypass."""
        budget = BudgetTracker(limit_usd=1.0)
        budget.record(Usage(input_tokens=10, output_tokens=10, cost=0.5))
        with pytest.raises(ValueError, match="non-negative"):
            budget.record(Usage(input_tokens=10, output_tokens=10, cost=-0.1))
        # Tally is not modified by the rejected record
        assert budget.spent_usd == 0.5

    def test_nan_cost_rejected(self):
        """NaN bypasses <0 and >limit comparisons — reject outright."""
        budget = BudgetTracker(limit_usd=1.0)
        with pytest.raises(ValueError, match="finite"):
            budget.record(Usage(input_tokens=10, output_tokens=10, cost=float("nan")))

    def test_inf_cost_rejected(self):
        """Inf would poison spent_usd — reject outright."""
        budget = BudgetTracker(limit_usd=1.0)
        with pytest.raises(ValueError, match="finite"):
            budget.record(Usage(input_tokens=10, output_tokens=10, cost=float("inf")))

    def test_nan_limit_rejected(self):
        """NaN limit would disable enforcement — reject at construction."""
        with pytest.raises(ValueError, match="finite"):
            BudgetTracker(limit_usd=float("nan"))

    def test_inf_limit_rejected(self):
        """Inf limit effectively disables the cap — reject."""
        with pytest.raises(ValueError, match="finite"):
            BudgetTracker(limit_usd=float("inf"))


@pytest.mark.asyncio
class TestBudgetTrackerReservations:
    async def test_reserve_claims_budget_before_call(self):
        budget = BudgetTracker(limit_usd=0.10)
        reservation = await budget.reserve(0.06)
        # Second reservation would push projected to 0.12, over the cap
        with pytest.raises(JigBudgetError):
            await budget.reserve(0.06)
        await reservation.release()

    async def test_reconcile_charges_actual_cost(self):
        budget = BudgetTracker(limit_usd=1.0)
        reservation = await budget.reserve(0.10)
        assert budget._active_reserved_usd == pytest.approx(0.10)
        await reservation.reconcile(0.07)
        assert budget.spent_usd == pytest.approx(0.07)
        assert budget._active_reserved_usd == pytest.approx(0.0)

    async def test_reconcile_charges_estimate_when_actual_unknown(self):
        budget = BudgetTracker(limit_usd=1.0)
        reservation = await budget.reserve(0.10)
        await reservation.reconcile(None)
        assert budget.spent_usd == pytest.approx(0.10)

    async def test_release_frees_reservation_without_cost(self):
        budget = BudgetTracker(limit_usd=0.10)
        reservation = await budget.reserve(0.08)
        await reservation.release()
        assert budget.spent_usd == pytest.approx(0.0)
        assert budget._active_reserved_usd == pytest.approx(0.0)
        # Budget is available again
        r2 = await budget.reserve(0.08)
        await r2.release()

    async def test_release_is_idempotent(self):
        budget = BudgetTracker(limit_usd=1.0)
        reservation = await budget.reserve(0.10)
        await reservation.release()
        await reservation.release()  # second call must not double-subtract
        assert budget._active_reserved_usd == pytest.approx(0.0)

    async def test_reconcile_is_idempotent(self):
        budget = BudgetTracker(limit_usd=1.0)
        reservation = await budget.reserve(0.10)
        await reservation.reconcile(0.07)
        await reservation.reconcile(0.07)  # second call must not double-charge
        assert budget.spent_usd == pytest.approx(0.07)

    async def test_reconcile_raises_when_actual_exceeds_cap(self):
        budget = BudgetTracker(limit_usd=0.10)
        reservation = await budget.reserve(0.09)
        # Actual cost exceeds the cap
        with pytest.raises(JigBudgetError):
            await reservation.reconcile(0.15)
        # Reservation was settled; active reserved is cleared
        assert budget._active_reserved_usd == pytest.approx(0.0)

    async def test_concurrent_reservations_respect_cap(self):
        """Two concurrent reserves that together exceed the cap — only one admitted.

        The coroutines must hold their reservation across an await point (the
        simulated provider call) so the lock can observe overlapping reservations.
        Without the yield, one coroutine finishes before the other starts and
        both would succeed sequentially.
        """
        import asyncio

        budget = BudgetTracker(limit_usd=0.10)
        results: list[bool] = []

        async def try_reserve() -> None:
            try:
                r = await budget.reserve(0.07)
                results.append(True)
                await asyncio.sleep(0)  # yield — simulates in-flight provider call
                await r.release()
            except JigBudgetError:
                results.append(False)

        await asyncio.gather(try_reserve(), try_reserve())
        # Exactly one reservation admitted, one rejected
        assert results.count(True) == 1
        assert results.count(False) == 1


@pytest.mark.asyncio
class TestBudgetedLLMClient:
    def _make_inner(self, cost: float = 0.02) -> AsyncMock:
        inner = AsyncMock()
        inner.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                tool_calls=None,
                usage=Usage(input_tokens=100, output_tokens=50, cost=cost),
                latency_ms=1.0,
                model="claude-sonnet-4-5",
            )
        )
        return inner

    async def test_tallies_on_every_call(self):
        inner = self._make_inner(cost=0.02)
        budget = BudgetTracker(limit_usd=0.10)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        await client.complete(params)
        await client.complete(params)
        assert budget.spent_usd == pytest.approx(0.04)

    async def test_raises_when_exceeded(self):
        inner = self._make_inner(cost=0.08)
        budget = BudgetTracker(limit_usd=0.10)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        await client.complete(params)
        with pytest.raises(JigBudgetError):
            await client.complete(params)

    async def test_preflight_check_raises_before_call(self):
        inner = AsyncMock()
        inner.complete = AsyncMock()
        budget = BudgetTracker(limit_usd=1.0)
        budget.spent_usd = 2.0  # already over
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(JigBudgetError):
            await client.complete(params)
        inner.complete.assert_not_called()

    async def test_no_estimate_raises_before_call(self):
        """No estimate_cost_usd → JigBudgetError before provider call."""
        inner = AsyncMock()
        inner.complete = AsyncMock()
        budget = BudgetTracker(limit_usd=1.0)
        client = BudgetedLLMClient(inner=inner, budget=budget)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(JigBudgetError, match="estimate_cost_usd"):
            await client.complete(params)
        inner.complete.assert_not_called()

    async def test_reservation_released_on_provider_failure(self):
        """If the provider call raises, the reservation is freed."""
        inner = AsyncMock()
        inner.complete = AsyncMock(side_effect=RuntimeError("network error"))
        budget = BudgetTracker(limit_usd=0.10)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.08)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(RuntimeError):
            await client.complete(params)

        # Reservation must be released — budget is not stranded
        assert budget._active_reserved_usd == pytest.approx(0.0)
        assert budget.spent_usd == pytest.approx(0.0)
        # A subsequent call can still be admitted
        inner.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok", tool_calls=None,
                usage=Usage(10, 10, cost=0.05),
                latency_ms=1.0, model="x",
            )
        )
        await client.complete(params)
        assert budget.spent_usd == pytest.approx(0.05)

    async def test_aclose_delegates_to_inner(self):
        inner = AsyncMock()
        inner.aclose = AsyncMock()
        budget = BudgetTracker(limit_usd=1.0)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)
        await client.aclose()
        inner.aclose.assert_called_once()

    async def test_aclose_no_op_when_inner_lacks_aclose(self):
        inner = AsyncMock(spec=[])  # no aclose attribute
        budget = BudgetTracker(limit_usd=1.0)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)
        await client.aclose()  # must not raise

    async def test_stream_fails_closed(self):
        """Streaming can't be enforced until usage is surfaced — refuse."""
        inner = AsyncMock()
        budget = BudgetTracker(limit_usd=1.0)
        client = BudgetedLLMClient(inner=inner, budget=budget, estimate_cost_usd=0.01)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(NotImplementedError, match="streaming"):
            async for _ in client.stream(params):
                pass
