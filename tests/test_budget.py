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


@pytest.mark.asyncio
class TestBudgetedLLMClient:
    async def test_tallies_on_every_call(self):
        inner = AsyncMock()
        inner.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                tool_calls=None,
                usage=Usage(input_tokens=100, output_tokens=50, cost=0.02),
                latency_ms=1.0,
                model="claude-sonnet-4-5",
            )
        )
        budget = BudgetTracker(limit_usd=0.10)
        client = BudgetedLLMClient(inner=inner, budget=budget)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        await client.complete(params)
        await client.complete(params)
        assert budget.spent_usd == pytest.approx(0.04)

    async def test_raises_when_exceeded(self):
        inner = AsyncMock()
        inner.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                tool_calls=None,
                usage=Usage(input_tokens=100, output_tokens=50, cost=0.08),
                latency_ms=1.0,
                model="claude-sonnet-4-5",
            )
        )
        budget = BudgetTracker(limit_usd=0.10)
        client = BudgetedLLMClient(inner=inner, budget=budget)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        await client.complete(params)
        with pytest.raises(JigBudgetError):
            await client.complete(params)

    async def test_preflight_check_raises_before_call(self):
        inner = AsyncMock()
        inner.complete = AsyncMock()
        budget = BudgetTracker(limit_usd=1.0)
        budget.spent_usd = 2.0  # already over
        client = BudgetedLLMClient(inner=inner, budget=budget)

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(JigBudgetError):
            await client.complete(params)
        inner.complete.assert_not_called()
