"""Budget tracking for experimentation — caps total LLM spend.

Usage::

    budget = BudgetTracker(limit_usd=5.0)
    response = await client.complete(params)
    budget.record(response.usage)   # raises JigBudgetError past the cap

Or wrap a client so every call is tallied automatically::

    client = BudgetedLLMClient(inner=AnthropicClient(), budget=budget)
"""
from __future__ import annotations

import math
from typing import AsyncIterator

from jig.core.errors import JigBudgetError
from jig.core.types import CompletionParams, LLMClient, LLMResponse, Usage


class BudgetTracker:
    """Running USD tally with a hard cap.

    Usages that lack a ``cost`` field (unknown model, unpriced provider) are
    ignored rather than estimated — we prefer no signal over wrong signal.
    Call :meth:`record` after every completion; it raises when the tally
    exceeds ``limit_usd``. Callers that want a pre-flight check (e.g. before
    a sweep fan-out) can call :meth:`check`.
    """

    def __init__(self, limit_usd: float) -> None:
        # Non-finite limits (NaN/Inf) make every comparison with spent_usd
        # produce False, which would silently disable the cap.
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError(
                f"limit_usd must be a positive finite number, got {limit_usd!r}"
            )
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    def record(self, usage: Usage) -> None:
        if usage.cost is None:
            return
        # NaN bypasses <0 and >limit checks both; Inf poisons spent_usd.
        if not math.isfinite(usage.cost):
            raise ValueError(
                f"usage.cost must be finite, got {usage.cost!r}"
            )
        if usage.cost < 0:
            raise ValueError(
                f"usage.cost must be non-negative, got {usage.cost!r}"
            )
        self.spent_usd += usage.cost
        self.check()

    def check(self) -> None:
        if self.spent_usd > self.limit_usd:
            raise JigBudgetError(
                f"Budget exceeded: ${self.spent_usd:.4f} > ${self.limit_usd:.2f}",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
            )

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def reset(self) -> None:
        self.spent_usd = 0.0


class BudgetedLLMClient(LLMClient):
    """Wrap any ``LLMClient`` so every completion tallies against a budget."""

    def __init__(self, inner: LLMClient, budget: BudgetTracker) -> None:
        self._inner = inner
        self._budget = budget

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self._budget.check()
        response = await self._inner.complete(params)
        self._budget.record(response.usage)
        return response

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        # Streaming can't be budget-enforced because LLMClient.stream yields
        # only content chunks — no post-call Usage is surfaced, so spend
        # would silently escape the tally. Fail closed until the streaming
        # interface carries usage at completion.
        raise NotImplementedError(
            "BudgetedLLMClient.stream() is unsupported because the streaming "
            "interface does not surface usage for budget accounting. Use "
            "complete(), or wire a streaming interface that exposes final "
            "usage before re-enabling."
        )
        yield  # pragma: no cover — makes this a generator for the return type
