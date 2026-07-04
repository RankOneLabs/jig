"""Budget tracking for experimentation — caps total LLM spend.

Usage::

    budget = BudgetTracker(limit_usd=5.0)
    response = await client.complete(params)
    budget.record(response.usage)   # raises JigBudgetError past the cap

Or wrap a client so every call is tallied automatically::

    client = BudgetedLLMClient(
        inner=AnthropicClient(),
        budget=budget,
        estimate_cost_usd=0.01,  # conservative per-call reservation
    )

The wrapped client uses a reservation model: ``estimate_cost_usd`` is
reserved before each provider call and reconciled with the actual cost
afterwards, so concurrent calls cannot all pass a pre-flight check and
then jointly overshoot the cap.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import AsyncIterator

from jig.core.errors import JigBudgetError
from jig.core.types import CompletionParams, LLMClient, LLMResponse, Usage

logger = logging.getLogger(__name__)


class _BudgetReservation:
    """A single in-flight cost reservation held against a :class:`BudgetTracker`.

    Obtain via :meth:`BudgetTracker.reserve`. Call :meth:`reconcile` on
    success (replaces the reservation with actual cost) or :meth:`release`
    on failure (removes the reservation without recording spend). Both are
    idempotent after the first call.
    """

    def __init__(self, tracker: BudgetTracker, estimate_usd: float) -> None:
        self._tracker = tracker
        self._estimate_usd = estimate_usd
        self._settled = False

    async def reconcile(self, actual_usd: float | None) -> None:
        """Replace the reservation with actual cost.

        If ``actual_usd`` is ``None`` (unknown), the reserved estimate is
        charged instead — the only defensible cost already admitted by policy.
        """
        if self._settled:
            return
        cost = actual_usd if actual_usd is not None else self._estimate_usd
        if not math.isfinite(cost) or cost < 0:
            # Guard against bad values from providers without poisoning the tracker.
            cost = self._estimate_usd
        async with self._tracker._lock:
            self._tracker._active_reserved_usd -= self._estimate_usd
            self._tracker.spent_usd += cost
            self._settled = True
            self._tracker._check_unlocked()

    async def release(self) -> None:
        """Remove the reservation without recording any spend."""
        if self._settled:
            return
        async with self._tracker._lock:
            self._tracker._active_reserved_usd -= self._estimate_usd
            self._settled = True


class BudgetTracker:
    """Running USD tally with a hard cap.

    ``record`` / ``check`` provide the simple (non-concurrent) API. For
    concurrent usage, use :meth:`reserve` which returns a
    :class:`_BudgetReservation` — the reservation claims estimated cost
    before the provider call so concurrent callers cannot all pass a
    pre-flight check and jointly overshoot the cap.

    All state-mutating operations are protected by an ``asyncio.Lock`` so
    the tracker is safe to share across concurrent workers.
    """

    def __init__(self, limit_usd: float) -> None:
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError(
                f"limit_usd must be a positive finite number, got {limit_usd!r}"
            )
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._active_reserved_usd = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Simple (single-threaded / non-concurrent) API
    # ------------------------------------------------------------------

    def record(self, usage: Usage) -> None:
        if usage.cost is None:
            return
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

    def _check_unlocked(self) -> None:
        """Like ``check()`` but called while ``_lock`` is already held."""
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

    # ------------------------------------------------------------------
    # Reservation API (concurrent-safe)
    # ------------------------------------------------------------------

    async def reserve(self, estimate_usd: float) -> _BudgetReservation:
        """Claim ``estimate_usd`` against the budget before a provider call.

        Raises :class:`JigBudgetError` if admitting the reservation would
        push ``spent + active_reserved + estimate`` past the cap. The lock
        is held only for the short admission check — provider calls happen
        outside it.
        """
        if not math.isfinite(estimate_usd) or estimate_usd < 0:
            raise ValueError(
                f"estimate_usd must be a non-negative finite number, got {estimate_usd!r}"
            )
        async with self._lock:
            projected = self.spent_usd + self._active_reserved_usd + estimate_usd
            if projected > self.limit_usd:
                raise JigBudgetError(
                    f"Budget admission refused: projected ${projected:.4f} > "
                    f"${self.limit_usd:.2f} (spent=${self.spent_usd:.4f}, "
                    f"reserved=${self._active_reserved_usd:.4f}, "
                    f"estimate=${estimate_usd:.4f})",
                    spent_usd=self.spent_usd,
                    limit_usd=self.limit_usd,
                )
            self._active_reserved_usd += estimate_usd
        return _BudgetReservation(self, estimate_usd)


class BudgetedLLMClient(LLMClient):
    """Wrap any ``LLMClient`` so every completion tallies against a budget.

    ``estimate_cost_usd`` is a conservative per-call cost estimate used to
    reserve budget before each provider call. It is required: without an
    estimate the tracker cannot admit the call safely under concurrency.

    On success the reservation is reconciled with the actual ``Usage.cost``
    (or the estimate if the actual cost is unknown). On failure the
    reservation is released so it does not strand active budget indefinitely.
    """

    def __init__(
        self,
        inner: LLMClient,
        budget: BudgetTracker,
        *,
        estimate_cost_usd: float | None = None,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._estimate_cost_usd = estimate_cost_usd

    async def complete(self, params: CompletionParams) -> LLMResponse:
        estimate = self._estimate_cost_usd
        if estimate is None:
            raise JigBudgetError(
                "BudgetedLLMClient requires estimate_cost_usd to enforce the "
                "hard budget cap under concurrency. Provide a conservative "
                "per-call cost estimate at construction time.",
                spent_usd=self._budget.spent_usd,
                limit_usd=self._budget.limit_usd,
            )
        reservation = await self._budget.reserve(estimate)
        try:
            response = await self._inner.complete(params)
            await reservation.reconcile(response.usage.cost)
            return response
        except BaseException:
            await reservation.release()
            raise

    async def aclose(self) -> None:
        """Delegate teardown to the inner client if it exposes ``aclose``."""
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()

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
