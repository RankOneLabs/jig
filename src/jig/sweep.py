"""Experimentation primitives: compare one input across configs, or sweep
cases × configs for side-by-side evaluation.

These are thin coordinators over :func:`run_agent` — the heavy lifting
(LLM dispatch, tool execution, grading, typed outputs) already lives in
the runner. ``SweepResult.rollup()`` turns the grid of runs into
per-config aggregates that answer "which config wins?" without the
caller having to stitch the data by hand.

Auto-persistence to :class:`FeedbackLoop` and dispatch-backed parallelism
land in phase 8 — for now, concurrency is local via ``asyncio.Semaphore``.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from jig.core.runner import AgentConfig, AgentResult, run_agent
from jig.core.types import EvalCase

_VALID_DISPATCH = frozenset({"smithers"})

# Reference-count for the sweep-owned listener. Two concurrent
# compare()/sweep() calls with dispatch="smithers" must share the
# listener; whichever exits first must not tear it down under the
# other one. Counter starts at 0, incremented on 0→1 transition
# (which starts the listener) and decremented on exit (which stops
# the listener on the 1→0 transition). Lock-protected so check-then-
# act on the counter can't race.
_managed_listener_refs: int = 0
_managed_listener_lock: asyncio.Lock | None = None


def _get_managed_lock() -> asyncio.Lock:
    """Lazy construction to avoid binding to an import-time event loop."""
    global _managed_listener_lock
    if _managed_listener_lock is None:
        _managed_listener_lock = asyncio.Lock()
    return _managed_listener_lock


@asynccontextmanager
async def _dispatch_listener(dispatch: str | None) -> AsyncIterator[None]:
    """Ensure a callback listener runs for the wrapped block when
    ``dispatch == "smithers"``. No-op otherwise.

    Ownership is reference-counted: if the caller pre-started a
    listener (outside any ``_dispatch_listener`` context), this context
    reuses it and leaves it running. Otherwise the first entering
    context starts it; the last exiting context stops it. Concurrent
    sweeps therefore share one listener without stomping on each
    other's shutdown.
    """
    if dispatch is None:
        yield
        return
    if dispatch not in _VALID_DISPATCH:
        raise ValueError(
            f"Unknown dispatch backend {dispatch!r}; "
            f"expected one of {sorted(_VALID_DISPATCH)}"
        )

    # Import lazily so non-dispatch callers never pay for the aiohttp
    # extra. The try/except surfaces a clean install hint.
    try:
        from jig.dispatch import listener as _listener_mod
    except ImportError as exc:
        raise ImportError(
            "sweep(dispatch='smithers') requires the callback listener. "
            "Install with: pip install 'jig[callback]'"
        ) from exc

    global _managed_listener_refs
    owns_managed_ref = False
    async with _get_managed_lock():
        active = _listener_mod._active_listener()
        if active is None:
            await _listener_mod.listen()
            _managed_listener_refs = 1
            owns_managed_ref = True
        elif _managed_listener_refs > 0:
            # Another sweep already owns the lifecycle — join as a
            # ref holder so we keep it alive past that sweep's exit.
            _managed_listener_refs += 1
            owns_managed_ref = True
        # else: caller pre-started the listener outside any sweep
        # context. They own the lifecycle; we're just borrowing.

    try:
        yield
    finally:
        if owns_managed_ref:
            async with _get_managed_lock():
                _managed_listener_refs -= 1
                if _managed_listener_refs == 0:
                    await _listener_mod.stop()


@dataclass
class CompareRun[T]:
    config_index: int
    config_name: str
    result: AgentResult[T]


@dataclass
class CompareResult[T]:
    """Outcome of :func:`compare` — one input run against N configs."""

    input: str
    runs: list[CompareRun[T]]

    def rollup(self) -> dict[str, dict[str, Any]]:
        """Per-config summary keyed by ``config.name``."""
        out: dict[str, dict[str, Any]] = {}
        for run in self.runs:
            r = run.result
            by_dim: dict[str, list[float]] = {}
            if r.scores:
                for s in r.scores:
                    by_dim.setdefault(s.dimension, []).append(s.value)
            out[run.config_name] = {
                "avg_scores": {d: sum(vs) / len(vs) for d, vs in by_dim.items()},
                "cost_usd": r.usage.get("total_cost", 0.0),
                "latency_ms": r.duration_ms,
                "error_category": r.error.category if r.error is not None else None,
            }
        return out


@dataclass
class SweepRun[T]:
    case_index: int
    config_index: int
    config_name: str
    input: str
    result: AgentResult[T]


@dataclass
class SweepResult[T]:
    """Outcome of :func:`sweep` — cases × configs grid."""

    sweep_id: str
    runs: list[SweepRun[T]]

    def rollup(self) -> dict[str, dict[str, Any]]:
        """Per-config aggregates across all cases.

        Each entry:
        ``n`` (run count), ``avg_scores`` (per dimension),
        ``avg_cost_usd``, ``avg_latency_ms``, ``success_rate``
        (fraction with ``error is None``), ``error_categories``
        (Counter of category tags).
        """
        by_config: dict[str, list[AgentResult]] = {}
        for r in self.runs:
            by_config.setdefault(r.config_name, []).append(r.result)

        out: dict[str, dict[str, Any]] = {}
        for name, results in by_config.items():
            scores_by_dim: dict[str, list[float]] = {}
            costs: list[float] = []
            latencies: list[float] = []
            errors: list[str] = []
            successes = 0
            for res in results:
                if res.scores:
                    for s in res.scores:
                        scores_by_dim.setdefault(s.dimension, []).append(s.value)
                costs.append(float(res.usage.get("total_cost", 0.0)))
                latencies.append(res.duration_ms)
                if res.error is not None:
                    errors.append(res.error.category)
                else:
                    successes += 1
            n = len(results)
            out[name] = {
                "n": n,
                "avg_scores": {d: sum(vs) / len(vs) for d, vs in scores_by_dim.items()},
                "avg_cost_usd": sum(costs) / n if n else 0.0,
                "avg_latency_ms": sum(latencies) / n if n else 0.0,
                "success_rate": successes / n if n else 0.0,
                "error_categories": dict(Counter(errors)),
            }
        return out


def _case_to_input(case: EvalCase | str) -> str:
    return case if isinstance(case, str) else case.input


def _ensure_unique_names(configs: Sequence[AgentConfig[Any]]) -> None:
    """Guard against duplicate config names.

    Rollups key on ``config.name``. Duplicates would silently merge /
    overwrite entries, losing data. Force callers to name variants
    distinctly so the rollup maps back to a specific config.
    """
    seen: set[str] = set()
    dupes: list[str] = []
    for cfg in configs:
        if cfg.name in seen:
            dupes.append(cfg.name)
        seen.add(cfg.name)
    if dupes:
        unique = sorted(set(dupes))
        raise ValueError(
            f"Config names must be unique for rollup keying; "
            f"duplicates: {unique}"
        )


async def compare[T](
    input: str,
    configs: list[AgentConfig[T]],
    *,
    concurrency: int = 1,
    dispatch: str | None = None,
) -> CompareResult[T]:
    """Run the same input through N configs; return the grid.

    ``concurrency > 1`` runs configs in parallel via
    ``asyncio.Semaphore``. Use this to A/B (or A/B/C) test model
    swaps / prompt variants / retriever choices on a single probe.

    ``dispatch="smithers"`` wraps the call in a callback-listener
    lifecycle. Configs that route LLM calls through smithers get the
    callback path automatically; configs that run locally are
    unaffected.
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    _ensure_unique_names(configs)

    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int, cfg: AgentConfig[T]) -> CompareRun[T]:
        async with sem:
            result = await run_agent(cfg, input)
            return CompareRun(
                config_index=idx,
                config_name=cfg.name,
                result=result,
            )

    async with _dispatch_listener(dispatch):
        runs = await asyncio.gather(*[_one(i, c) for i, c in enumerate(configs)])
    return CompareResult(input=input, runs=list(runs))


async def sweep[T](
    cases: list[EvalCase] | list[str],
    configs: list[AgentConfig[T]],
    *,
    concurrency: int = 1,
    sweep_id: str | None = None,
    dispatch: str | None = None,
) -> SweepResult[T]:
    """Run every (case, config) pair; return a SweepResult for rollup.

    Inputs accept plain strings or :class:`EvalCase` (for metadata). A
    fresh ``sweep_id`` is generated per call unless supplied; downstream
    analytics can join ``run_agent`` traces by that tag once phase 5's
    auto-persist path lands.

    ``dispatch="smithers"`` ensures a callback listener runs for the
    sweep's duration. With N parallel runs, this replaces N polling
    coroutines with one HTTP receiver. If the caller has already
    started a listener, the sweep uses it and leaves it running;
    otherwise the sweep owns the listener for its duration.
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    _ensure_unique_names(configs)

    resolved_sweep_id = sweep_id or str(uuid.uuid4())
    sem = asyncio.Semaphore(concurrency)

    async def _one(
        case_idx: int,
        cfg_idx: int,
        case: EvalCase | str,
        cfg: AgentConfig[T],
    ) -> SweepRun[T]:
        async with sem:
            input_text = _case_to_input(case)
            result = await run_agent(cfg, input_text)
            return SweepRun(
                case_index=case_idx,
                config_index=cfg_idx,
                config_name=cfg.name,
                input=input_text,
                result=result,
            )

    async with _dispatch_listener(dispatch):
        tasks = [
            _one(ci, gi, case, cfg)
            for ci, case in enumerate(cases)
            for gi, cfg in enumerate(configs)
        ]
        runs = await asyncio.gather(*tasks)
    return SweepResult(sweep_id=resolved_sweep_id, runs=list(runs))
