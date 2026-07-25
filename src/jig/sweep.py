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
import inspect
import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jig.core.errors import AgentBudgetError, AgentError, JigBudgetError
from jig.core.runner import (
    AgentConfig,
    AgentResult,
    _CLOSEABLE_CONFIG_FIELDS,
    _close_resource,
    run_agent,
)
from jig.core.types import EvalCase

logger = logging.getLogger(__name__)

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
    # Index of this run within the (case, config) repetition group.
    # 0 for single-run sweeps (the default ``seeds=1``); 0..seeds-1
    # when ``sweep(..., seeds=N)`` was called. Analyses like
    # ``pass_at_k`` group by this implicitly via the case_index +
    # config_name pair.
    seed_index: int = 0


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


@dataclass(frozen=True, slots=True)
class SweepRunInfo:
    """Identifies one (case, config, seed) slot for the run-lifecycle
    callbacks below. Passed to ``on_run_complete``/``on_run_error``/
    ``on_run_abandoned`` so a callback can correlate the notification
    with its own bookkeeping without re-deriving the slot's coordinates.

    ``case`` is typed ``EvalCase | str`` rather than just ``EvalCase``:
    ``sweep()`` accepts plain strings as cases (see ``_case_to_input``),
    and this carries the case exactly as passed in, unconverted.
    """

    sweep_id: str
    case_index: int
    case: EvalCase | str
    config_name: str
    seed_index: int


async def _fire_callback(
    hook: Callable[..., Awaitable[None] | None] | None,
    *args: Any,
    hook_name: str,
) -> None:
    """Best-effort invocation of a sweep run-lifecycle callback.

    Sync or async; awaited either way. Exceptions raised by the hook are
    logged at warning and swallowed — bookkeeping (this is purely a
    notification, not part of the run's outcome) must never take down
    the worker processing it or any sibling run.
    """
    if hook is None:
        return
    try:
        outcome = hook(*args)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        logger.warning("sweep %s callback raised", hook_name, exc_info=True)


async def _close_all_configs(configs: Sequence[AgentConfig[Any]]) -> None:
    """Close every distinct resource across every config in ``configs``,
    each exactly once.

    ``AgentConfig.with_()`` derivatives commonly share the same
    underlying ``llm``/``tracer``/``store``/``feedback``/``retriever``
    object — e.g. gecko derives one config per attempt off a shared base
    config and appends every copy to the sweep's config list. Looping
    ``await cfg.aclose()`` once per config would invoke a shared
    resource's own ``aclose()``/``close()`` once per config that
    references it; not every resource tolerates (or silently no-ops on)
    a repeat close. ``AgentConfig.aclose()`` itself can't fix this — two
    separate config *instances* have no way to know they share a
    resource with each other; that de-duplication has to happen here,
    across the whole list, before anything is closed.

    Each config is also marked closed as it is collected — before
    anything is actually closed, matching ``AgentConfig.aclose()``'s own
    ordering. Without that, ``close_configs=True`` would close every
    resource while leaving every config still claiming it had not been
    closed, so a later ``await cfg.aclose()`` would re-close precisely
    the shared resources this de-duplication exists to protect.
    """
    seen: dict[int, object] = {}
    for cfg in configs:
        object.__setattr__(cfg, "_aclose_done", True)
        for field_name in _CLOSEABLE_CONFIG_FIELDS:
            resource = getattr(cfg, field_name)
            if resource is not None:
                seen.setdefault(id(resource), resource)
    for resource in seen.values():
        await _close_resource(resource)


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


def _error_result(error: AgentError) -> AgentResult[Any]:
    """Synthetic AgentResult for a run that could not be dispatched."""
    return AgentResult(
        output="",
        trace_id="",
        usage={
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
        scores=None,
        duration_ms=0.0,
        error=error,
    )


def _budget_error_result(exc: JigBudgetError) -> AgentResult[Any]:
    return _error_result(
        AgentBudgetError(
            str(exc),
            spent_usd=exc.spent_usd,
            limit_usd=exc.limit_usd,
        )
    )


def _infra_error_result(exc: Exception) -> AgentResult[Any]:
    detail = str(exc)
    message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return _error_result(AgentError(message))


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

    Per-config budget and infrastructure failures become structured
    :class:`CompareRun` results (``result.error`` is set) rather than
    propagating exceptions. True global errors (dispatch setup, etc.)
    still abort the call.
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    _ensure_unique_names(configs)

    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int, cfg: AgentConfig[T]) -> CompareRun[T]:
        async with sem:
            try:
                result = await run_agent(cfg, input)
            # warning, not debug, for the same reason as the sweep worker
            # loop below: a run that dies pre-trace produces no spans, so
            # a debug-level line is the only evidence it ever ran, and it
            # is invisible at default log levels. compare() gets no
            # callbacks in this PR — only the visibility fix, since the
            # silent-death trap is identical here.
            except JigBudgetError as exc:
                logger.warning(
                    "compare config=%s budget exhausted: %s", cfg.name, exc
                )
                result = _budget_error_result(exc)
            except Exception as exc:
                logger.warning(
                    "compare config=%s infrastructure error: %s", cfg.name, exc
                )
                result = _infra_error_result(exc)
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
    seeds: int = 1,
    on_run_complete: Callable[[SweepRunInfo, AgentResult[T]], Awaitable[None] | None] | None = None,
    on_run_error: Callable[[SweepRunInfo, BaseException], Awaitable[None] | None] | None = None,
    on_run_abandoned: Callable[[SweepRunInfo, str], Awaitable[None] | None] | None = None,
    close_configs: bool = False,
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

    ``seeds`` runs each ``(case, config)`` pair this many times. The
    extra repetitions are needed for distributional analyses
    (``pass_at_k``, ``win_rate``). Default ``1`` preserves existing
    behavior — backward compatible. The runner does not currently
    expose explicit per-call seed control, so ``seeds > 1`` relies
    on temperature variance for sample diversity; with
    ``temperature=0`` configs every repetition is identical and
    ``pass_at_k`` will warn at analysis time.

    Per-case budget and infrastructure failures from ``run_agent``
    become structured :class:`SweepRun` results (``result.error`` is
    set) and are preserved alongside successes — workers continue
    after isolated per-case failures. Only global setup errors (dispatch
    startup, cancellation) abort the sweep without partial results.

    **Run-lifecycle callbacks** (all optional, default ``None``; sync or
    async, awaited either way; a raising callback is logged at warning
    and swallowed — bookkeeping must never kill a sibling run):

    - ``on_run_complete(info, result)`` — fires once ``run_agent``
      returns successfully for a slot (before that slot's result is
      recorded).
    - ``on_run_error(info, exc)`` — fires whenever ``run_agent`` raises
      ``JigBudgetError`` or any other ``Exception`` for a slot, i.e. on
      exactly the two paths that today are downgraded to a synthesized
      error :class:`SweepRun` (see below). Regardless of whether a
      callback is registered, the log line for both of those paths is
      now emitted at **warning** (previously ``debug``) — a run that
      dies before opening a trace used to leave no spans and no visible
      log line; this is the unconditional fix for that, not something
      callers opt into.
    - ``on_run_abandoned(info, reason)`` — fires from this function's
      ``finally`` for every slot that is still unfilled once the sweep
      exits, whether by normal completion, exception, or cancellation.
      ``reason`` is ``"cancelled"`` when a worker had already dequeued
      that slot's work item (and so was inside ``run_agent``, or
      between it and recording the result) when a ``BaseException`` —
      in practice ``asyncio.CancelledError`` from the sweep itself being
      cancelled — cut the run short; ``"never_scheduled"`` when no
      worker ever got to it at all (still queued, still with the
      producer, or the whole worker loop never started because e.g.
      ``dispatch`` failed to set up). A normal, uninterrupted sweep
      fills every slot, so this never fires.

    ``close_configs=True`` closes every distinct resource (``llm``,
    ``tracer``, ``store``, ``feedback``, ``retriever``) across every
    config in ``configs`` — each exactly once, even when several configs
    share the same underlying object (the common ``with_()`` pattern) —
    once the sweep is done, success or not. Default ``False`` closes
    nothing, same as today. See :func:`_close_all_configs` for why this
    doesn't simply loop ``cfg.aclose()``.
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    if seeds <= 0:
        raise ValueError(f"seeds must be positive, got {seeds}")
    _ensure_unique_names(configs)

    resolved_sweep_id = sweep_id or str(uuid.uuid4())

    # Worker-pool pattern: instead of building a list of
    # ``len(cases) * len(configs) * seeds`` coroutine objects up
    # front and handing them all to ``asyncio.gather``, we run
    # exactly ``concurrency`` workers that pull work items from a
    # bounded queue. This caps in-memory state at O(concurrency)
    # rather than O(cases * configs * seeds) — for large eval
    # sweeps the difference is real (50k+ pending coroutines is a
    # measurable spike).
    total = len(cases) * len(configs) * seeds
    runs: list[SweepRun[T] | None] = [None] * total
    # Slot -> (case_index, config_index, seed_index, case, config), built
    # once up front so the (case, config, seed) mapping is available for
    # on_run_abandoned even for a slot whose coroutine never got a queue.get()
    # turn — including the total-setup-failure case where the worker loop
    # never starts at all (e.g. an invalid `dispatch` value). Also the single
    # source of truth _produce() enqueues from, so the two can't disagree.
    slot_meta: list[tuple[int, int, int, EvalCase | str, AgentConfig[T]]] = []
    for ci, case in enumerate(cases):
        for gi, cfg in enumerate(configs):
            for si in range(seeds):
                slot_meta.append((ci, gi, si, case, cfg))
    # Slots a worker has dequeued (and so is either inside run_agent or
    # between it and recording the result) — distinguishes "cancelled"
    # (started, never finished) from "never_scheduled" (no worker ever
    # reached it) among slots left unfilled in the finally below.
    started: set[int] = set()
    # Bound the queue to ``concurrency`` so the producer can't run
    # ahead of the workers and pre-materialize every coroutine.
    queue: asyncio.Queue[tuple[int, int, int, int, EvalCase | str, AgentConfig[T]] | None] = (
        asyncio.Queue(maxsize=concurrency)
    )

    async def _worker(worker_id: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                logger.debug("sweep worker %d shutting down", worker_id)
                queue.task_done()
                return
            slot, case_idx, cfg_idx, seed_idx, case, cfg = item
            started.add(slot)
            info = SweepRunInfo(
                sweep_id=resolved_sweep_id,
                case_index=case_idx,
                case=case,
                config_name=cfg.name,
                seed_index=seed_idx,
            )
            try:
                input_text = _case_to_input(case)
                logger.debug(
                    "sweep worker %d dispatching slot=%d case=%d cfg=%s seed=%d",
                    worker_id, slot, case_idx, cfg.name, seed_idx,
                )
                try:
                    result = await run_agent(cfg, input_text)
                except JigBudgetError as exc:
                    # Unconditional debug->warning bump (not gated on a
                    # callback being registered): a run that dies before
                    # opening a trace used to leave no spans and no visible
                    # log line, which cost a full forensic session more than
                    # once. See on_run_error below for the callback.
                    logger.warning(
                        "sweep worker %d slot=%d cfg=%s budget exhausted: %s",
                        worker_id, slot, cfg.name, exc,
                    )
                    result = _budget_error_result(exc)
                    await _fire_callback(on_run_error, info, exc, hook_name="on_run_error")
                except Exception as exc:
                    logger.warning(
                        "sweep worker %d slot=%d cfg=%s infrastructure error: %s",
                        worker_id, slot, cfg.name, exc,
                    )
                    result = _infra_error_result(exc)
                    await _fire_callback(on_run_error, info, exc, hook_name="on_run_error")
                else:
                    await _fire_callback(on_run_complete, info, result, hook_name="on_run_complete")
                logger.debug(
                    "sweep worker %d completed slot=%d cfg=%s",
                    worker_id, slot, cfg.name,
                )
                runs[slot] = SweepRun(
                    case_index=case_idx,
                    config_index=cfg_idx,
                    config_name=cfg.name,
                    input=input_text,
                    result=result,
                    seed_index=seed_idx,
                )
            finally:
                queue.task_done()

    async def _produce() -> None:
        for slot, (ci, gi, si, case, cfg) in enumerate(slot_meta):
            await queue.put((slot, ci, gi, si, case, cfg))
        for _ in range(n_workers):
            await queue.put(None)

    try:
        async with _dispatch_listener(dispatch):
            # Cap the worker count at the actual workload — spinning up
            # more workers than items just wastes scheduler overhead.
            n_workers = min(concurrency, total) if total > 0 else 0
            logger.debug(
                "sweep starting: total=%d concurrency=%d workers=%d cases=%d configs=%d seeds=%d",
                total, concurrency, n_workers, len(cases), len(configs), seeds,
            )
            worker_tasks = [asyncio.create_task(_worker(i)) for i in range(n_workers)]
            producer_task = asyncio.create_task(_produce())
            try:
                await asyncio.gather(producer_task, *worker_tasks)
            except BaseException:
                # On cancellation or unexpected worker/producer failure, cancel
                # everything so neither a blocked producer nor idle workers outlive
                # the sweep call.
                producer_task.cancel()
                for w in worker_tasks:
                    w.cancel()
                await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)
                raise
    finally:
        # Runs regardless of how the sweep exits — normal completion,
        # an exception from setup (e.g. a bad `dispatch` value, in which
        # case the loop above never started and every slot is unfilled),
        # or cancellation partway through. A normal sweep fills every
        # slot, so this is a no-op unless something above didn't finish.
        for slot, (ci, gi, si, case, cfg) in enumerate(slot_meta):
            if runs[slot] is not None:
                continue
            reason = "cancelled" if slot in started else "never_scheduled"
            info = SweepRunInfo(
                sweep_id=resolved_sweep_id,
                case_index=ci,
                case=case,
                config_name=cfg.name,
                seed_index=si,
            )
            await _fire_callback(on_run_abandoned, info, reason, hook_name="on_run_abandoned")
        if close_configs:
            await _close_all_configs(configs)

    # Filter None slots (empty sweep) and cast for type checker.
    return SweepResult(
        sweep_id=resolved_sweep_id,
        runs=[r for r in runs if r is not None],  # type: ignore[misc]
    )
