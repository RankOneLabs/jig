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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from jig.core.runner import AgentConfig, AgentResult, run_agent
from jig.core.types import EvalCase


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
) -> CompareResult[T]:
    """Run the same input through N configs; return the grid.

    ``concurrency > 1`` runs configs in parallel via
    ``asyncio.Semaphore``. Use this to A/B (or A/B/C) test model
    swaps / prompt variants / retriever choices on a single probe.
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

    runs = await asyncio.gather(*[_one(i, c) for i, c in enumerate(configs)])
    return CompareResult(input=input, runs=list(runs))


async def sweep[T](
    cases: list[EvalCase] | list[str],
    configs: list[AgentConfig[T]],
    *,
    concurrency: int = 1,
    sweep_id: str | None = None,
) -> SweepResult[T]:
    """Run every (case, config) pair; return a SweepResult for rollup.

    Inputs accept plain strings or :class:`EvalCase` (for metadata). A
    fresh ``sweep_id`` is generated per call unless supplied; downstream
    analytics can join ``run_agent`` traces by that tag once phase 5's
    auto-persist path lands.
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

    tasks = [
        _one(ci, gi, case, cfg)
        for ci, case in enumerate(cases)
        for gi, cfg in enumerate(configs)
    ]
    runs = await asyncio.gather(*tasks)
    return SweepResult(sweep_id=resolved_sweep_id, runs=list(runs))
