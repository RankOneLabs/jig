"""Statistical analysis over ``SweepResult.runs``.

`SweepResult.rollup()` produces point estimates. Probabilistic agents
need distributions: pass@k under temperature, win rates with bootstrap
CIs. The runs are already in `SweepResult.runs`; this module turns the
grid into the analyses.

These functions all require the sweep to have been run with
``seeds > 1`` for results that aren't degenerate. With deterministic
LLM settings (temperature 0, no seed jitter) every "seed" produces
the same output and the metrics collapse. ``pass_at_k`` warns when
this happens; ``win_rate`` accepts whatever data it gets.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from jig.sweep import SweepResult


@dataclass
class PassAtK:
    """pass@k for one (config, dimension) pair across the sweep."""

    config_name: str
    dimension: str
    k: int
    n_per_case: int
    pass_at_k: float
    n_cases: int


@dataclass
class WinRate:
    """Pairwise win rate of ``config_a`` over ``config_b`` on a dimension.

    ``ci_low``/``ci_high`` are the 2.5th/97.5th percentiles of a
    bootstrap over cases. ``n_compared`` is the count of cases where
    both configs reported scores on the dimension.
    """

    config_a: str
    config_b: str
    dimension: str
    win_rate: float
    ci_low: float
    ci_high: float
    n_compared: int


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Codex/HumanEval pass@k estimator.

    Implementation: ``1.0 - prod((n-c-i) / (n-i)) for i in 0..k-1``.
    Adequate for typical eval ``k <= 32``; for very large k consider
    log-space accumulation.
    """
    # When failures are fewer than k, you can't draw k all-failing
    # samples — so at least one of k samples passes by construction.
    if n - c < k:
        return 1.0
    return 1.0 - float(
        np.prod(np.array([(n - c - i) / (n - i) for i in range(k)]))
    )


def pass_at_k(
    result: SweepResult,
    *,
    dimension: str,
    threshold: float = 0.5,
    k: int | None = None,
) -> list[PassAtK]:
    """Compute pass@k per config for a given score dimension.

    A run "passes" if its score on ``dimension`` is ``>= threshold``.
    For each ``(config, case)``, counts how many of the runs passed;
    pass@k is the probability that at least one of k samples passes,
    via the Codex/HumanEval unbiased estimator. ``k`` defaults to
    ``n_per_case`` (pass@n).

    Configs whose runs are non-uniformly sampled (different
    ``n_per_case`` across cases for the same config) are skipped
    cleanly rather than guessed. Configs missing the requested
    dimension entirely are also skipped — their absence in the
    return list is the signal.

    **Variance precondition.** With deterministic LLM settings
    (``temperature=0``, no seed jitter) every "seed" produces the
    same output, so per-case score vectors collapse to constants and
    pass@k degenerates to pass@1 silently. When this is detected,
    a ``RuntimeWarning`` is emitted naming the config so the metric
    isn't silently misinterpreted.
    """
    by_config_case: dict[str, dict[int, list[float]]] = {}
    for run in result.runs:
        scores = run.result.scores or []
        by_dim = [s.value for s in scores if s.dimension == dimension]
        if not by_dim:
            continue
        by_config_case.setdefault(run.config_name, {}).setdefault(
            run.case_index, []
        ).append(by_dim[0])

    out: list[PassAtK] = []
    for cfg, by_case in by_config_case.items():
        # Require uniform n_per_case across cases for clean math.
        ns = {len(vs) for vs in by_case.values()}
        if len(ns) != 1:
            continue
        n = ns.pop()
        k_eff = k if k is not None else n
        if k_eff > n:
            continue

        # Variance check: warn if all per-case score vectors are
        # constant. With n>1 this means seeds produced identical
        # outputs — typically temperature=0 without seed control.
        if n > 1 and all(len(set(vs)) == 1 for vs in by_case.values()):
            warnings.warn(
                f"pass@k for config {cfg!r} dimension {dimension!r}: "
                f"all per-case score vectors are constant across "
                f"{n} seeds — likely temperature=0 without seed "
                f"jitter. Metric collapses to pass@1.",
                RuntimeWarning,
                stacklevel=2,
            )

        passes_per_case = [
            sum(1 for v in vs if v >= threshold) for vs in by_case.values()
        ]
        pak = float(
            np.mean(
                [_pass_at_k_unbiased(n, c, k_eff) for c in passes_per_case]
            )
        )
        out.append(
            PassAtK(
                config_name=cfg,
                dimension=dimension,
                k=k_eff,
                n_per_case=n,
                pass_at_k=pak,
                n_cases=len(by_case),
            )
        )
    return out


def win_rate(
    result: SweepResult,
    *,
    dimension: str,
    config_a: str,
    config_b: str,
    bootstrap_samples: int = 1000,
    seed: int | None = None,
) -> WinRate:
    """Pairwise win rate of A over B on ``dimension`` with bootstrap CI.

    For each case, A's per-case mean is compared to B's; A wins on
    that case if ``avg_a > avg_b`` (ties don't count as wins). The
    point estimate is the fraction of cases A wins; the CI is the
    95% bootstrap CI over cases.

    Cases where either config didn't report ``dimension`` are
    skipped. When no case is comparable, returns a zero-everywhere
    ``WinRate`` rather than raising — caller can branch on
    ``n_compared``.

    Pass ``seed`` for reproducible bootstrap.
    """
    by_case_a: dict[int, list[float]] = {}
    by_case_b: dict[int, list[float]] = {}
    for run in result.runs:
        scores = run.result.scores or []
        vals = [s.value for s in scores if s.dimension == dimension]
        if not vals:
            continue
        if run.config_name == config_a:
            by_case_a.setdefault(run.case_index, []).extend(vals)
        elif run.config_name == config_b:
            by_case_b.setdefault(run.case_index, []).extend(vals)

    pairs: list[tuple[float, float]] = []
    for case_idx in sorted(set(by_case_a) & set(by_case_b)):
        pairs.append(
            (
                float(np.mean(by_case_a[case_idx])),
                float(np.mean(by_case_b[case_idx])),
            )
        )
    if not pairs:
        return WinRate(
            config_a=config_a,
            config_b=config_b,
            dimension=dimension,
            win_rate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            n_compared=0,
        )

    arr = np.array(pairs)
    point = float(np.mean(arr[:, 0] > arr[:, 1]))

    rng = np.random.default_rng(seed)
    n = len(pairs)
    boot = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        boot[i] = float(np.mean(sample[:, 0] > sample[:, 1]))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))

    return WinRate(
        config_a=config_a,
        config_b=config_b,
        dimension=dimension,
        win_rate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_compared=n,
    )
