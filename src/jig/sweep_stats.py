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

    ``win_rate`` is computed in the strict pairwise sense: ties are
    excluded from both numerator and denominator, so it answers
    "given a decisive comparison, how often did A win?"

    ``n_compared`` is the count of cases where both configs reported
    scores on the dimension. ``n_decisive`` is the subset of those
    where ``avg_a != avg_b``; the win rate's denominator. When all
    cases tie, ``n_decisive == 0`` and ``win_rate`` is ``0.0`` by
    convention (no decisive evidence either way).

    ``ci_low``/``ci_high`` are the 2.5th/97.5th percentiles of a
    bootstrap over cases.
    """

    config_a: str
    config_b: str
    dimension: str
    win_rate: float
    ci_low: float
    ci_high: float
    n_compared: int
    n_decisive: int = 0


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
    return list is the signal. Configs that report the dimension on
    *some* but not all of their cases emit a ``RuntimeWarning`` and
    are skipped: silently dropping the missing cases would overstate
    pass@k by hiding the hardest cases from the rollup.

    **Variance precondition.** With deterministic LLM settings
    (``temperature=0``, no seed jitter) every "seed" produces the
    same output, so per-case score vectors collapse to constants and
    pass@k degenerates to pass@1 silently. When this is detected,
    a ``RuntimeWarning`` is emitted naming the config so the metric
    isn't silently misinterpreted.
    """
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive when provided, got {k}")

    # Track expected case coverage per config so we can detect
    # configs that reported the dimension on a subset of cases
    # (silent drop = overstated pass@k).
    expected_cases_by_config: dict[str, set[int]] = {}
    by_config_case: dict[str, dict[int, list[float]]] = {}
    for run in result.runs:
        expected_cases_by_config.setdefault(run.config_name, set()).add(
            run.case_index
        )
        scores = run.result.scores or []
        by_dim = [s.value for s in scores if s.dimension == dimension]
        if not by_dim:
            continue
        # If a single run has multiple Score entries for the same
        # dimension (e.g. a CompositeGrader composes two graders that
        # both report ``dimension``), aggregate them as the mean.
        # Same shape ``win_rate`` uses; keeps results deterministic
        # rather than depending on score-list order.
        per_run_value = float(np.mean(by_dim))
        by_config_case.setdefault(run.config_name, {}).setdefault(
            run.case_index, []
        ).append(per_run_value)

    out: list[PassAtK] = []
    for cfg, by_case in by_config_case.items():
        observed_cases = set(by_case)
        expected_cases = expected_cases_by_config.get(cfg, set())
        if observed_cases != expected_cases:
            missing = expected_cases - observed_cases
            warnings.warn(
                f"pass@k for config {cfg!r} dimension {dimension!r}: "
                f"dimension missing on {len(missing)} of "
                f"{len(expected_cases)} sweep cases (case indices "
                f"{sorted(missing)}); skipped to avoid overstating "
                f"pass@k.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Require uniform n_per_case across cases for clean math.
        ns = {len(vs) for vs in by_case.values()}
        if len(ns) != 1:
            continue
        n = ns.pop()
        k_eff = k if k is not None else n
        if k_eff > n:
            # Caller asked for a larger k than this config's sample
            # count supports. Warn so the empty result isn't a silent
            # misconfiguration; skip per-config so a single under-
            # sampled config doesn't sink the whole report.
            warnings.warn(
                f"pass@k for config {cfg!r} dimension {dimension!r}: "
                f"requested k={k_eff} exceeds n_per_case={n}; skipped. "
                f"Reduce k or increase ``sweep(seeds=)``.",
                RuntimeWarning,
                stacklevel=2,
            )
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

    For each case, A's per-case mean is compared to B's. **Strict
    pairwise semantics:** ties are excluded from both numerator and
    denominator, so the win rate answers "given a decisive
    comparison, how often did A win?" The point estimate is
    ``count(avg_a > avg_b) / n_decisive``; the CI is the 95%
    bootstrap over cases (each bootstrap sample applies the same
    strict-pairwise rule to its resampled pairs).

    Cases where either config didn't report ``dimension`` are
    skipped. When no case is decisive — either no overlap or every
    overlapping case ties — returns a zero-everywhere ``WinRate``
    rather than raising; caller can branch on ``n_compared`` or
    ``n_decisive``.

    Within a single run, multiple Score entries on ``dimension``
    (e.g. via ``CompositeGrader``) are aggregated as the mean before
    averaging across runs in a case. Matches ``pass_at_k``.

    Pass ``seed`` for reproducible bootstrap.
    """
    if bootstrap_samples <= 0:
        raise ValueError(
            f"bootstrap_samples must be positive, got {bootstrap_samples}"
        )

    by_case_a: dict[int, list[float]] = {}
    by_case_b: dict[int, list[float]] = {}
    for run in result.runs:
        scores = run.result.scores or []
        vals = [s.value for s in scores if s.dimension == dimension]
        if not vals:
            continue
        # Collapse within-run duplicates to a single per-run value
        # (mean) so a CompositeGrader emitting two scores on the
        # same dimension doesn't double-count for that run.
        per_run_value = float(np.mean(vals))
        if run.config_name == config_a:
            by_case_a.setdefault(run.case_index, []).append(per_run_value)
        elif run.config_name == config_b:
            by_case_b.setdefault(run.case_index, []).append(per_run_value)

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
            n_decisive=0,
        )

    arr = np.array(pairs)
    decisive_mask = arr[:, 0] != arr[:, 1]
    n_decisive = int(np.sum(decisive_mask))
    if n_decisive == 0:
        # All ties — no decisive evidence either way. Return zero CI;
        # caller branches on n_decisive.
        return WinRate(
            config_a=config_a,
            config_b=config_b,
            dimension=dimension,
            win_rate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            n_compared=len(pairs),
            n_decisive=0,
        )
    point = float(
        np.mean(arr[decisive_mask, 0] > arr[decisive_mask, 1])
    )

    rng = np.random.default_rng(seed)
    n = len(pairs)
    boot = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        sample_decisive = sample[:, 0] != sample[:, 1]
        if not np.any(sample_decisive):
            # This bootstrap sample drew only ties — record 0.0
            # rather than NaN; over many samples the CI still
            # represents the resampling distribution honestly.
            boot[i] = 0.0
        else:
            boot[i] = float(
                np.mean(
                    sample[sample_decisive, 0] > sample[sample_decisive, 1]
                )
            )
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
        n_decisive=n_decisive,
    )
