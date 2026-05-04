"""Regression detection over a ``SweepResult``.

A sweep produces a grid of scores. Regression detection asks "did
this candidate config drop more than X on dimension Y compared to
the baseline?" Same machinery, different report — pure analysis on
top of :meth:`SweepResult.rollup`.

Use as a CI gate::

    report = detect_regressions(result, baseline="prod-config")
    if report.has_regressions:
        sys.exit(1)
"""
from __future__ import annotations

from dataclasses import dataclass

from jig.sweep import SweepResult


@dataclass
class RegressionAlert:
    """One regression — a candidate config dropped on a dimension.

    ``dimension`` is either a score-dimension name from the sweep or
    the literal ``"success_rate"`` (the synthetic dimension covering
    fraction of runs that completed without an :class:`AgentError`).
    """

    config_name: str
    dimension: str
    baseline_avg: float
    candidate_avg: float
    delta: float
    threshold: float


@dataclass
class RegressionReport:
    """Regression analysis result.

    ``deltas`` carries every observed delta (not just alert-worthy
    ones) so dashboards and CI logs can show the full picture even
    when no alerts fire.
    """

    baseline_config: str
    alerts: list[RegressionAlert]
    deltas: dict[str, dict[str, float]]

    @property
    def has_regressions(self) -> bool:
        return bool(self.alerts)


def detect_regressions(
    result: SweepResult,
    *,
    baseline: str,
    threshold: float = 0.05,
    success_rate_threshold: float | None = None,
) -> RegressionReport:
    """Compare every candidate config in ``result`` against ``baseline``.

    Reports a :class:`RegressionAlert` when any candidate config's
    avg score on a dimension drops more than ``threshold`` below the
    baseline's average for the same dimension. Triggers strictly on
    ``delta < -threshold`` (an exact ``-threshold`` drop does NOT
    alert). ``threshold`` is absolute on the ``[0, 1]`` score scale.

    ``success_rate`` is treated as a synthetic dimension. When
    ``success_rate_threshold`` is ``None``, it inherits ``threshold``;
    pass an explicit value to gate reliability separately from soft
    score regressions. Most CI gates care more about runs erroring
    out than a small drop in a judge dimension, so the gate is on
    by default.

    Raises ``ValueError`` if ``baseline`` is not a config in the
    sweep — surfaces the typo at gate-time rather than producing an
    empty report.
    """
    rollup = result.rollup()
    if baseline not in rollup:
        raise ValueError(
            f"baseline config {baseline!r} not present in sweep "
            f"(configs: {sorted(rollup.keys())})"
        )

    baseline_scores = rollup[baseline]["avg_scores"]
    baseline_success = float(rollup[baseline]["success_rate"])
    sr_threshold = (
        threshold if success_rate_threshold is None
        else success_rate_threshold
    )

    alerts: list[RegressionAlert] = []
    deltas: dict[str, dict[str, float]] = {}

    for cfg_name, summary in rollup.items():
        if cfg_name == baseline:
            continue
        cand_scores = summary["avg_scores"]
        deltas[cfg_name] = {}

        # Score-dimension regressions
        for dim, base_avg in baseline_scores.items():
            cand_avg = cand_scores.get(dim)
            if cand_avg is None:
                # Candidate didn't report this dimension — likely a
                # different grader config rather than a regression.
                # Skip rather than guess.
                continue
            delta = cand_avg - base_avg
            deltas[cfg_name][dim] = delta
            if delta < -threshold:
                alerts.append(
                    RegressionAlert(
                        config_name=cfg_name,
                        dimension=dim,
                        baseline_avg=base_avg,
                        candidate_avg=cand_avg,
                        delta=delta,
                        threshold=threshold,
                    )
                )

        # success_rate regression (synthetic dimension)
        cand_success = float(summary["success_rate"])
        sr_delta = cand_success - baseline_success
        deltas[cfg_name]["success_rate"] = sr_delta
        if sr_delta < -sr_threshold:
            alerts.append(
                RegressionAlert(
                    config_name=cfg_name,
                    dimension="success_rate",
                    baseline_avg=baseline_success,
                    candidate_avg=cand_success,
                    delta=sr_delta,
                    threshold=sr_threshold,
                )
            )

    return RegressionReport(
        baseline_config=baseline,
        alerts=alerts,
        deltas=deltas,
    )
