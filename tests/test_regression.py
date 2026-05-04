"""Tests for jig.regression — detect_regressions over SweepResult.

Tests build SweepResult/SweepRun/AgentResult fakes directly rather
than running run_agent. The function under test is read-only over
SweepResult.rollup(), so a synthetic rollup is all we need.
"""
from __future__ import annotations

import pytest

from jig import (
    AgentMaxLLMCallsError,
    RegressionAlert,
    RegressionReport,
    Score,
    ScoreSource,
    SweepResult,
    SweepRun,
    detect_regressions,
)
from jig.core.runner import AgentResult


def _agent_result(
    *,
    scores: list[Score] | None = None,
    error: bool = False,
    cost: float = 0.001,
    latency_ms: float = 1.0,
) -> AgentResult:
    return AgentResult(
        output="ok",
        trace_id="t",
        usage={"total_cost": cost},
        scores=scores,
        duration_ms=latency_ms,
        error=AgentMaxLLMCallsError(15) if error else None,
    )


def _run(
    case_idx: int,
    config_idx: int,
    config_name: str,
    *,
    scores: list[Score] | None = None,
    error: bool = False,
) -> SweepRun:
    return SweepRun(
        case_index=case_idx,
        config_index=config_idx,
        config_name=config_name,
        input=f"input-{case_idx}",
        result=_agent_result(scores=scores, error=error),
    )


def _score(dim: str, value: float) -> Score:
    return Score(dimension=dim, value=value, source=ScoreSource.HEURISTIC)


def _sweep(runs: list[SweepRun]) -> SweepResult:
    return SweepResult(sweep_id="test", runs=runs)


# --- Score-dimension regressions ---


def test_no_regression_when_candidate_matches_baseline():
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
    ])
    report = detect_regressions(result, baseline="base")
    assert report.has_regressions is False
    assert report.alerts == []
    # Delta should still be recorded for visibility
    assert report.deltas["cand"]["q"] == pytest.approx(0.0)


def test_regression_when_candidate_drops_below_threshold():
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.90)]),
        _run(0, 1, "cand", scores=[_score("q", 0.70)]),
    ])
    report = detect_regressions(result, baseline="base", threshold=0.05)
    assert report.has_regressions is True
    assert len(report.alerts) == 1
    a = report.alerts[0]
    assert a.config_name == "cand"
    assert a.dimension == "q"
    assert a.baseline_avg == pytest.approx(0.90)
    assert a.candidate_avg == pytest.approx(0.70)
    assert a.delta == pytest.approx(-0.20)
    assert a.threshold == 0.05


def test_no_regression_when_candidate_improves():
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.50)]),
        _run(0, 1, "cand", scores=[_score("q", 0.95)]),
    ])
    report = detect_regressions(result, baseline="base")
    assert report.has_regressions is False
    assert report.deltas["cand"]["q"] == pytest.approx(0.45)


def test_threshold_boundary_exact_does_not_alert():
    """Strict ``delta < -threshold`` — exactly -threshold passes.

    Uses powers-of-two values (0.5, 0.25) so the subtraction is
    exactly representable in float and the boundary really is hit.
    """
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.5)]),
        _run(0, 1, "cand", scores=[_score("q", 0.25)]),
    ])
    report = detect_regressions(result, baseline="base", threshold=0.25)
    assert report.deltas["cand"]["q"] == -0.25  # exact float
    assert report.has_regressions is False


def test_missing_dimension_in_candidate_skipped():
    """Candidate doesn't report a dim that baseline does — skip it."""
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80), _score("speed", 0.90)]),
        _run(0, 1, "cand", scores=[_score("q", 0.30)]),  # missing 'speed'
    ])
    report = detect_regressions(result, baseline="base", threshold=0.05)
    # Only 'q' regression alerts; 'speed' not present in candidate so skipped
    alert_dims = [a.dimension for a in report.alerts]
    assert "q" in alert_dims
    assert "speed" not in alert_dims
    # deltas dict shouldn't carry the skipped dim either
    assert "speed" not in report.deltas["cand"]


def test_negative_threshold_rejected():
    """Negative threshold inverts gate semantics silently — fail fast."""
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
    ])
    with pytest.raises(ValueError, match=r"threshold must be non-negative"):
        detect_regressions(result, baseline="base", threshold=-0.05)


def test_negative_success_rate_threshold_rejected():
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
    ])
    with pytest.raises(
        ValueError,
        match=r"success_rate_threshold must be non-negative",
    ):
        detect_regressions(
            result,
            baseline="base",
            threshold=0.05,
            success_rate_threshold=-0.1,
        )


def test_unknown_baseline_raises_helpful_error():
    result = _sweep([
        _run(0, 0, "alpha", scores=[_score("q", 0.80)]),
        _run(0, 1, "beta", scores=[_score("q", 0.70)]),
    ])
    with pytest.raises(ValueError, match=r"baseline config 'gamma' not present"):
        detect_regressions(result, baseline="gamma")


# --- success_rate regressions ---


def test_success_rate_regression_alerts():
    """Candidate fails 30% more often than baseline."""
    result = _sweep([
        # Baseline: all succeed (success_rate = 1.0)
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(1, 0, "base", scores=[_score("q", 0.80)]),
        _run(2, 0, "base", scores=[_score("q", 0.80)]),
        # Candidate: 1 of 3 errors (success_rate = 2/3)
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
        _run(1, 1, "cand", scores=[_score("q", 0.80)]),
        _run(2, 1, "cand", error=True),
    ])
    report = detect_regressions(result, baseline="base", threshold=0.05)
    sr_alerts = [a for a in report.alerts if a.dimension == "success_rate"]
    assert len(sr_alerts) == 1
    assert sr_alerts[0].baseline_avg == pytest.approx(1.0)
    assert sr_alerts[0].candidate_avg == pytest.approx(2 / 3)
    assert sr_alerts[0].delta == pytest.approx(-1 / 3)


def test_success_rate_uses_separate_threshold_when_supplied():
    """A ~30% drop passes the relaxed sr_threshold of 0.5."""
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(1, 0, "base", scores=[_score("q", 0.80)]),
        _run(2, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
        _run(1, 1, "cand", scores=[_score("q", 0.80)]),
        _run(2, 1, "cand", error=True),
    ])
    report = detect_regressions(
        result,
        baseline="base",
        threshold=0.05,
        success_rate_threshold=0.5,
    )
    sr_alerts = [a for a in report.alerts if a.dimension == "success_rate"]
    # 1/3 drop is bigger than 0.05 (would alert with default sr_threshold)
    # but smaller than 0.5 (relaxed threshold), so it should NOT alert
    assert sr_alerts == []


def test_success_rate_no_alert_when_candidate_improves_reliability():
    result = _sweep([
        _run(0, 0, "base", error=True),
        _run(1, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "cand", scores=[_score("q", 0.80)]),
        _run(1, 1, "cand", scores=[_score("q", 0.80)]),
    ])
    report = detect_regressions(result, baseline="base", threshold=0.05)
    sr_alerts = [a for a in report.alerts if a.dimension == "success_rate"]
    assert sr_alerts == []
    # Delta should be positive (candidate is more reliable)
    assert report.deltas["cand"]["success_rate"] == pytest.approx(0.5)


# --- Report shape ---


def test_has_regressions_property_reflects_alerts():
    empty = RegressionReport(baseline_config="x", alerts=[], deltas={})
    assert empty.has_regressions is False

    populated = RegressionReport(
        baseline_config="x",
        alerts=[
            RegressionAlert(
                config_name="y",
                dimension="q",
                baseline_avg=0.9,
                candidate_avg=0.7,
                delta=-0.2,
                threshold=0.05,
            )
        ],
        deltas={"y": {"q": -0.2}},
    )
    assert populated.has_regressions is True


def test_deltas_populated_for_all_candidates_even_without_alert():
    """CI dashboards want every delta visible, not just regressions."""
    result = _sweep([
        _run(0, 0, "base", scores=[_score("q", 0.80)]),
        _run(0, 1, "improver", scores=[_score("q", 0.95)]),
        _run(0, 2, "regressor", scores=[_score("q", 0.50)]),
    ])
    report = detect_regressions(result, baseline="base", threshold=0.05)
    # Only regressor alerts, but both deltas should be captured
    assert set(report.deltas.keys()) == {"improver", "regressor"}
    assert report.deltas["improver"]["q"] == pytest.approx(0.15)
    assert report.deltas["regressor"]["q"] == pytest.approx(-0.30)
