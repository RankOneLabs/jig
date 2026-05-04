"""Tests for jig.sweep_stats — pass_at_k and win_rate over SweepResult.

Tests build SweepRun/AgentResult fakes directly. The functions under
test are read-only over SweepResult.runs.
"""
from __future__ import annotations

import warnings

import pytest

from jig import (
    PassAtK,
    Score,
    ScoreSource,
    SweepResult,
    SweepRun,
    WinRate,
    pass_at_k,
    win_rate,
)
from jig.core.runner import AgentResult


def _agent_result(scores: list[Score] | None) -> AgentResult:
    return AgentResult(
        output="ok",
        trace_id="t",
        usage={"total_cost": 0.0},
        scores=scores,
        duration_ms=1.0,
        error=None,
    )


def _score(dim: str, value: float) -> Score:
    return Score(dimension=dim, value=value, source=ScoreSource.HEURISTIC)


def _run(
    case_idx: int,
    config_idx: int,
    config_name: str,
    seed_idx: int,
    *,
    score_value: float,
    dim: str = "q",
) -> SweepRun:
    return SweepRun(
        case_index=case_idx,
        config_index=config_idx,
        config_name=config_name,
        input=f"input-{case_idx}",
        result=_agent_result([_score(dim, score_value)]),
        seed_index=seed_idx,
    )


def _sweep(runs: list[SweepRun]) -> SweepResult:
    return SweepResult(sweep_id="test", runs=runs)


# --- pass_at_k ---


def test_pass_at_k_perfect_passes():
    """All runs >= threshold → pak == 1.0."""
    # Slight per-seed variance keeps the constant-vector warning quiet
    # — the metric is the same either way; this just keeps the test
    # output clean.
    seed_jitter = [0.85, 0.90, 0.95, 0.99]
    runs = []
    for case_idx in range(3):
        for seed_idx in range(4):
            runs.append(
                _run(case_idx, 0, "cfg", seed_idx, score_value=seed_jitter[seed_idx])
            )
    result = _sweep(runs)
    out = pass_at_k(result, dimension="q", threshold=0.5)
    assert len(out) == 1
    pak = out[0]
    assert pak.config_name == "cfg"
    assert pak.k == 4
    assert pak.n_per_case == 4
    assert pak.n_cases == 3
    assert pak.pass_at_k == pytest.approx(1.0)


def test_pass_at_k_zero_passes():
    """All runs below threshold → pak == 0.0."""
    seed_jitter = [0.05, 0.10, 0.15, 0.20]
    runs = []
    for case_idx in range(3):
        for seed_idx in range(4):
            runs.append(
                _run(case_idx, 0, "cfg", seed_idx, score_value=seed_jitter[seed_idx])
            )
    result = _sweep(runs)
    out = pass_at_k(result, dimension="q", threshold=0.5)
    assert out[0].pass_at_k == pytest.approx(0.0)


def test_pass_at_k_partial_with_known_estimator_value():
    """Hand-computed: n=5, c=2 (40% pass rate). pass@2 = 0.7.

    Codex estimator: 1 - prod((n-c-i)/(n-i)) for i in 0..k-1
                   = 1 - (3/5)*(2/4) = 1 - 0.3 = 0.7.
    """
    # One case, 5 seeds, 2 passing (>= 0.5), 3 failing
    runs = [
        _run(0, 0, "cfg", 0, score_value=0.9),
        _run(0, 0, "cfg", 1, score_value=0.8),
        _run(0, 0, "cfg", 2, score_value=0.2),
        _run(0, 0, "cfg", 3, score_value=0.1),
        _run(0, 0, "cfg", 4, score_value=0.0),
    ]
    result = _sweep(runs)
    out = pass_at_k(result, dimension="q", threshold=0.5, k=2)
    assert out[0].pass_at_k == pytest.approx(0.7)


def test_pass_at_k_skips_non_uniform_sampling():
    """Different n_per_case across cases → config skipped cleanly."""
    runs = [
        # case 0: 3 seeds
        _run(0, 0, "cfg", 0, score_value=0.9),
        _run(0, 0, "cfg", 1, score_value=0.9),
        _run(0, 0, "cfg", 2, score_value=0.9),
        # case 1: 2 seeds (intentional asymmetry)
        _run(1, 0, "cfg", 0, score_value=0.9),
        _run(1, 0, "cfg", 1, score_value=0.9),
    ]
    result = _sweep(runs)
    out = pass_at_k(result, dimension="q", threshold=0.5)
    # cfg has non-uniform sampling, so it's omitted from the report
    assert out == []


def test_pass_at_k_warns_on_constant_score_vectors():
    """All seeds produce identical scores → RuntimeWarning + pak still computed."""
    runs = []
    for case_idx in range(2):
        for seed_idx in range(3):
            # Same value for every seed — simulates temperature=0
            runs.append(_run(case_idx, 0, "cfg", seed_idx, score_value=0.9))
    result = _sweep(runs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pass_at_k(result, dimension="q", threshold=0.5)
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "constant" in str(w.message)
        and "cfg" in str(w.message)
        for w in caught
    )
    # Metric still returned (it's well-defined, just degenerate)
    assert out[0].pass_at_k == pytest.approx(1.0)


def test_pass_at_k_does_not_warn_when_n_per_case_is_one():
    """n=1 is single-shot — no variance to expect, no warning."""
    runs = [_run(0, 0, "cfg", 0, score_value=0.9)]
    result = _sweep(runs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pass_at_k(result, dimension="q", threshold=0.5)
    assert not any(
        issubclass(w.category, RuntimeWarning) for w in caught
    )


def test_pass_at_k_rejects_non_positive_k():
    """``k <= 0`` is meaningless — fail fast at entry."""
    runs = [_run(0, 0, "cfg", 0, score_value=0.9)]
    result = _sweep(runs)
    with pytest.raises(ValueError, match=r"k must be positive"):
        pass_at_k(result, dimension="q", k=0)
    with pytest.raises(ValueError, match=r"k must be positive"):
        pass_at_k(result, dimension="q", k=-1)


def test_pass_at_k_warns_and_skips_incomplete_dimension_coverage():
    """A config that reported ``dimension`` on some but not all of
    its cases is skipped + warned about — silently dropping the
    missing cases would overstate pass@k by hiding hard cases.
    """
    runs = [
        # cfg_partial: ran 3 cases but only reported 'q' on 2 of them
        _run(0, 0, "cfg_partial", 0, score_value=0.9, dim="q"),
        _run(1, 0, "cfg_partial", 0, score_value=0.9, dim="q"),
        _run(2, 0, "cfg_partial", 0, score_value=0.9, dim="other"),
        # cfg_full: ran 3 cases with full coverage
        _run(0, 1, "cfg_full", 0, score_value=0.9, dim="q"),
        _run(1, 1, "cfg_full", 0, score_value=0.9, dim="q"),
        _run(2, 1, "cfg_full", 0, score_value=0.9, dim="q"),
    ]
    result = _sweep(runs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pass_at_k(result, dimension="q", threshold=0.5)
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "dimension missing on" in str(w.message)
        and "cfg_partial" in str(w.message)
        for w in caught
    )
    names = [p.config_name for p in out]
    assert "cfg_partial" not in names
    assert "cfg_full" in names


def test_pass_at_k_warns_when_k_exceeds_n_per_case():
    """Caller-supplied k > n_per_case → RuntimeWarning naming the
    config + skipped from result so it can't be silently empty.
    """
    runs = [
        _run(0, 0, "cfg", 0, score_value=0.85),
        _run(0, 0, "cfg", 1, score_value=0.95),
    ]
    result = _sweep(runs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pass_at_k(result, dimension="q", threshold=0.5, k=5)
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "exceeds n_per_case" in str(w.message)
        and "cfg" in str(w.message)
        for w in caught
    )
    assert out == []


def test_pass_at_k_aggregates_multiple_scores_per_run_via_mean():
    """A single run with multiple scores on the same dimension
    (possible with CompositeGrader) is aggregated via mean rather
    than silently using the first entry's order.
    """
    # Build a run whose AgentResult has TWO scores on dim "q".
    multi_score_result = AgentResult(
        output="ok",
        trace_id="t",
        usage={"total_cost": 0.0},
        scores=[
            _score("q", 0.0),
            _score("q", 1.0),  # mean = 0.5
        ],
        duration_ms=1.0,
        error=None,
    )
    multi_run = SweepRun(
        case_index=0,
        config_index=0,
        config_name="cfg",
        input="i",
        result=multi_score_result,
        seed_index=0,
    )
    # Use a different second value so the per-case vector has
    # variance — keeps the constant-vector warning from firing.
    other_run = _run(0, 0, "cfg", 1, score_value=0.9)
    result = _sweep([multi_run, other_run])
    out = pass_at_k(result, dimension="q", threshold=0.5)
    # Aggregated values: 0.5 (mean of 0/1) and 0.9 — both >= 0.5,
    # so passes_per_case=[2] and pass@2 == 1.0.
    assert out[0].pass_at_k == pytest.approx(1.0)


def test_pass_at_k_skips_config_missing_dimension():
    """A config that didn't report the requested dimension is omitted."""
    runs = [
        _run(0, 0, "with_q", 0, score_value=0.85, dim="q"),
        _run(0, 0, "with_q", 1, score_value=0.95, dim="q"),
        _run(0, 1, "without_q", 0, score_value=0.85, dim="other"),
        _run(0, 1, "without_q", 1, score_value=0.95, dim="other"),
    ]
    result = _sweep(runs)
    out = pass_at_k(result, dimension="q", threshold=0.5)
    names = [p.config_name for p in out]
    assert "with_q" in names
    assert "without_q" not in names


# --- win_rate ---


def test_win_rate_a_dominates():
    """A wins every case → win_rate == 1.0."""
    runs = []
    for case_idx in range(5):
        runs.append(_run(case_idx, 0, "A", 0, score_value=0.9))
        runs.append(_run(case_idx, 1, "B", 0, score_value=0.1))
    result = _sweep(runs)
    wr = win_rate(
        result,
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    assert wr.win_rate == pytest.approx(1.0)
    assert wr.n_compared == 5


def test_win_rate_all_ties_returns_zero_with_zero_decisive():
    """Strict pairwise: all-ties → n_decisive=0; win_rate is 0.0
    by convention (no decisive evidence either way).
    """
    runs = []
    for case_idx in range(5):
        runs.append(_run(case_idx, 0, "A", 0, score_value=0.5))
        runs.append(_run(case_idx, 1, "B", 0, score_value=0.5))
    result = _sweep(runs)
    wr = win_rate(
        result,
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    assert wr.win_rate == pytest.approx(0.0)
    assert wr.n_compared == 5
    assert wr.n_decisive == 0


def test_win_rate_strict_pairwise_excludes_ties_from_denominator():
    """1 A win + 1 tie → win_rate = 1/1 = 1.0 (tie not in denom).
    With the old loose semantic this would have been 0.5.
    """
    runs = [
        # case 0: A wins decisively
        _run(0, 0, "A", 0, score_value=0.9),
        _run(0, 1, "B", 0, score_value=0.1),
        # case 1: tie — should not count in denominator
        _run(1, 0, "A", 0, score_value=0.5),
        _run(1, 1, "B", 0, score_value=0.5),
    ]
    result = _sweep(runs)
    wr = win_rate(
        result,
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    assert wr.win_rate == pytest.approx(1.0)
    assert wr.n_compared == 2
    assert wr.n_decisive == 1


def test_win_rate_ci_widens_with_fewer_cases():
    """A bootstrap CI computed over fewer cases is wider."""
    # 3 of 5 cases A wins — same point estimate at both sample sizes
    def _build_runs(n_cases: int) -> list[SweepRun]:
        runs = []
        for case_idx in range(n_cases):
            a_score = 0.9 if case_idx < (n_cases * 3 // 5) else 0.1
            runs.append(_run(case_idx, 0, "A", 0, score_value=a_score))
            runs.append(_run(case_idx, 1, "B", 0, score_value=0.5))
        return runs

    small = win_rate(
        _sweep(_build_runs(5)),
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    large = win_rate(
        _sweep(_build_runs(50)),
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    small_width = small.ci_high - small.ci_low
    large_width = large.ci_high - large.ci_low
    assert small_width > large_width


def test_win_rate_seeded_bootstrap_is_reproducible():
    """Same seed → identical CI bounds across calls."""
    runs = []
    for case_idx in range(8):
        a = 0.9 if case_idx % 2 == 0 else 0.1
        runs.append(_run(case_idx, 0, "A", 0, score_value=a))
        runs.append(_run(case_idx, 1, "B", 0, score_value=0.5))
    result = _sweep(runs)
    wr1 = win_rate(result, dimension="q", config_a="A", config_b="B", seed=7)
    wr2 = win_rate(result, dimension="q", config_a="A", config_b="B", seed=7)
    assert wr1.ci_low == wr2.ci_low
    assert wr1.ci_high == wr2.ci_high


def test_win_rate_rejects_non_positive_bootstrap_samples():
    """``bootstrap_samples <= 0`` would crash on np.percentile([])."""
    runs = [
        _run(0, 0, "A", 0, score_value=0.9),
        _run(0, 1, "B", 0, score_value=0.1),
    ]
    result = _sweep(runs)
    with pytest.raises(ValueError, match=r"bootstrap_samples"):
        win_rate(
            result,
            dimension="q",
            config_a="A",
            config_b="B",
            bootstrap_samples=0,
        )
    with pytest.raises(ValueError, match=r"bootstrap_samples"):
        win_rate(
            result,
            dimension="q",
            config_a="A",
            config_b="B",
            bootstrap_samples=-1,
        )


def test_win_rate_aggregates_within_run_duplicates_via_mean():
    """Multiple Score entries on the same dim within a single run
    (e.g. CompositeGrader) collapse to mean, not extend — otherwise
    that run is double-counted in the per-case average.
    """
    # A's run has two scores on 'q' that mean to 0.5; B has one score 0.4.
    a_run = SweepRun(
        case_index=0,
        config_index=0,
        config_name="A",
        input="x",
        result=AgentResult(
            output="ok",
            trace_id="t",
            usage={"total_cost": 0.0},
            scores=[_score("q", 0.0), _score("q", 1.0)],  # mean 0.5
            duration_ms=1.0,
            error=None,
        ),
        seed_index=0,
    )
    b_run = _run(0, 1, "B", 0, score_value=0.4)
    result = _sweep([a_run, b_run])
    wr = win_rate(
        result, dimension="q", config_a="A", config_b="B", seed=42
    )
    # A's per-case mean is 0.5, B's is 0.4 → A wins decisively.
    assert wr.win_rate == pytest.approx(1.0)
    assert wr.n_decisive == 1


def test_win_rate_no_overlap_returns_zero_compared():
    """Configs that don't share any case → no comparison possible."""
    runs = [
        _run(0, 0, "A", 0, score_value=0.9),
        _run(1, 1, "B", 0, score_value=0.1),
    ]
    result = _sweep(runs)
    wr = win_rate(
        result,
        dimension="q",
        config_a="A",
        config_b="B",
        seed=42,
    )
    assert wr.n_compared == 0
    assert wr.win_rate == 0.0
    assert wr.ci_low == 0.0
    assert wr.ci_high == 0.0
