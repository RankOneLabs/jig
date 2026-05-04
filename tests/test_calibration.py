"""Tests for jig.eval.calibration — calibrate_judge + numpy
tie-aware Spearman.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jig.core.types import EvalCase, Grader, Score, ScoreSource
from jig.eval.calibration import (
    CalibrationReport,
    _avg_ranks,
    _safe_pearson,
    _tie_aware_spearman,
    calibrate_judge,
)


# --- Fakes ---


class _ScriptedJudge(Grader):
    """Returns a precomputed score per call, indexed by case.input.

    The mapping ``input -> score_value`` lets tests pair judge
    outputs with reference scores via the case's input string.
    """

    def __init__(self, scores_by_input: dict[str, float], dimension: str = "q"):
        self._scores = scores_by_input
        self._dim = dimension

    async def grade(
        self, input: Any, output: Any, context: dict[str, Any] | None = None
    ) -> list[Score]:
        return [
            Score(
                dimension=self._dim,
                value=self._scores[str(input)],
                source=ScoreSource.LLM_JUDGE,
            )
        ]


class _ConstantJudge(Grader):
    def __init__(self, value: float, dimension: str = "q"):
        self._value = value
        self._dim = dimension

    async def grade(
        self, input: Any, output: Any, context: dict[str, Any] | None = None
    ) -> list[Score]:
        return [
            Score(
                dimension=self._dim,
                value=self._value,
                source=ScoreSource.LLM_JUDGE,
            )
        ]


def _case(input_text: str, *, ref: float | None, expected: str = "candidate") -> EvalCase:
    meta: dict[str, Any] | None = None
    if ref is not None:
        meta = {"reference_score": ref}
    return EvalCase(input=input_text, expected=expected, metadata=meta)


# --- Pure-function tests for the numpy helpers ---


def test_avg_ranks_no_ties():
    assert list(_avg_ranks(np.array([10.0, 30.0, 20.0]))) == [1.0, 3.0, 2.0]


def test_avg_ranks_handles_ties():
    """[1, 2, 2, 3] → ranks [1, 2.5, 2.5, 4] (positions 2 and 3 averaged)."""
    ranks = _avg_ranks(np.array([1.0, 2.0, 2.0, 3.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_avg_ranks_all_equal():
    """Five identical values → all ranked 3.0 (mean of 1..5)."""
    ranks = _avg_ranks(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    assert list(ranks) == [3.0, 3.0, 3.0, 3.0, 3.0]


def test_safe_pearson_returns_zero_on_zero_variance():
    j = np.array([0.5, 0.5, 0.5, 0.5])
    r = np.array([0.1, 0.5, 0.7, 0.9])
    assert _safe_pearson(j, r) == 0.0
    assert _safe_pearson(r, j) == 0.0


def test_tie_aware_spearman_differs_from_naive_argsort_on_ties():
    """Tied judge scores: naive argsort-of-argsort assigns sequential
    ranks, biasing the correlation. Tie-aware ranks resolve it.
    """
    # Tied judge values cluster on 0/0.5/1; reference is monotone
    j = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    r = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
    naive = float(
        np.corrcoef(j.argsort().argsort(), r.argsort().argsort())[0, 1]
    )
    tie_aware = _tie_aware_spearman(j, r)
    # The two should disagree (the whole point of the tie-aware path)
    assert naive != pytest.approx(tie_aware)
    # Tie-aware Spearman on monotone data should still be high
    assert tie_aware > 0.9


def test_tie_aware_spearman_zero_variance_returns_zero():
    j = np.array([0.5, 0.5, 0.5])
    r = np.array([0.1, 0.5, 0.9])
    assert _tie_aware_spearman(j, r) == 0.0


# --- calibrate_judge ---


async def test_calibrate_judge_perfect_correlation():
    """Judge scores match reference exactly → Pearson and Spearman == 1."""
    cases = [
        _case("c1", ref=0.1),
        _case("c2", ref=0.4),
        _case("c3", ref=0.7),
        _case("c4", ref=0.95),
    ]
    judge = _ScriptedJudge({"c1": 0.1, "c2": 0.4, "c3": 0.7, "c4": 0.95})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert report.n_cases == 4
    assert report.pearson_r == pytest.approx(1.0)
    assert report.spearman_r == pytest.approx(1.0)
    assert report.mean_abs_error == pytest.approx(0.0)


async def test_calibrate_judge_uncorrelated_centered_around_zero():
    """Random-pattern judge scores → near-zero correlation."""
    rng = np.random.default_rng(7)
    n = 200
    cases = [_case(f"c{i}", ref=float(rng.random())) for i in range(n)]
    # Judge scores independent of references
    judge_vals = rng.random(n)
    judge = _ScriptedJudge({f"c{i}": float(judge_vals[i]) for i in range(n)})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert report.n_cases == n
    # With n=200 and uncorrelated uniform draws, |r| should be small
    assert abs(report.pearson_r) < 0.2
    assert abs(report.spearman_r) < 0.2


async def test_calibrate_judge_skips_unlabeled_cases():
    """Cases without reference_score are silently skipped."""
    cases = [
        _case("c1", ref=0.5),
        _case("c2", ref=None),  # no reference
        _case("c3", ref=0.8),
    ]
    judge = _ScriptedJudge({"c1": 0.5, "c2": 0.0, "c3": 0.8})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert report.n_cases == 2
    assert report.judge_scores == [0.5, 0.8]


async def test_calibrate_judge_skips_cases_missing_expected():
    cases = [
        _case("c1", ref=0.5, expected="x"),
        EvalCase(input="c2", expected=None, metadata={"reference_score": 0.7}),
    ]
    judge = _ScriptedJudge({"c1": 0.5, "c2": 0.7})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert report.n_cases == 1


async def test_calibrate_judge_too_few_cases_returns_empty_report():
    cases = [_case("c1", ref=0.5)]
    judge = _ScriptedJudge({"c1": 0.5})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert report.n_cases == 1
    assert report.pearson_r == 0.0
    assert report.spearman_r == 0.0
    assert report.mean_abs_error == 0.0


async def test_calibrate_judge_handles_constant_judge_scores():
    """Judge always says 1.0 → no variance → Pearson safely 0.0, not NaN."""
    cases = [_case(f"c{i}", ref=float(i) / 10) for i in range(10)]
    judge = _ConstantJudge(1.0)
    report = await calibrate_judge(judge, cases, dimension="q")
    # NaN guard kicks in
    assert report.pearson_r == 0.0
    assert report.spearman_r == 0.0
    # MAE is real and non-zero
    assert report.mean_abs_error > 0.0


async def test_calibrate_judge_skips_when_dimension_absent():
    cases = [_case(f"c{i}", ref=0.5) for i in range(5)]
    # Judge returns dimension 'other', not 'q'
    judge = _ScriptedJudge(
        {f"c{i}": 0.5 for i in range(5)},
        dimension="other",
    )
    report = await calibrate_judge(judge, cases, dimension="q")
    # No matches on 'q' → too-few-cases path
    assert report.n_cases == 0


async def test_calibrate_judge_returns_report_dataclass():
    cases = [_case(f"c{i}", ref=float(i) / 5) for i in range(6)]
    judge = _ScriptedJudge({f"c{i}": float(i) / 5 for i in range(6)})
    report = await calibrate_judge(judge, cases, dimension="q")
    assert isinstance(report, CalibrationReport)
    assert report.dimension == "q"
    assert len(report.judge_scores) == report.n_cases
    assert len(report.reference_scores) == report.n_cases
