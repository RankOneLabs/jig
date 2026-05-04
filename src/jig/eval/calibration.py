"""Judge calibration harness.

Run a judge against a labeled holdout (cases with a known
human/ground-truth reference score on the candidate output) and
report agreement metrics. Use to decide whether a judge is reliable
enough to trust on unlabeled data, and to compare candidate judge
prompts.

**Calibration semantics.** The harness asks the judge to score
``case.expected`` as if it were the agent's output, then compares
the judge's score to the human label stored in
``case.metadata[reference_key]``. So ``case.expected`` here is the
*candidate output the judge sees*, not "the answer the agent should
produce." This is a deliberate overload of the field — calibration
sets are typically built by replaying production outputs and
labeling them, so reusing ``expected`` keeps the on-disk format
compatible with regular eval sets.

No scipy dependency: tie-aware Spearman is implemented in numpy
(average ranks for ties, then Pearson on ranks). Calibration was
the only would-be consumer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jig.core.types import EvalCase, Grader


@dataclass
class CalibrationReport:
    n_cases: int
    dimension: str
    pearson_r: float
    spearman_r: float
    mean_abs_error: float
    judge_scores: list[float]
    reference_scores: list[float]


def _safe_pearson(j: np.ndarray, r: np.ndarray) -> float:
    """Pearson that returns 0.0 when either array has zero variance.

    ``np.corrcoef`` returns NaN in that case, which propagates badly
    through downstream reporting. Treating it as zero correlation is
    the right semantic — there is literally no co-variation to
    measure.
    """
    if j.std() == 0.0 or r.std() == 0.0:
        return 0.0
    val = float(np.corrcoef(j, r)[0, 1])
    return 0.0 if math.isnan(val) else val


def _avg_ranks(values: np.ndarray) -> np.ndarray:
    """Return ranks where ties get the average of their positions.

    Equivalent to ``scipy.stats.rankdata(values, method="average")``,
    written in numpy to avoid the scipy dependency. The fastest
    correct implementation: stable sort + scan for tie groups +
    assign each group the average of its 1-indexed positions.
    """
    n = len(values)
    sorted_idx = np.argsort(values, kind="stable")
    sorted_vals = values[sorted_idx]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # Positions i..j-1 are tied; average rank is the mean of
        # their 1-indexed positions: (i+1 + j) / 2.
        avg = (i + 1 + j) / 2.0
        ranks[sorted_idx[i:j]] = avg
        i = j
    return ranks


def _tie_aware_spearman(j: np.ndarray, r: np.ndarray) -> float:
    """Spearman correlation honoring tied ranks.

    Naive ``argsort().argsort()`` assigns ties sequential ranks,
    which biases the correlation when judge scores cluster on a few
    discrete values (very common: ``0.0/0.5/1.0``). This computes
    average ranks for tied values and then Pearson on the ranks,
    which is the textbook tie-corrected Spearman.
    """
    if len(j) < 2 or j.std() == 0.0 or r.std() == 0.0:
        return 0.0
    rj = _avg_ranks(j)
    rr = _avg_ranks(r)
    val = float(np.corrcoef(rj, rr)[0, 1])
    return 0.0 if math.isnan(val) else val


async def calibrate_judge(
    judge: Grader,
    cases: list[EvalCase],
    *,
    dimension: str,
    reference_key: str = "reference_score",
) -> CalibrationReport:
    """Score each case with ``judge``, compare against references.

    For each case with both ``case.expected`` (the candidate output
    the judge will score) and ``case.metadata[reference_key]`` (the
    human/ground-truth label), invokes the judge and records the
    score on ``dimension``. Cases missing either side are skipped.

    Returns Pearson, tie-aware Spearman, and MAE between the judge's
    scores and the references. With fewer than 2 comparable cases,
    returns a zero-everywhere report — caller can branch on
    ``n_cases``.
    """
    judge_vals: list[float] = []
    ref_vals: list[float] = []
    for case in cases:
        if not case.metadata or reference_key not in case.metadata:
            continue
        if case.expected is None:
            continue
        scores = await judge.grade(
            case.input,
            case.expected,
            context=case.context,
        )
        match = next((s for s in scores if s.dimension == dimension), None)
        if match is None:
            continue
        judge_vals.append(float(match.value))
        ref_vals.append(float(case.metadata[reference_key]))

    if len(judge_vals) < 2:
        return CalibrationReport(
            n_cases=len(judge_vals),
            dimension=dimension,
            pearson_r=0.0,
            spearman_r=0.0,
            mean_abs_error=0.0,
            judge_scores=judge_vals,
            reference_scores=ref_vals,
        )

    j = np.array(judge_vals)
    r = np.array(ref_vals)
    pearson = _safe_pearson(j, r)
    spearman = _tie_aware_spearman(j, r)
    mae = float(np.mean(np.abs(j - r)))

    return CalibrationReport(
        n_cases=len(judge_vals),
        dimension=dimension,
        pearson_r=pearson,
        spearman_r=spearman,
        mean_abs_error=mae,
        judge_scores=judge_vals,
        reference_scores=ref_vals,
    )
