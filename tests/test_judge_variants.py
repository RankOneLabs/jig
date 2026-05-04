"""Tests for PairwiseLLMJudge and CommitteeJudge."""
from __future__ import annotations

import json
import warnings
from typing import Any

import pytest

from jig import (
    CommitteeJudge,
    PairwiseLLMJudge,
    Score,
    ScoreSource,
)
from jig.core.types import (
    CompletionParams,
    Grader,
    LLMClient,
    LLMResponse,
    Usage,
)


# --- Fakes ---


class _FakeLLM(LLMClient):
    """Returns a scripted JSON-encoded judge verdict per call.

    Tracks ``last_user_msg`` so tests can assert on which output
    was shown in position A vs B after the judge's randomization.
    """

    def __init__(self, verdicts: list[str]):
        # Each verdict is one of "A", "B", "tie" — wrapped into JSON.
        self._verdicts = list(verdicts)
        self._call = 0
        self.last_user_msg: str | None = None

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.last_user_msg = params.messages[-1].content
        verdict = self._verdicts[self._call % len(self._verdicts)]
        self._call += 1
        body = json.dumps({"winner": verdict, "reasoning": "test"})
        return LLMResponse(
            content=body,
            tool_calls=None,
            usage=Usage(1, 1, cost=0.0),
            latency_ms=1.0,
            model="fake",
        )


class _BadJsonLLM(LLMClient):
    async def complete(self, params: CompletionParams) -> LLMResponse:
        return LLMResponse(
            content="this is not json",
            tool_calls=None,
            usage=Usage(1, 1, cost=0.0),
            latency_ms=1.0,
            model="fake",
        )


class _FixedGrader(Grader):
    """Returns the same scores every call. Used to build committees."""

    def __init__(self, scores: list[Score]):
        self._scores = scores

    async def grade(
        self, input: Any, output: Any, context: dict[str, Any] | None = None
    ) -> list[Score]:
        return list(self._scores)


# --- PairwiseLLMJudge ---


async def test_pairwise_returns_empty_without_compare_to():
    judge = PairwiseLLMJudge(_FakeLLM(["A"]), seed=0)
    assert await judge.grade("input", "output", context=None) == []
    assert await judge.grade("input", "output", context={}) == []


async def test_pairwise_handles_compare_to_missing_keys():
    judge = PairwiseLLMJudge(_FakeLLM(["A"]), seed=0)
    # compare_to without 'output' or 'id' is ill-formed — fail soft
    assert await judge.grade(
        "i", "o", context={"compare_to": {"output": "x"}}
    ) == []
    assert await judge.grade(
        "i", "o", context={"compare_to": {"id": "x"}}
    ) == []


async def test_pairwise_position_randomized_self_wins():
    """Mock LLM that always picks 'A'. With 100 calls and a fair RNG,
    self should win roughly half the time (position is randomized)."""
    judge = PairwiseLLMJudge(_FakeLLM(["A"] * 200), seed=42)
    wins = 0
    for _ in range(200):
        scores = await judge.grade(
            "input",
            "self_output",
            context={"compare_to": {"output": "other_output", "id": "other"}},
        )
        if scores[0].value == 1.0:
            wins += 1
    # Expect roughly 100/200 — generous tolerance for the RNG
    assert 70 < wins < 130


async def test_pairwise_seed_reproducibility():
    """Same seed → identical outcome sequence."""

    async def run_seq(seed: int) -> list[float]:
        judge = PairwiseLLMJudge(_FakeLLM(["A"] * 10), seed=seed)
        out = []
        for _ in range(10):
            scores = await judge.grade(
                "i",
                "self",
                context={"compare_to": {"output": "other", "id": "x"}},
            )
            out.append(scores[0].value)
        return out

    a = await run_seq(123)
    b = await run_seq(123)
    assert a == b


async def test_pairwise_tie_returns_half():
    judge = PairwiseLLMJudge(_FakeLLM(["tie"]), seed=0)
    scores = await judge.grade(
        "i", "self", context={"compare_to": {"output": "other", "id": "z"}}
    )
    assert scores[0].dimension == "vs_z"
    assert scores[0].value == 0.5
    assert scores[0].source == ScoreSource.LLM_JUDGE


async def test_pairwise_handles_malformed_judge_response():
    """Bad JSON from the judge → score as a tie, don't raise."""
    judge = PairwiseLLMJudge(_BadJsonLLM(), seed=0)
    scores = await judge.grade(
        "i", "self", context={"compare_to": {"output": "other", "id": "z"}}
    )
    assert len(scores) == 1
    assert scores[0].value == 0.5  # tie fallback
    assert scores[0].dimension == "vs_z"


async def test_pairwise_dimension_uses_other_id():
    """The score dimension carries the comparison id so multiple
    pairwise comparisons in one context don't collide."""
    judge = PairwiseLLMJudge(_FakeLLM(["A", "A"]), seed=0)
    a = await judge.grade(
        "i", "self", context={"compare_to": {"output": "x", "id": "alpha"}}
    )
    b = await judge.grade(
        "i", "self", context={"compare_to": {"output": "y", "id": "beta"}}
    )
    assert a[0].dimension == "vs_alpha"
    assert b[0].dimension == "vs_beta"


# --- CommitteeJudge ---


def test_committee_requires_non_empty_judges():
    with pytest.raises(ValueError, match="at least one judge"):
        CommitteeJudge([])


async def test_committee_mean_aggregation():
    """Three judges with values 0.6, 0.8, 1.0 → mean 0.8."""
    committee = CommitteeJudge([
        _FixedGrader([Score(dimension="q", value=0.6, source=ScoreSource.LLM_JUDGE)]),
        _FixedGrader([Score(dimension="q", value=0.8, source=ScoreSource.LLM_JUDGE)]),
        _FixedGrader([Score(dimension="q", value=1.0, source=ScoreSource.LLM_JUDGE)]),
    ])
    scores = await committee.grade("i", "o")
    by_dim = {s.dimension: s.value for s in scores}
    assert by_dim["q"] == pytest.approx(0.8)


async def test_committee_agreement_dimension_added():
    """Unanimous judges → agreement == 1.0; agreement Score added per dim."""
    committee = CommitteeJudge([
        _FixedGrader([Score(dimension="q", value=0.7, source=ScoreSource.LLM_JUDGE)]),
        _FixedGrader([Score(dimension="q", value=0.7, source=ScoreSource.LLM_JUDGE)]),
    ])
    scores = await committee.grade("i", "o")
    by_dim = {s.dimension: s.value for s in scores}
    assert "q" in by_dim
    assert "q_agreement" in by_dim
    assert by_dim["q_agreement"] == pytest.approx(1.0)


async def test_committee_handles_disagreement():
    """Three judges with one outlier — mean dampens, agreement < 1."""
    committee = CommitteeJudge([
        _FixedGrader([Score(dimension="q", value=0.9, source=ScoreSource.LLM_JUDGE)]),
        _FixedGrader([Score(dimension="q", value=0.9, source=ScoreSource.LLM_JUDGE)]),
        _FixedGrader([Score(dimension="q", value=0.0, source=ScoreSource.LLM_JUDGE)]),
    ])
    scores = await committee.grade("i", "o")
    by_dim = {s.dimension: s.value for s in scores}
    # Mean of 0.9, 0.9, 0.0 = 0.6
    assert by_dim["q"] == pytest.approx(0.6)
    # Agreement signal should be well below 1.0
    assert by_dim["q_agreement"] < 0.5


async def test_committee_warns_on_dimension_set_mismatch():
    """Judges returning different dim sets → RuntimeWarning + average
    over only the judges that reported each dim."""
    committee = CommitteeJudge([
        _FixedGrader([
            Score(dimension="q", value=0.8, source=ScoreSource.LLM_JUDGE),
            Score(dimension="speed", value=0.5, source=ScoreSource.LLM_JUDGE),
        ]),
        _FixedGrader([
            Score(dimension="q", value=0.6, source=ScoreSource.LLM_JUDGE),
            # Missing 'speed'
        ]),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = await committee.grade("i", "o")
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "different dimension sets" in str(w.message)
        for w in caught
    )
    by_dim = {s.dimension: s.value for s in scores}
    # 'q' averaged over both judges
    assert by_dim["q"] == pytest.approx(0.7)
    # 'speed' only reported by the first judge → mean is 0.5
    assert by_dim["speed"] == pytest.approx(0.5)
