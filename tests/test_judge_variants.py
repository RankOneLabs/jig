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


async def test_pairwise_position_randomized_across_inputs():
    """Position flip is a stable function of (seed, other_id, input).
    Across many distinct inputs, the flip distribution should be
    roughly 50/50 — that's the position-bias mitigation guarantee.
    """
    judge = PairwiseLLMJudge(_FakeLLM(["A"] * 200), seed=42)
    wins = 0
    for i in range(200):
        scores = await judge.grade(
            f"input_{i}",  # vary input across calls
            "self_output",
            context={"compare_to": {"output": "other_output", "id": "other"}},
        )
        if scores[0].value == 1.0:
            wins += 1
    # Mock LLM always picks "A". With ~50/50 position flip, self
    # ends up in A about half the time → ~100 wins.
    assert 70 < wins < 130


async def test_pairwise_same_inputs_produce_same_flip():
    """The new design: identical (seed, other_id, input) MUST produce
    the same A/B assignment so concurrent calls in any order are
    reproducible.
    """
    judge = PairwiseLLMJudge(_FakeLLM(["A"] * 10), seed=42)
    results = []
    for _ in range(10):
        scores = await judge.grade(
            "same_input",
            "self_output",
            context={"compare_to": {"output": "other", "id": "x"}},
        )
        results.append(scores[0].value)
    # All 10 calls have identical inputs → all 10 must produce the
    # same outcome. This is the property that breaks under shared
    # mutable RNG with concurrent dispatch.
    assert len(set(results)) == 1


async def test_pairwise_concurrent_calls_reproducible():
    """asyncio.gather over identical-input calls must produce the
    same A/B assignments regardless of dispatch order — the property
    a shared mutable RNG can't guarantee.
    """
    import asyncio

    judge = PairwiseLLMJudge(_FakeLLM(["A"] * 20), seed=99)
    coros = [
        judge.grade(
            "x",
            "self",
            context={"compare_to": {"output": "other", "id": "z"}},
        )
        for _ in range(20)
    ]
    results = await asyncio.gather(*coros)
    values = [r[0].value for r in results]
    # All identical inputs → identical outcome under deterministic
    # per-call RNG, regardless of how gather scheduled them.
    assert len(set(values)) == 1


async def test_pairwise_seed_reproducibility_across_judge_instances():
    """Two separate judge instances with the same seed produce the
    same outcome sequence on the same inputs — the seed is the only
    state that matters now.
    """
    inputs = [(f"i_{i}", f"id_{i}") for i in range(10)]

    async def run(seed: int) -> list[float]:
        judge = PairwiseLLMJudge(_FakeLLM(["A"] * 10), seed=seed)
        out = []
        for inp, oid in inputs:
            scores = await judge.grade(
                inp,
                "self",
                context={"compare_to": {"output": "other", "id": oid}},
            )
            out.append(scores[0].value)
        return out

    a = await run(123)
    b = await run(123)
    assert a == b


async def test_pairwise_unseeded_uses_module_random():
    """Without a seed, randomization still happens (just not
    reproducible). This is the documented escape hatch.
    """
    judge = PairwiseLLMJudge(_FakeLLM(["A"] * 200), seed=None)
    wins = 0
    for i in range(200):
        scores = await judge.grade(
            f"input_{i}",
            "self",
            context={"compare_to": {"output": "other", "id": "x"}},
        )
        if scores[0].value == 1.0:
            wins += 1
    # Random module's flip + always-A judge → ~half wins
    assert 60 < wins < 140  # wider tolerance: process-random


async def test_pairwise_tie_returns_half():
    judge = PairwiseLLMJudge(_FakeLLM(["tie"]), seed=0)
    scores = await judge.grade(
        "i", "self", context={"compare_to": {"output": "other", "id": "z"}}
    )
    assert scores[0].dimension == "vs_z"
    assert scores[0].value == 0.5
    assert scores[0].source == ScoreSource.LLM_JUDGE


async def test_pairwise_unknown_winner_value_returns_tie():
    """JSON parses but ``winner`` isn't A/B/tie → score as a tie,
    not as a loss. A meaningless verdict is "no opinion."
    """
    judge = PairwiseLLMJudge(_FakeLLM(["neither"]), seed=0)
    scores = await judge.grade(
        "i", "self", context={"compare_to": {"output": "other", "id": "z"}}
    )
    assert scores[0].dimension == "vs_z"
    assert scores[0].value == 0.5


async def test_pairwise_distinguishes_none_from_empty_criteria():
    """``criteria=None`` uses default; explicit ``[]`` renders empty.

    Inspect the formatted system prompt by reading the user msg the
    fake LLM captured (it's the easier-to-introspect side); the
    point is to confirm None and [] don't both produce the default.
    """
    # The criteria show up only in the system prompt, but we can
    # still distinguish behaviors by checking the internal state.
    j_default = PairwiseLLMJudge(_FakeLLM(["A"]), criteria=None, seed=0)
    j_empty = PairwiseLLMJudge(_FakeLLM(["A"]), criteria=[], seed=0)
    assert j_default._criteria == ["overall quality"]
    assert j_empty._criteria == []


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


async def test_committee_collapses_per_judge_duplicate_dimensions():
    """A single judge returning two ``q`` scores must NOT count as
    two votes — each judge contributes one value per dimension.
    """
    # Judge 1 returns TWO scores on 'q' (0.0 and 1.0 → mean 0.5).
    # Judge 2 returns one score on 'q' (0.5).
    # Without per-judge dedupe: by_dim['q'] = [0.0, 1.0, 0.5] mean=0.5
    # but stddev would be artificially high.
    # With dedupe: by_dim['q'] = [0.5, 0.5] mean=0.5 stddev=0 (unanimous).
    committee = CommitteeJudge([
        _FixedGrader([
            Score(dimension="q", value=0.0, source=ScoreSource.LLM_JUDGE),
            Score(dimension="q", value=1.0, source=ScoreSource.LLM_JUDGE),
        ]),
        _FixedGrader([
            Score(dimension="q", value=0.5, source=ScoreSource.LLM_JUDGE),
        ]),
    ])
    scores = await committee.grade("i", "o")
    by_dim = {s.dimension: s.value for s in scores}
    assert by_dim["q"] == pytest.approx(0.5)
    # Both judges effectively voted 0.5 — full agreement.
    assert by_dim["q_agreement"] == pytest.approx(1.0)


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
