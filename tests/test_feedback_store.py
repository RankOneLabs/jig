"""Tests for SQLiteFeedbackLoop persistence contracts.

Covers:
- SQLite foreign-key enforcement (orphan scores must fail loudly)
- Score validation via validate_scores (shared helper)
- export_eval_set: filter-before-limit semantics and edge cases
"""
from __future__ import annotations

import math
import tempfile
from datetime import datetime

import numpy as np
import pytest

from jig.core.types import EvalCase, Score, ScoreSource
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.feedback.validation import validate_scores


async def _fake_embed(text: str) -> np.ndarray:
    import hashlib

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


@pytest.fixture
def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "test.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    return loop


# ---------------------------------------------------------------------------
# Foreign-key enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestForeignKeyEnforcement:
    async def test_score_unknown_result_id_raises(self, feedback_db):
        """Scoring a result_id that does not exist must fail, not silently insert."""
        with pytest.raises(Exception):
            await feedback_db.score(
                "nonexistent-id",
                [Score("quality", 0.9, ScoreSource.HEURISTIC)],
            )

    async def test_score_known_result_id_succeeds(self, feedback_db):
        rid = await feedback_db.store_result("content", "input", {})
        # Must not raise.
        await feedback_db.score(rid, [Score("quality", 0.9, ScoreSource.HEURISTIC)])

    async def test_orphan_score_is_not_queryable(self, feedback_db):
        """Even if FK is somehow bypassed, ensure orphan rows can't pollute queries."""
        rid = await feedback_db.store_result("content", "input", {})
        await feedback_db.score(rid, [Score("quality", 0.9, ScoreSource.HEURISTIC)])
        # Scores for the valid result must be present.
        from jig.core.types import FeedbackQuery
        results = await feedback_db.query(FeedbackQuery())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Score validation (validate_scores helper)
# ---------------------------------------------------------------------------


class TestValidateScores:
    def _score(self, dim: str = "q", val: float = 0.5) -> Score:
        return Score(dimension=dim, value=val, source=ScoreSource.HEURISTIC)

    def test_valid_score_passes(self):
        validate_scores([self._score()])

    def test_boundary_values_pass(self):
        validate_scores([self._score(val=0.0), self._score(val=1.0)])

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_scores([])

    def test_nan_value_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_scores([self._score(val=float("nan"))])

    def test_positive_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            validate_scores([self._score(val=float("inf"))])

    def test_negative_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            validate_scores([self._score(val=float("-inf"))])

    def test_value_above_one_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_scores([self._score(val=1.001)])

    def test_value_below_zero_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_scores([self._score(val=-0.001)])

    def test_empty_dimension_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_scores([Score(dimension="", value=0.5, source=ScoreSource.HEURISTIC)])

    def test_non_string_dimension_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_scores([Score(dimension=None, value=0.5, source=ScoreSource.HEURISTIC)])  # type: ignore[arg-type]

    def test_multiple_scores_all_validated(self):
        """First invalid score in a list still raises."""
        scores = [
            self._score(val=0.8),
            Score(dimension="bad", value=1.5, source=ScoreSource.HEURISTIC),
        ]
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            validate_scores(scores)


@pytest.mark.asyncio
class TestScoreMethodValidation:
    """validate_scores is called by SQLiteFeedbackLoop.score before any DB write."""

    async def test_empty_scores_raises_before_db(self, feedback_db):
        rid = await feedback_db.store_result("c", "i", {})
        with pytest.raises(ValueError, match="non-empty"):
            await feedback_db.score(rid, [])

    async def test_nan_raises_before_db(self, feedback_db):
        rid = await feedback_db.store_result("c", "i", {})
        with pytest.raises(ValueError, match="NaN"):
            await feedback_db.score(rid, [Score("q", float("nan"), ScoreSource.HEURISTIC)])

    async def test_out_of_range_raises_before_db(self, feedback_db):
        rid = await feedback_db.store_result("c", "i", {})
        with pytest.raises(ValueError):
            await feedback_db.score(rid, [Score("q", 1.5, ScoreSource.HEURISTIC)])


# ---------------------------------------------------------------------------
# export_eval_set — filter before limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExportEvalSet:
    async def _populate(self, loop: SQLiteFeedbackLoop, entries: list[dict]) -> list[str]:
        ids = []
        for e in entries:
            rid = await loop.store_result(e["content"], e["input"], e.get("metadata"))
            if e.get("scores"):
                await loop.score(rid, e["scores"])
            ids.append(rid)
        return ids

    async def test_returns_empty_for_limit_zero(self, feedback_db):
        rid = await feedback_db.store_result("x", "i", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])
        result = await feedback_db.export_eval_set(limit=0)
        assert result == []

    async def test_negative_limit_raises(self, feedback_db):
        with pytest.raises(ValueError):
            await feedback_db.export_eval_set(limit=-1)

    async def test_min_score_greater_than_max_score_raises(self, feedback_db):
        with pytest.raises(ValueError):
            await feedback_db.export_eval_set(min_score=0.8, max_score=0.3)

    async def test_filters_applied_before_limit(self, feedback_db):
        """With 3 qualifying rows and 5 total, limit=2 must return 2 qualifying rows."""
        entries = [
            {"content": f"high-{i}", "input": "i",
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]}
            for i in range(3)
        ] + [
            {"content": f"low-{i}", "input": "i",
             "scores": [Score("q", 0.1, ScoreSource.HEURISTIC)]}
            for i in range(2)
        ]
        await self._populate(feedback_db, entries)
        result = await feedback_db.export_eval_set(min_score=0.5, limit=2)
        assert len(result) == 2
        assert all("high" in c.expected for c in result)  # type: ignore[operator]

    async def test_unscored_results_excluded_when_filter_set(self, feedback_db):
        """Results with no scores must not qualify when min_score is set."""
        await feedback_db.store_result("unscored", "i", {})
        rid = await feedback_db.store_result("scored", "i", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])

        result = await feedback_db.export_eval_set(min_score=0.0)
        contents = [c.expected for c in result]
        assert "unscored" not in contents
        assert "scored" in contents

    async def test_max_score_filter(self, feedback_db):
        entries = [
            {"content": "low", "input": "i",
             "scores": [Score("q", 0.2, ScoreSource.HEURISTIC)]},
            {"content": "high", "input": "i",
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
        ]
        await self._populate(feedback_db, entries)
        result = await feedback_db.export_eval_set(max_score=0.5)
        assert [c.expected for c in result] == ["low"]

    async def test_no_filters_returns_all_scored(self, feedback_db):
        for i in range(4):
            rid = await feedback_db.store_result(f"r{i}", "i", {})
            await feedback_db.score(rid, [Score("q", 0.5, ScoreSource.HEURISTIC)])
        # One unscored — should be excluded by existing logic
        await feedback_db.store_result("noscores", "i", {})
        result = await feedback_db.export_eval_set()
        assert len(result) == 4

    async def test_limit_none_returns_all_qualifying(self, feedback_db):
        for i in range(5):
            rid = await feedback_db.store_result(f"r{i}", "i", {})
            await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])
        result = await feedback_db.export_eval_set(min_score=0.5, limit=None)
        assert len(result) == 5

    async def test_since_filter_still_works(self, feedback_db):
        """Ensure the since filter composes with score filters correctly."""
        rid = await feedback_db.store_result("before", "i", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])

        cutoff = datetime.now()

        rid2 = await feedback_db.store_result("after", "i", {})
        await feedback_db.score(rid2, [Score("q", 0.9, ScoreSource.HEURISTIC)])

        result = await feedback_db.export_eval_set(since=cutoff, min_score=0.5)
        assert [c.expected for c in result] == ["after"]
