"""Tests for FeedbackLoop.query + the PastResults tool."""
from __future__ import annotations

import tempfile
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from jig import FeedbackQuery, PastResults, Score, ScoreSource
from jig.feedback.loop import SQLiteFeedbackLoop


async def _fake_embed(text: str) -> np.ndarray:
    """Deterministic fake embedding — no Ollama required.

    Seeds via ``hashlib.sha256`` so text → vector is stable across
    processes (Python's built-in ``hash()`` is salted per-process,
    which would make similarity-order assertions flaky).
    """
    import hashlib

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


@pytest.fixture
def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "test.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    return loop


async def _seed(loop: SQLiteFeedbackLoop, entries: list[dict]) -> list[str]:
    """Seed the loop with {content, input, metadata, scores} entries."""
    ids = []
    for e in entries:
        rid = await loop.store_result(e["content"], e["input"], e.get("metadata"))
        await loop.score(rid, e.get("scores", []))
        ids.append(rid)
    return ids


@pytest.mark.asyncio
class TestSQLiteFeedbackLoopQuery:
    async def test_empty_returns_empty(self, feedback_db):
        assert await feedback_db.query(FeedbackQuery()) == []

    async def test_filters_by_agent_name(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "A", "input": "a", "metadata": {"agent_name": "explorer"},
             "scores": [Score("q", 0.8, ScoreSource.HEURISTIC)]},
            {"content": "B", "input": "b", "metadata": {"agent_name": "refiner"},
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
        ])

        out = await feedback_db.query(FeedbackQuery(agent_name="explorer"))
        assert len(out) == 1
        assert out[0].content == "A"

    async def test_filters_by_model(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "S", "input": "x", "metadata": {"model": "claude-sonnet-4-5"},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
            {"content": "H", "input": "x", "metadata": {"model": "claude-haiku-4-5"},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
        ])

        out = await feedback_db.query(FeedbackQuery(model="claude-sonnet-4-5"))
        assert [r.content for r in out] == ["S"]

    async def test_filters_by_tags_intersection(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "X", "input": "i", "metadata": {"tags": ["mean_reversion", "fast"]},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
            {"content": "Y", "input": "i", "metadata": {"tags": ["breakout"]},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
            {"content": "Z", "input": "i", "metadata": {"tags": ["mean_reversion"]},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
        ])

        out = await feedback_db.query(FeedbackQuery(tags=["mean_reversion"]))
        assert sorted(r.content for r in out) == ["X", "Z"]

    async def test_filters_by_min_score(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "low", "input": "i", "metadata": {},
             "scores": [Score("q", 0.2, ScoreSource.HEURISTIC)]},
            {"content": "high", "input": "i", "metadata": {},
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
        ])

        out = await feedback_db.query(FeedbackQuery(min_score=0.5))
        assert [r.content for r in out] == ["high"]

    async def test_limit_caps_results(self, feedback_db):
        await _seed(feedback_db, [
            {"content": f"r{i}", "input": "i", "metadata": {},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]}
            for i in range(5)
        ])

        out = await feedback_db.query(FeedbackQuery(limit=2))
        assert len(out) == 2

    async def test_skips_results_without_scores(self, feedback_db):
        """store_result without a matching score row is not retrievable."""
        await feedback_db.store_result("unsorted", "i", {})
        await _seed(feedback_db, [
            {"content": "scored", "input": "i", "metadata": {},
             "scores": [Score("q", 0.5, ScoreSource.HEURISTIC)]},
        ])
        out = await feedback_db.query(FeedbackQuery())
        assert [r.content for r in out] == ["scored"]

    async def test_combines_filters(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "want",    "input": "i", "metadata": {"agent_name": "explorer", "tags": ["mr"]},
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
            {"content": "wrong-agent", "input": "i", "metadata": {"agent_name": "refiner", "tags": ["mr"]},
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
            {"content": "wrong-tag",   "input": "i", "metadata": {"agent_name": "explorer", "tags": ["other"]},
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
            {"content": "low-score",   "input": "i", "metadata": {"agent_name": "explorer", "tags": ["mr"]},
             "scores": [Score("q", 0.1, ScoreSource.HEURISTIC)]},
        ])

        out = await feedback_db.query(FeedbackQuery(
            agent_name="explorer",
            tags=["mr"],
            min_score=0.5,
        ))
        assert [r.content for r in out] == ["want"]


@pytest.mark.asyncio
class TestPastResultsTool:
    async def test_returns_no_prior_when_empty(self):
        from unittest.mock import AsyncMock
        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[])
        tool = PastResults(feedback)
        out = await tool.execute({"hypothesis": "try mean reversion"})
        assert "No prior results" in out

    async def test_formats_results_with_scores(self):
        from jig.core.types import ScoredResult
        from datetime import datetime

        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[
            ScoredResult(
                result_id="r1",
                content="Mean reversion on BTC worked well",
                scores=[Score("sharpe", 1.5, ScoreSource.HEURISTIC)],
                avg_score=1.5,
                metadata={},
                created_at=datetime.now(),
            ),
        ])
        tool = PastResults(feedback)
        out = await tool.execute({"hypothesis": "mean reversion"})
        assert "Found 1 prior" in out
        assert "1.50" in out
        assert "Mean reversion on BTC" in out

    async def test_scopes_by_agent_name(self):
        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[])
        tool = PastResults(feedback, agent_name="explorer")
        await tool.execute({"hypothesis": "x"})

        q: FeedbackQuery = feedback.query.call_args[0][0]
        assert q.agent_name == "explorer"

    async def test_respects_k_override(self):
        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[])
        tool = PastResults(feedback, default_k=5)
        await tool.execute({"hypothesis": "x", "k": 2})

        q: FeedbackQuery = feedback.query.call_args[0][0]
        assert q.limit == 2

    async def test_passes_min_score(self):
        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[])
        tool = PastResults(feedback)
        await tool.execute({"hypothesis": "x", "min_score": 0.7})

        q: FeedbackQuery = feedback.query.call_args[0][0]
        assert q.min_score == 0.7

    async def test_tool_definition_shape(self):
        feedback = AsyncMock()
        tool = PastResults(feedback)
        d = tool.definition
        assert d.name == "past_results"
        assert "hypothesis" in d.parameters["required"]

    async def test_rejects_non_integer_k(self):
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="k must be an integer"):
            await tool.execute({"hypothesis": "x", "k": "three"})

    async def test_rejects_non_positive_k(self):
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="positive integer"):
            await tool.execute({"hypothesis": "x", "k": 0})
        with pytest.raises(ValueError, match="positive integer"):
            await tool.execute({"hypothesis": "x", "k": -1})

    async def test_rejects_non_numeric_min_score(self):
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="min_score must be a number"):
            await tool.execute({"hypothesis": "x", "min_score": "high"})


class TestFeedbackQueryValidation:
    def test_rejects_non_positive_limit(self):
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=0)
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=-5)

    def test_rejects_non_integer_limit(self):
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=1.5)  # type: ignore[arg-type]
