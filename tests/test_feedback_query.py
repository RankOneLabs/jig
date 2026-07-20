"""Tests for FeedbackLoop.query + the PastResults tool."""
from __future__ import annotations

import tempfile
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from jig import EffectiveScoreFilter, FeedbackQuery, PastResults, Score, ScoreSource
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
async def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "test.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    try:
        yield loop
    finally:
        await loop.close()


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

    async def test_rejects_bool_k(self):
        """bool is an int subclass; True would silently mean k=1."""
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="k must be an integer"):
            await tool.execute({"hypothesis": "x", "k": True})

    async def test_rejects_non_integral_float_k(self):
        """2.7 silently becoming 2 is a footgun."""
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="k must be an integer"):
            await tool.execute({"hypothesis": "x", "k": 2.7})

    async def test_accepts_integral_float_k(self):
        """5.0 is fine — intent is unambiguous."""
        feedback = AsyncMock()
        feedback.query = AsyncMock(return_value=[])
        tool = PastResults(feedback)
        await tool.execute({"hypothesis": "x", "k": 5.0})

    async def test_rejects_bool_min_score(self):
        tool = PastResults(AsyncMock())
        with pytest.raises(ValueError, match="min_score must be a number"):
            await tool.execute({"hypothesis": "x", "min_score": True})


@pytest.mark.asyncio
class TestSourceFilteringParity:
    """Source filtering and min-score behave identically through query() and get_signals()."""

    async def test_source_filter_returns_only_matching_source(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [
            Score("q", 0.9, ScoreSource.LLM_JUDGE),
            Score("q", 0.5, ScoreSource.HEURISTIC),
        ])

        out = await feedback_db.query(FeedbackQuery(source=ScoreSource.LLM_JUDGE))
        assert len(out) == 1
        assert all(s.source == ScoreSource.LLM_JUDGE for s in out[0].scores)

    async def test_source_filter_drops_results_with_no_matching_source(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(source=ScoreSource.LLM_JUDGE))
        assert out == []

    async def test_avg_score_computed_over_source_filtered_scores(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [
            Score("q", 0.8, ScoreSource.LLM_JUDGE),
            Score("q", 0.2, ScoreSource.HEURISTIC),
        ])

        out = await feedback_db.query(FeedbackQuery(source=ScoreSource.LLM_JUDGE))
        assert len(out) == 1
        assert out[0].avg_score == pytest.approx(0.8)

    async def test_get_signals_and_query_agree_on_source_filter(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "alpha", "input": "signal query",
             "scores": [Score("q", 0.9, ScoreSource.LLM_JUDGE)]},
            {"content": "beta", "input": "signal query",
             "scores": [Score("q", 0.7, ScoreSource.HEURISTIC)]},
            {"content": "gamma", "input": "signal query",
             "scores": [
                 Score("q", 0.8, ScoreSource.LLM_JUDGE),
                 Score("q", 0.3, ScoreSource.HEURISTIC),
             ]},
        ])

        via_query = await feedback_db.query(
            FeedbackQuery(similar_to="signal query", source=ScoreSource.LLM_JUDGE, limit=10)
        )
        via_signals = await feedback_db.get_signals(
            "signal query", limit=10, source=ScoreSource.LLM_JUDGE
        )

        assert {r.content for r in via_query} == {r.content for r in via_signals}
        assert {r.content for r in via_query} == {"alpha", "gamma"}

    async def test_min_score_identical_through_both_apis(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "high", "input": "min score query",
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
            {"content": "low", "input": "min score query",
             "scores": [Score("q", 0.3, ScoreSource.HEURISTIC)]},
        ])

        via_query = await feedback_db.query(
            FeedbackQuery(similar_to="min score query", min_score=0.5, limit=10)
        )
        via_signals = await feedback_db.get_signals(
            "min score query", limit=10, min_score=0.5
        )

        assert {r.content for r in via_query} == {r.content for r in via_signals}
        assert {r.content for r in via_query} == {"high"}

    async def test_source_and_min_score_combined_through_both_apis(self, feedback_db):
        await _seed(feedback_db, [
            {"content": "want", "input": "combo query",
             "scores": [Score("q", 0.9, ScoreSource.LLM_JUDGE)]},
            {"content": "wrong-source", "input": "combo query",
             "scores": [Score("q", 0.9, ScoreSource.HEURISTIC)]},
            {"content": "low-score", "input": "combo query",
             "scores": [Score("q", 0.2, ScoreSource.LLM_JUDGE)]},
        ])

        via_query = await feedback_db.query(
            FeedbackQuery(
                similar_to="combo query",
                source=ScoreSource.LLM_JUDGE,
                min_score=0.5,
                limit=10,
            )
        )
        via_signals = await feedback_db.get_signals(
            "combo query", limit=10, min_score=0.5, source=ScoreSource.LLM_JUDGE
        )

        assert {r.content for r in via_query} == {r.content for r in via_signals}
        assert {r.content for r in via_query} == {"want"}


@pytest.mark.asyncio
class TestScoreMetadataRoundTrip:
    async def test_query_returns_score_metadata(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [
            Score("q", 0.9, ScoreSource.HEURISTIC, metadata={"offending_claim": "x"}),
        ])

        out = await feedback_db.query(FeedbackQuery())
        assert len(out) == 1
        assert out[0].scores[0].metadata == {"offending_claim": "x"}

    async def test_query_returns_none_metadata_for_scores_without_it(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery())
        assert out[0].scores[0].metadata is None

    async def test_query_preserves_score_insertion_order(self, feedback_db):
        rid = await feedback_db.store_result("A", "query text", {})
        await feedback_db.score(rid, [
            Score("first", 0.1, ScoreSource.HEURISTIC),
            Score("second", 0.2, ScoreSource.HEURISTIC),
            Score("third", 0.3, ScoreSource.HEURISTIC),
        ])

        out = await feedback_db.query(FeedbackQuery())
        assert [s.dimension for s in out[0].scores] == ["first", "second", "third"]

    async def test_get_signals_also_round_trips_metadata(self, feedback_db):
        rid = await feedback_db.store_result("A", "signal text", {})
        await feedback_db.score(rid, [
            Score("q", 0.9, ScoreSource.HEURISTIC, metadata={"note": "y"}),
        ])

        out = await feedback_db.get_signals("signal text")
        assert out[0].scores[0].metadata == {"note": "y"}


class TestFeedbackQueryValidation:
    def test_rejects_non_positive_limit(self):
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=0)
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=-5)

    def test_rejects_non_integer_limit(self):
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=1.5)  # type: ignore[arg-type]

    def test_rejects_bool_limit(self):
        """True/False should never be accepted as a limit value."""
        with pytest.raises(ValueError, match="positive int"):
            FeedbackQuery(limit=True)  # type: ignore[arg-type]


async def _insert_score_row(
    loop: SQLiteFeedbackLoop,
    result_id: str,
    dimension: str,
    value: float,
    source: ScoreSource,
    created_at: str,
    metadata: dict | None = None,
) -> None:
    """Insert a scores row with an explicit created_at, bypassing score()'s
    now()-stamping so tests can control ordering/ties deterministically.
    Insertion order still drives rowid, the documented tie-breaker.
    """
    from jig._sqlite import json_dumps

    db = await loop._get_db()
    await loop._ensure_scores_metadata_column(db)
    await db.execute(
        "INSERT INTO scores (result_id, dimension, value, source, created_at, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (result_id, dimension, value, source.value, created_at,
         json_dumps(metadata) if metadata is not None else None),
    )
    await db.commit()


@pytest.mark.asyncio
class TestEffectiveScoreResolution:
    """Opt-in human-over-heuristic, newest-wins effective score resolution."""

    async def test_legacy_query_leaves_effective_scores_none(self, feedback_db):
        """Existing callers that don't opt in see no shape change."""
        rid = await feedback_db.store_result("A", "i", {})
        await feedback_db.score(rid, [Score("plausibility", 1.0, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery())
        assert len(out) == 1
        assert out[0].effective_scores is None

    async def test_human_beats_heuristic_even_when_older(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HUMAN, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC, "2025-01-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        assert len(out) == 1
        es = out[0].effective_scores["plausibility"]
        assert es.value == 1.0
        assert es.source == ScoreSource.HUMAN

    async def test_newest_heuristic_wins_when_no_human(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HEURISTIC, "2025-01-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        es = out[0].effective_scores["plausibility"]
        assert es.value == 1.0
        assert es.source == ScoreSource.HEURISTIC

    async def test_newest_human_wins_when_multiple_human_rows(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HUMAN, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HUMAN, "2025-01-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        es = out[0].effective_scores["plausibility"]
        assert es.value == 1.0

    async def test_timestamp_tie_broken_by_rowid_descending(self, feedback_db):
        """Equal created_at: the later-inserted (higher rowid) row wins."""
        rid = await feedback_db.store_result("A", "i", {})
        same_ts = "2025-01-01T00:00:00+00:00"
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC, same_ts)
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HEURISTIC, same_ts)

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        es = out[0].effective_scores["plausibility"]
        assert es.value == 1.0

    async def test_llm_judge_and_ground_truth_never_become_effective(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.LLM_JUDGE, "2025-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.GROUND_TRUTH, "2025-01-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        assert "plausibility" not in out[0].effective_scores

    async def test_missing_dimension_has_no_effective_entry(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await feedback_db.score(rid, [Score("other_dim", 0.5, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        assert "plausibility" not in out[0].effective_scores

    async def test_full_history_retained_alongside_effective_resolution(self, feedback_db):
        """Effective resolution never deletes/mutates score rows."""
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HUMAN, "2025-01-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        assert len(out[0].scores) == 2
        assert {s.source for s in out[0].scores} == {ScoreSource.HEURISTIC, ScoreSource.HUMAN}

    async def test_effective_score_metadata_round_trips(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await _insert_score_row(
            feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC,
            "2025-01-01T00:00:00+00:00",
            metadata={"rule_id": "best_sharpe_ceiling", "rule_version": 1},
        )

        out = await feedback_db.query(FeedbackQuery(resolve_effective=True))
        es = out[0].effective_scores["plausibility"]
        assert es.metadata == {"rule_id": "best_sharpe_ceiling", "rule_version": 1}


@pytest.mark.asyncio
class TestEffectiveScoreFilters:
    async def test_filter_excludes_below_min_value(self, feedback_db):
        rid_low = await feedback_db.store_result("low", "i", {})
        await feedback_db.score(rid_low, [Score("plausibility", 0.0, ScoreSource.HEURISTIC)])
        rid_high = await feedback_db.store_result("high", "i", {})
        await feedback_db.score(rid_high, [Score("plausibility", 1.0, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert [r.content for r in out] == ["high"]

    async def test_filter_is_inclusive_at_exact_boundary(self, feedback_db):
        rid = await feedback_db.store_result("exact", "i", {})
        await feedback_db.score(rid, [Score("plausibility", 0.5, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert [r.content for r in out] == ["exact"]

    async def test_filter_excludes_above_max_value(self, feedback_db):
        rid = await feedback_db.store_result("too_high", "i", {})
        await feedback_db.score(rid, [Score("sharpe_plausibility", 1.0, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="sharpe_plausibility", max_value=0.5)],
        ))
        assert out == []

    async def test_missing_effective_dimension_fails_filter(self, feedback_db):
        rid = await feedback_db.store_result("ungraded", "i", {})
        await feedback_db.score(rid, [Score("other_dim", 0.9, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert out == []

    async def test_multiple_filters_combine_with_and(self, feedback_db):
        rid_both = await feedback_db.store_result("both", "i", {})
        await feedback_db.score(rid_both, [
            Score("plausibility", 1.0, ScoreSource.HEURISTIC),
            Score("idea_quality", 1.0, ScoreSource.HEURISTIC),
        ])
        rid_one = await feedback_db.store_result("one", "i", {})
        await feedback_db.score(rid_one, [
            Score("plausibility", 1.0, ScoreSource.HEURISTIC),
            Score("idea_quality", 0.0, ScoreSource.HEURISTIC),
        ])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[
                EffectiveScoreFilter(dimension="plausibility", min_value=0.5),
                EffectiveScoreFilter(dimension="idea_quality", min_value=0.5),
            ],
        ))
        assert [r.content for r in out] == ["both"]

    async def test_human_override_reinstates_excluded_result(self, feedback_db):
        """A human plausibility row can override a disqualifying heuristic."""
        rid = await feedback_db.store_result("reinstated", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HEURISTIC, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HUMAN, "2020-06-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert [r.content for r in out] == ["reinstated"]

    async def test_human_override_can_also_exclude(self, feedback_db):
        """A human row can veto a result that heuristics alone would pass."""
        rid = await feedback_db.store_result("excluded", "i", {})
        await _insert_score_row(feedback_db, rid, "plausibility", 1.0, ScoreSource.HEURISTIC, "2020-01-01T00:00:00+00:00")
        await _insert_score_row(feedback_db, rid, "plausibility", 0.0, ScoreSource.HUMAN, "2020-06-01T00:00:00+00:00")

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert out == []

    async def test_filter_applied_before_limit_fills_full_pool(self, feedback_db):
        """A large corpus of qualifying results still fills the requested
        limit after exclusions, rather than starving on an early window."""
        for i in range(40):
            rid = await feedback_db.store_result(f"unsafe-{i}", "i", {})
            await feedback_db.score(rid, [Score("plausibility", 0.0, ScoreSource.HEURISTIC)])
        for i in range(10):
            rid = await feedback_db.store_result(f"safe-{i}", "i", {})
            await feedback_db.score(rid, [Score("plausibility", 1.0, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            limit=10,
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert len(out) == 10
        assert all(r.content.startswith("safe-") for r in out)

    async def test_effective_filters_imply_effective_scores_exposed(self, feedback_db):
        rid = await feedback_db.store_result("A", "i", {})
        await feedback_db.score(rid, [Score("plausibility", 1.0, ScoreSource.HEURISTIC)])

        out = await feedback_db.query(FeedbackQuery(
            effective_filters=[EffectiveScoreFilter(dimension="plausibility", min_value=0.5)],
        ))
        assert out[0].effective_scores["plausibility"].value == 1.0
