"""Tests for SQLiteFeedbackLoop persistence contracts.

Covers:
- SQLite foreign-key enforcement (orphan scores must fail loudly)
- Score validation via validate_scores (shared helper)
- export_eval_set: filter-before-limit semantics and edge cases
- Atomic score batch: partial inserts must be rolled back on failure
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from typing import Any

import aiosqlite
import numpy as np
import pytest

from jig.core.types import Score, ScoreSource
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.feedback.validation import validate_scores


async def _fake_embed(text: str) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Foreign-key enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestForeignKeyEnforcement:
    async def test_score_unknown_result_id_raises(self, feedback_db):
        """Scoring a result_id that does not exist must raise an IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            await feedback_db.score(
                "nonexistent-id",
                [Score("quality", 0.9, ScoreSource.HEURISTIC)],
            )

    async def test_score_known_result_id_succeeds(self, feedback_db):
        rid = await feedback_db.store_result("content", "input", {})
        # Must not raise.
        await feedback_db.score(rid, [Score("quality", 0.9, ScoreSource.HEURISTIC)])

    async def test_valid_score_appears_in_query(self, feedback_db):
        """A scored result is retrievable via query()."""
        rid = await feedback_db.store_result("content", "input", {})
        await feedback_db.score(rid, [Score("quality", 0.9, ScoreSource.HEURISTIC)])
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


# ---------------------------------------------------------------------------
# Atomic score batch: partial inserts must be rolled back on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScoreBatchAtomicity:
    """score() must be all-or-nothing: a failure on any dimension insert
    must roll back every insert already executed in that call.

    Pre-fix: inserts ran without an explicit transaction boundary.  A failure
    after the Nth insert left N-1 inserts in an open implicit transaction that
    was visible on the same connection and would be committed by the next
    unrelated commit — partial batch corruption.

    Post-fix: explicit BEGIN → rollback on exception ensures zero rows persist.
    """

    async def test_partial_insert_failure_rolls_back_entire_batch(
        self, feedback_db: SQLiteFeedbackLoop
    ) -> None:
        rid = await feedback_db.store_result("content", "input", {})
        db = await feedback_db._get_db()

        insert_count = 0
        original_execute = db.execute

        async def failing_execute(sql: str, params: Any = ()) -> Any:
            nonlocal insert_count
            if "INSERT INTO scores" in sql:
                insert_count += 1
                if insert_count == 2:
                    raise aiosqlite.Error("simulated second-insert failure")
            return await original_execute(sql, params)

        db.execute = failing_execute  # type: ignore[method-assign]
        try:
            with pytest.raises(aiosqlite.Error):
                await feedback_db.score(
                    rid,
                    [
                        Score("dim1", 0.8, ScoreSource.HEURISTIC),
                        Score("dim2", 0.6, ScoreSource.HEURISTIC),
                        Score("dim3", 0.4, ScoreSource.HEURISTIC),
                    ],
                )
        finally:
            db.execute = original_execute  # type: ignore[method-assign]

        # No scores must survive — the rollback must have undone insert #1.
        cursor = await db.execute(
            "SELECT COUNT(*) FROM scores WHERE result_id = ?", (rid,)
        )
        row = await cursor.fetchone()
        assert row[0] == 0, (
            f"expected 0 score rows after rollback, got {row[0]}; "
            "partial batch was not rolled back"
        )

    async def test_successful_batch_persists_all_dimensions(
        self, feedback_db: SQLiteFeedbackLoop
    ) -> None:
        """Sanity-check: a clean batch with 3 dimensions commits all 3."""
        rid = await feedback_db.store_result("content", "input", {})
        await feedback_db.score(
            rid,
            [
                Score("accuracy", 0.9, ScoreSource.HEURISTIC),
                Score("relevance", 0.8, ScoreSource.HEURISTIC),
                Score("fluency", 0.7, ScoreSource.HEURISTIC),
            ],
        )
        db = await feedback_db._get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM scores WHERE result_id = ?", (rid,)
        )
        row = await cursor.fetchone()
        assert row[0] == 3, "all three dimensions must be persisted on success"

    async def test_feedback_writes_are_serialized_on_shared_connection(
        self, feedback_db: SQLiteFeedbackLoop
    ) -> None:
        """A concurrent feedback write must not enter during a score transaction."""
        rid = await feedback_db.store_result("content", "input", {})
        db = await feedback_db._get_db()

        score_insert_started = asyncio.Event()
        allow_score_insert = asyncio.Event()
        result_insert_started = asyncio.Event()
        original_execute = db.execute

        async def blocking_execute(sql: str, params: Any = ()) -> Any:
            if "INSERT INTO scores" in sql:
                score_insert_started.set()
                await allow_score_insert.wait()
            if "INSERT INTO results" in sql:
                result_insert_started.set()
            return await original_execute(sql, params)

        db.execute = blocking_execute  # type: ignore[method-assign]
        score_task = asyncio.create_task(
            feedback_db.score(rid, [Score("q", 0.8, ScoreSource.HEURISTIC)])
        )
        store_task: asyncio.Task[str] | None = None
        try:
            await asyncio.wait_for(score_insert_started.wait(), timeout=1)
            store_task = asyncio.create_task(
                feedback_db.store_result("other", "input", {})
            )
            await asyncio.sleep(0)
            assert not result_insert_started.is_set(), (
                "store_result entered the shared connection during score transaction"
            )
            allow_score_insert.set()
            await score_task
            await store_task
            assert result_insert_started.is_set()
        finally:
            allow_score_insert.set()
            db.execute = original_execute  # type: ignore[method-assign]
            pending = [
                task for task in (score_task, store_task)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)


class TestFeedbackWriteLockLifecycle:
    def test_write_lock_is_created_lazily(self, tmp_path) -> None:
        loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "lazy-lock.db"))
        assert loop._db_lock is None


# ---------------------------------------------------------------------------
# Score metadata: legacy-schema migration + serialization contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScoreMetadataMigration:
    async def _make_legacy_db(self, path: str) -> None:
        """Build a pre-metadata-column database, matching a real old deploy."""
        conn = await aiosqlite.connect(path)
        try:
            await conn.executescript(
                """
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    input TEXT NOT NULL,
                    metadata JSON,
                    embedding BLOB,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE scores (
                    result_id TEXT NOT NULL REFERENCES results(id),
                    dimension TEXT NOT NULL,
                    value REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX idx_scores_result ON scores(result_id);
                """
            )
            await conn.commit()
        finally:
            await conn.close()

    async def test_opening_legacy_db_migrates_scores_metadata_column(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")
        await self._make_legacy_db(db_path)

        loop = SQLiteFeedbackLoop(db_path=db_path)
        loop._embed = _fake_embed  # type: ignore[method-assign]
        try:
            rid = await loop.store_result("content", "input", {})
            await loop.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC, metadata={"k": "v"})])

            db = await loop._get_db()
            cursor = await db.execute("PRAGMA table_info(scores)")
            columns = {row[1] for row in await cursor.fetchall()}
            assert "metadata" in columns

            row = await (
                await db.execute(
                    "SELECT metadata FROM scores WHERE result_id = ?", (rid,)
                )
            ).fetchone()
            assert row[0] == '{"k": "v"}'
        finally:
            await loop.close()

    async def test_legacy_rows_stay_null_after_migration(self, tmp_path):
        """Historical rows inserted before the migration keep metadata=NULL."""
        db_path = str(tmp_path / "legacy2.db")
        await self._make_legacy_db(db_path)

        conn = await aiosqlite.connect(db_path)
        try:
            await conn.execute(
                "INSERT INTO results (id, content, input, metadata, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("legacy-1", "c", "i", "{}", b"", datetime.now().isoformat()),
            )
            await conn.execute(
                "INSERT INTO scores (result_id, dimension, value, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("legacy-1", "q", 0.5, "heuristic", datetime.now().isoformat()),
            )
            await conn.commit()
        finally:
            await conn.close()

        loop = SQLiteFeedbackLoop(db_path=db_path)
        loop._embed = _fake_embed  # type: ignore[method-assign]
        try:
            db = await loop._get_db()
            await loop._ensure_scores_metadata_column(db)
            row = await (
                await db.execute(
                    "SELECT metadata FROM scores WHERE result_id = ?", ("legacy-1",)
                )
            ).fetchone()
            assert row[0] is None
        finally:
            await loop.close()

    async def test_none_metadata_stored_as_null_not_empty_object(self, feedback_db):
        rid = await feedback_db.store_result("c", "i", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])
        db = await feedback_db._get_db()
        row = await (
            await db.execute(
                "SELECT metadata FROM scores WHERE result_id = ?", (rid,)
            )
        ).fetchone()
        assert row[0] is None

    async def test_non_serializable_metadata_rejected_without_inserting(self, feedback_db):
        rid = await feedback_db.store_result("c", "i", {})
        bad_score = Score("q", 0.9, ScoreSource.HEURISTIC, metadata={"bad": object()})
        with pytest.raises(ValueError, match="JSON-serializable"):
            await feedback_db.score(rid, [bad_score])

        db = await feedback_db._get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM scores WHERE result_id = ?", (rid,)
        )
        row = await cursor.fetchone()
        assert row[0] == 0

    async def test_non_serializable_metadata_in_batch_inserts_nothing(self, feedback_db):
        """One bad score in a multi-score batch must reject the entire call."""
        rid = await feedback_db.store_result("c", "i", {})
        scores = [
            Score("good", 0.5, ScoreSource.HEURISTIC),
            Score("bad", 0.9, ScoreSource.HEURISTIC, metadata={"x": object()}),
        ]
        with pytest.raises(ValueError, match="JSON-serializable"):
            await feedback_db.score(rid, scores)

        db = await feedback_db._get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM scores WHERE result_id = ?", (rid,)
        )
        row = await cursor.fetchone()
        assert row[0] == 0
