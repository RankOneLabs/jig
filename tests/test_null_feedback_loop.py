"""Tests for the no-I/O NullFeedbackLoop."""
from __future__ import annotations

import inspect

import pytest

from jig import NullFeedbackLoop
from jig.core.types import FeedbackLoop, FeedbackQuery, Score, ScoreSource
from jig.feedback.null import NullFeedbackLoop as NullFeedbackLoopDirect


@pytest.mark.asyncio
class TestNullFeedbackLoop:
    async def test_store_result_returns_a_uuid_without_touching_disk(self, tmp_path):
        loop = NullFeedbackLoop()
        result_id = await loop.store_result("content", "input")
        assert isinstance(result_id, str)
        assert len(result_id) > 0
        assert list(tmp_path.iterdir()) == []

    async def test_store_result_ids_are_unique(self):
        loop = NullFeedbackLoop()
        a = await loop.store_result("c1", "i1")
        b = await loop.store_result("c2", "i2")
        assert a != b

    async def test_score_is_a_no_op(self):
        loop = NullFeedbackLoop()
        rid = await loop.store_result("content", "input")
        # Must not raise, even with an id score() never persisted.
        await loop.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])

    async def test_get_signals_returns_empty(self):
        loop = NullFeedbackLoop()
        assert await loop.get_signals("anything") == []

    async def test_query_returns_empty(self):
        loop = NullFeedbackLoop()
        assert await loop.query(FeedbackQuery()) == []

    async def test_export_eval_set_returns_empty(self):
        loop = NullFeedbackLoop()
        assert await loop.export_eval_set() == []

    async def test_top_level_and_submodule_exports_are_the_same_class(self):
        assert NullFeedbackLoop is NullFeedbackLoopDirect


def test_store_result_metadata_annotation_matches_protocol():
    """The override's metadata param must not narrow FeedbackLoop's
    dict[str, Any] | None — dict is invariant, so a narrower type
    (e.g. dict[str, object]) breaks callers passing concretely-typed
    dicts under static analysis."""
    base_param = inspect.signature(FeedbackLoop.store_result).parameters["metadata"]
    override_param = inspect.signature(NullFeedbackLoop.store_result).parameters["metadata"]
    assert override_param.annotation == base_param.annotation
