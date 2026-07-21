"""Tests for FeedbackLoop.get_human_examples — task-similar, human-graded
positive/negative exemplars for HumanFeedbackPromptConfig prompt injection."""
from __future__ import annotations

import numpy as np
import pytest

from jig import EffectiveScoreFilter, HumanFeedbackPromptConfig, Score, ScoreSource
from jig.core.types import HumanExampleSet
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.feedback.null import NullFeedbackLoop


def _vec(seed: int) -> np.ndarray:
    """Deterministic unit-ish vector keyed by a small integer seed.

    Two calls with the same seed return bit-identical vectors — used to
    force exact similarity ties between candidates in tie-break tests.
    """
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


@pytest.fixture
async def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "test.db"))
    # Default embed: every input maps to the same seed unless overridden by
    # a per-test lookup — see _embed_by_key below.
    loop._embed = _embed_default  # type: ignore[method-assign]
    try:
        yield loop
    finally:
        await loop.close()


async def _embed_default(text: str) -> np.ndarray:
    return _vec(abs(hash(("default", text))) % 997)


def _use_embedding_map(loop: SQLiteFeedbackLoop, mapping: dict[str, int]) -> None:
    """Route ``loop._embed`` through an explicit text -> seed lookup.

    Lets a test pin exact similarity relationships (including ties)
    between the task query and each stored result's input, which a
    content-hash-derived embedding can't guarantee.
    """
    async def _embed(text: str) -> np.ndarray:
        return _vec(mapping[text])

    loop._embed = _embed  # type: ignore[method-assign]


async def _store_human(
    loop: SQLiteFeedbackLoop,
    *,
    output: str,
    task_input: str,
    dimension: str,
    value: float,
    note: str = "operator note",
    created_at: str | None = None,
    extra_scores: list[Score] | None = None,
) -> str:
    rid = await loop.store_result(output, task_input, {})
    if created_at is None:
        await loop.score(rid, [
            Score(dimension, value, ScoreSource.HUMAN, metadata={"note": note}),
            *(extra_scores or []),
        ])
    else:
        from jig._sqlite import json_dumps

        db = await loop._get_db()
        await loop._ensure_scores_metadata_column(db)
        await db.execute(
            "INSERT INTO scores (result_id, dimension, value, source, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, dimension, value, ScoreSource.HUMAN.value, created_at,
             json_dumps({"note": note})),
        )
        await db.commit()
        if extra_scores:
            await loop.score(rid, extra_scores)
    return rid


DIMS = ("plausibility", "lookahead_safety")
CFG = HumanFeedbackPromptConfig(
    enabled=True,
    dimensions=DIMS,
    positive_threshold=0.75,
    negative_threshold=0.25,
    positive_limit=2,
    negative_limit=2,
    total_character_budget=6000,
)


@pytest.mark.asyncio
class TestGetHumanExamplesClassification:
    async def test_exact_positive_threshold_qualifies(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.75,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert len(out.positive) == 1
        assert out.positive[0].result_id
        assert out.positive[0].classification == "positive"
        assert out.positive[0].dimensions[0].value == 0.75

    async def test_exact_negative_threshold_qualifies(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.25,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert len(out.negative) == 1
        assert out.negative[0].classification == "negative"

    async def test_middle_value_is_omitted(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.5,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert out.negative == []

    async def test_just_above_negative_threshold_omitted(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.26,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert out.negative == []

    async def test_just_below_positive_threshold_omitted(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.74,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert out.negative == []

    async def test_heuristic_only_never_qualifies(self, feedback_db):
        """A heuristic score, even a perfect one, is not a human judgment."""
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await feedback_db.store_result("out-a", "in-a", {})
        await feedback_db.score(rid, [Score("plausibility", 1.0, ScoreSource.HEURISTIC)])
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert out.negative == []

    async def test_llm_judge_never_qualifies(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await feedback_db.store_result("out-a", "in-a", {})
        await feedback_db.score(rid, [Score("plausibility", 0.0, ScoreSource.LLM_JUDGE)])
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert out.negative == []

    async def test_negative_precedence_over_positive_on_same_result(self, feedback_db):
        """One dimension crosses negative, another crosses positive on the
        same result — the whole result classifies negative, and only the
        negative-crossing dimension is listed."""
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.1, note="bad plausibility",
        )
        await feedback_db.score(rid, [
            Score("lookahead_safety", 0.9, ScoreSource.HUMAN, metadata={"note": "safe"}),
        ])
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []
        assert len(out.negative) == 1
        shown = {d.dimension for d in out.negative[0].dimensions}
        assert shown == {"plausibility"}

    async def test_each_result_appears_at_most_once(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.1,
        )
        await feedback_db.score(rid, [
            Score("lookahead_safety", 0.9, ScoreSource.HUMAN, metadata={"note": "n"}),
        ])
        out = await feedback_db.get_human_examples("task", CFG)
        all_ids = [e.result_id for e in out.positive] + [e.result_id for e in out.negative]
        assert len(all_ids) == len(set(all_ids))

    async def test_note_carried_from_score_metadata(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="plausibility", value=0.9, note="great result",
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive[0].dimensions[0].note == "great result"

    async def test_input_and_output_text_are_carried(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="the output body", task_input="in-a",
            dimension="plausibility", value=0.9,
        )
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive[0].input_text == "in-a"
        assert out.positive[0].output == "the output body"


@pytest.mark.asyncio
class TestGetHumanExamplesEligibility:
    async def test_missing_eligibility_dimension_excludes_result(self, feedback_db):
        cfg = HumanFeedbackPromptConfig(
            enabled=True,
            dimensions=("idea_quality",),
            eligibility_filters=(EffectiveScoreFilter(dimension="plausibility", min_value=0.5),),
        )
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        await _store_human(
            feedback_db, output="out-a", task_input="in-a",
            dimension="idea_quality", value=0.9,
        )
        out = await feedback_db.get_human_examples("task", cfg)
        assert out.positive == []

    async def test_passing_eligibility_allows_classification(self, feedback_db):
        cfg = HumanFeedbackPromptConfig(
            enabled=True,
            dimensions=("idea_quality",),
            eligibility_filters=(EffectiveScoreFilter(dimension="plausibility", min_value=0.5),),
        )
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await feedback_db.store_result("out-a", "in-a", {})
        await feedback_db.score(rid, [
            Score("plausibility", 1.0, ScoreSource.HEURISTIC),
            Score("idea_quality", 0.9, ScoreSource.HUMAN, metadata={"note": "n"}),
        ])
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.positive) == 1

    async def test_implausible_result_excluded_even_if_negative_would_teach_a_lesson(self, feedback_db):
        """An implausible result can't enter the negative section either —
        the safe-exemplar gate is not itself bypassable via classification."""
        cfg = HumanFeedbackPromptConfig(
            enabled=True,
            dimensions=("idea_quality",),
            eligibility_filters=(EffectiveScoreFilter(dimension="plausibility", min_value=0.5),),
        )
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1})
        rid = await feedback_db.store_result("out-a", "in-a", {})
        await feedback_db.score(rid, [
            Score("plausibility", 0.0, ScoreSource.HEURISTIC),
            Score("idea_quality", 0.0, ScoreSource.HUMAN, metadata={"note": "bad"}),
        ])
        out = await feedback_db.get_human_examples("task", cfg)
        assert out.negative == []
        assert out.positive == []


@pytest.mark.asyncio
class TestGetHumanExamplesRankingAndLimits:
    async def test_missing_embedding_excludes_result(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1})
        # store_result always embeds at write time; simulate a legacy row
        # with no embedding by writing directly.
        import uuid
        from datetime import UTC, datetime

        rid = str(uuid.uuid4())
        db = await feedback_db._get_db()
        await db.execute(
            "INSERT INTO results (id, content, input, metadata, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, "out-a", "in-a", "{}", None, datetime.now(UTC).isoformat()),
        )
        await db.commit()
        await feedback_db.score(rid, [
            Score("plausibility", 0.9, ScoreSource.HUMAN, metadata={"note": "n"}),
        ])
        out = await feedback_db.get_human_examples("task", CFG)
        assert out.positive == []

    async def test_ranked_by_similarity_descending(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "near": 1, "far": 2})
        await _store_human(
            feedback_db, output="far-out", task_input="far",
            dimension="plausibility", value=0.9,
        )
        await _store_human(
            feedback_db, output="near-out", task_input="near",
            dimension="plausibility", value=0.9,
        )
        cfg = HumanFeedbackPromptConfig(
            enabled=True, dimensions=DIMS, positive_limit=1, negative_limit=1,
        )
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.positive) == 1
        assert out.positive[0].output == "near-out"

    async def test_similarity_tie_breaks_on_score_recency(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1, "in-b": 1})
        await _store_human(
            feedback_db, output="older", task_input="in-a",
            dimension="plausibility", value=0.9,
            created_at="2020-01-01T00:00:00+00:00",
        )
        await _store_human(
            feedback_db, output="newer", task_input="in-b",
            dimension="plausibility", value=0.9,
            created_at="2025-01-01T00:00:00+00:00",
        )
        cfg = HumanFeedbackPromptConfig(
            enabled=True, dimensions=DIMS, positive_limit=1, negative_limit=1,
        )
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.positive) == 1
        assert out.positive[0].output == "newer"

    async def test_similarity_and_recency_tie_breaks_on_result_id(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1, "in-a": 1, "in-b": 1})
        same_ts = "2025-01-01T00:00:00+00:00"
        rid_a = await _store_human(
            feedback_db, output="a", task_input="in-a",
            dimension="plausibility", value=0.9, created_at=same_ts,
        )
        rid_b = await _store_human(
            feedback_db, output="b", task_input="in-b",
            dimension="plausibility", value=0.9, created_at=same_ts,
        )
        expected_first = min(rid_a, rid_b)
        cfg = HumanFeedbackPromptConfig(
            enabled=True, dimensions=DIMS, positive_limit=1, negative_limit=1,
        )
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.positive) == 1
        assert out.positive[0].result_id == expected_first

    async def test_positive_limit_caps_section(self, feedback_db):
        mapping = {"task": 1}
        for i in range(5):
            mapping[f"in-{i}"] = 1
        _use_embedding_map(feedback_db, mapping)
        for i in range(5):
            await _store_human(
                feedback_db, output=f"out-{i}", task_input=f"in-{i}",
                dimension="plausibility", value=0.9,
            )
        cfg = HumanFeedbackPromptConfig(enabled=True, dimensions=DIMS, positive_limit=2)
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.positive) == 2

    async def test_negative_limit_caps_section(self, feedback_db):
        mapping = {"task": 1}
        for i in range(5):
            mapping[f"in-{i}"] = 1
        _use_embedding_map(feedback_db, mapping)
        for i in range(5):
            await _store_human(
                feedback_db, output=f"out-{i}", task_input=f"in-{i}",
                dimension="plausibility", value=0.1,
            )
        cfg = HumanFeedbackPromptConfig(enabled=True, dimensions=DIMS, negative_limit=2)
        out = await feedback_db.get_human_examples("task", cfg)
        assert len(out.negative) == 2

    async def test_no_qualified_examples_returns_empty_set(self, feedback_db):
        _use_embedding_map(feedback_db, {"task": 1})
        out = await feedback_db.get_human_examples("task", CFG)
        assert out == HumanExampleSet(positive=[], negative=[])


@pytest.mark.asyncio
async def test_null_feedback_loop_returns_empty_default():
    loop = NullFeedbackLoop()
    out = await loop.get_human_examples("task", CFG)
    assert out == HumanExampleSet(positive=[], negative=[])
