"""Real SQLite feedback lifecycle tests for graded pipeline execution.

Proves that store_result is called before score, the returned ID is what
score receives, and scores are queryable through the feedback result join
path. Uses SQLiteFeedbackLoop with a deterministic fake embedder so no
Ollama process is required.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from jig import FeedbackQuery, PipelineConfig, Score, ScoreSource, SpanKind, Step, run_pipeline
from jig.core.types import Grader, Span, TracingLogger
from jig.feedback.loop import SQLiteFeedbackLoop


# --- Minimal fake tracer ---


class FakeTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"span-{self._counter}"

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None, kind: SpanKind = SpanKind.AGENT_RUN) -> Span:
        sid = self._next_id()
        s = Span(id=sid, trace_id=f"trace-{sid}", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None, metadata: dict[str, Any] | None = None) -> Span:
        parent = next((s for s in self.spans if s.id == parent_id), None)
        sid = self._next_id()
        s = Span(
            id=sid,
            trace_id=parent.trace_id if parent else parent_id,
            kind=kind, name=name,
            started_at=datetime.now(),
            parent_id=parent_id,
            input=input, metadata=metadata,
        )
        self.spans.append(s)
        return s

    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Any = None) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.ended_at = datetime.now()
                s.output = output
                s.error = error

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(self, since: datetime | None = None, limit: int = 50, name: str | None = None) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]


# --- Deterministic fake embedder (no Ollama) ---


async def _fake_embed(text: str) -> np.ndarray:
    import hashlib
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


# --- Simple graders and step functions ---


class FixedGrader(Grader):
    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    async def grade(self, input: Any, output: Any, context: dict[str, Any] | None = None) -> list[Score]:
        return [Score(dimension="quality", value=self._value, source=ScoreSource.HEURISTIC)]


async def add_one(ctx: dict[str, Any]) -> int:
    return ctx["input"] + 1


# --- Fixture ---


@pytest.fixture
def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "lifecycle.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    return loop


# --- Tests ---


@pytest.mark.asyncio
class TestPipelineStepFeedbackLifecycle:
    async def test_store_result_called_before_score(self, feedback_db):
        """store_result must precede score; score must use the returned ID."""
        call_order: list[tuple[str, str]] = []
        original_store = feedback_db.store_result
        original_score = feedback_db.score

        async def tracked_store(content, input_text, metadata=None):
            rid = await original_store(content, input_text, metadata)
            call_order.append(("store", rid))
            return rid

        async def tracked_score(result_id, scores):
            call_order.append(("score", result_id))
            await original_score(result_id, scores)

        feedback_db.store_result = tracked_store
        feedback_db.score = tracked_score

        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="lifecycle_test",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.8))],
                tracer=tracer,
                feedback=feedback_db,
            ),
            input=5,
        )

        assert len(call_order) == 2
        assert call_order[0][0] == "store"
        assert call_order[1][0] == "score"
        # score received the ID that store_result returned
        assert call_order[1][1] == call_order[0][1]

    async def test_step_metadata_kind_pipeline_name_step_name_trace_id(self, feedback_db):
        """store_result metadata includes kind, pipeline_name, step_name, and trace_id."""
        captured_meta: list[dict] = []
        original_store = feedback_db.store_result

        async def capturing_store(content, input_text, metadata=None):
            captured_meta.append(metadata or {})
            return await original_store(content, input_text, metadata)

        feedback_db.store_result = capturing_store

        tracer = FakeTracer()
        result = await run_pipeline(
            PipelineConfig(
                name="meta_test",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.9))],
                tracer=tracer,
                feedback=feedback_db,
            ),
            input=5,
        )

        assert len(captured_meta) == 1
        meta = captured_meta[0]
        assert meta["kind"] == "pipeline_step_result"
        assert meta["pipeline_name"] == "meta_test"
        assert meta["step_name"] == "add_one"
        assert meta["trace_id"] == result.trace_id

    async def test_scores_queryable_via_join_path(self, feedback_db):
        """Scores stored by step grading are retrievable through query()."""
        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="query_test",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.75))],
                tracer=tracer,
                feedback=feedback_db,
            ),
            input=5,
        )

        results = await feedback_db.query(FeedbackQuery(limit=10))
        assert len(results) == 1
        assert results[0].scores[0].value == pytest.approx(0.75)

    async def test_pipeline_grading_uses_kind_pipeline_result(self, feedback_db):
        """Pipeline-level grading registers a result with kind=pipeline_result."""
        captured_meta: list[dict] = []
        original_store = feedback_db.store_result

        async def capturing_store(content, input_text, metadata=None):
            captured_meta.append(metadata or {})
            return await original_store(content, input_text, metadata)

        feedback_db.store_result = capturing_store

        tracer = FakeTracer()
        result = await run_pipeline(
            PipelineConfig(
                name="pipeline_grade_test",
                steps=[Step(name="add_one", fn=add_one)],
                tracer=tracer,
                grader=FixedGrader(0.85),
                feedback=feedback_db,
            ),
            input=5,
        )

        assert len(captured_meta) == 1
        meta = captured_meta[0]
        assert meta["kind"] == "pipeline_result"
        assert meta["pipeline_name"] == "pipeline_grade_test"
        assert meta["trace_id"] == result.trace_id
        assert "step_name" not in meta

    async def test_step_and_pipeline_grading_register_separate_results(self, feedback_db):
        """Step and pipeline graders each register an independent feedback result."""
        captured_meta: list[dict] = []
        original_store = feedback_db.store_result

        async def capturing_store(content, input_text, metadata=None):
            captured_meta.append(metadata or {})
            return await original_store(content, input_text, metadata)

        feedback_db.store_result = capturing_store

        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="dual_grade",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.7))],
                tracer=tracer,
                grader=FixedGrader(0.9),
                feedback=feedback_db,
            ),
            input=5,
        )

        assert len(captured_meta) == 2
        kinds = {m["kind"] for m in captured_meta}
        assert kinds == {"pipeline_step_result", "pipeline_result"}

        results = await feedback_db.query(FeedbackQuery(limit=10))
        assert len(results) == 2

    async def test_grading_span_carries_feedback_result_id_and_canonical_scores(self, feedback_db):
        """The grading span output includes feedback_result_id and scores in canonical shape."""
        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="span_shape_test",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.6))],
                tracer=tracer,
                feedback=feedback_db,
            ),
            input=5,
        )

        grading_spans = [s for s in tracer.spans if s.kind == SpanKind.GRADING]
        assert len(grading_spans) == 1
        out = grading_spans[0].output
        assert isinstance(out, dict)
        assert "feedback_result_id" in out
        assert isinstance(out["scores"], list)
        assert out["scores"][0]["dimension"] == "quality"
        assert out["scores"][0]["value"] == pytest.approx(0.6)

    async def test_config_metadata_source_tags_model_pass_through(self, feedback_db):
        """source, tags, and model from PipelineConfig.metadata appear in feedback metadata."""
        captured_meta: list[dict] = []
        original_store = feedback_db.store_result

        async def capturing_store(content, input_text, metadata=None):
            captured_meta.append(metadata or {})
            return await original_store(content, input_text, metadata)

        feedback_db.store_result = capturing_store

        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="meta_passthrough",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.5))],
                tracer=tracer,
                feedback=feedback_db,
                metadata={"source": "backtest", "tags": ["mean_reversion"], "model": "claude-sonnet-4-5"},
            ),
            input=5,
        )

        assert len(captured_meta) == 1
        meta = captured_meta[0]
        assert meta["source"] == "backtest"
        assert meta["tags"] == ["mean_reversion"]
        assert meta["model"] == "claude-sonnet-4-5"

    async def test_custom_feedback_serializer_used_at_storage_boundary(self, feedback_db):
        """PipelineConfig.feedback_serializer takes precedence over the default policy."""
        serialized: list[str] = []

        def my_serializer(value: Any) -> str:
            s = f"CUSTOM:{value!r}"
            serialized.append(s)
            return s

        tracer = FakeTracer()
        await run_pipeline(
            PipelineConfig(
                name="serializer_test",
                steps=[Step(name="add_one", fn=add_one, grader=FixedGrader(0.5))],
                tracer=tracer,
                feedback=feedback_db,
                feedback_serializer=my_serializer,
            ),
            input=5,
        )

        assert len(serialized) >= 2  # content and input_text both go through serializer
        assert all("CUSTOM:" in v for v in serialized)

    async def test_default_serializer_handles_str_int_dict(self, feedback_db):
        """Default serialization: strings pass through, ints and dicts become JSON text."""
        from jig.core.pipeline import _serialize_for_feedback

        assert _serialize_for_feedback("hello") == "hello"
        assert _serialize_for_feedback(42) == "42"
        assert _serialize_for_feedback({"a": 1}) == '{"a": 1}'

    async def test_default_serializer_falls_back_to_repr_for_unserializable(self, feedback_db):
        """Non-JSON-serializable objects fall back to repr."""
        from jig.core.pipeline import _serialize_for_feedback

        class Opaque:
            def __repr__(self) -> str:
                return "Opaque()"

        result = _serialize_for_feedback(Opaque())
        assert result == "Opaque()"
