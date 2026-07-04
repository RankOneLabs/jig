"""SQLite integration tests for the run_agent feedback lifecycle.

Proves that after run_agent completes with a grader:
  1. A feedback result row is registered via store_result.
  2. Score rows are attached to that feedback_result_id.
  3. query() returns the result with metadata filters.
  4. get_signals() returns the result.
  5. export_eval_set() returns the result as an EvalCase.
  6. The grade span carries feedback_result_id and canonical scores shape.
  7. Memory store IDs and feedback result IDs are distinct namespaces.

No Ollama or network calls — _embed is replaced with a deterministic
hash-based implementation for all tests.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from jig import AgentConfig, CompletionParams, LLMResponse, Score, ScoreSource, Usage, run_agent
from jig.core.types import (
    EvalCase,
    FeedbackQuery,
    Grader,
    LLMClient,
    MemoryEntry,
    MemoryStore,
    Message,
    Role,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Deterministic embed — no Ollama required
# ---------------------------------------------------------------------------

async def _fake_embed(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class FixedLLM(LLMClient):
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, params: CompletionParams) -> LLMResponse:
        return LLMResponse(
            content=self._content,
            tool_calls=None,
            usage=Usage(10, 5, cost=0.0),
            latency_ms=1,
            model="fixed",
        )


class FixedGrader(Grader):
    """Deterministic grader — no LLM, no network."""

    def __init__(self, dimension: str, value: float, source: ScoreSource) -> None:
        self._score = Score(dimension=dimension, value=value, source=source)

    async def grade(
        self, input: Any, output: Any, context: dict[str, Any] | None = None
    ) -> list[Score]:
        return [self._score]


class StubTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        s = Span(
            id="trace-0", trace_id="t-0", kind=kind, name=name,
            started_at=datetime.now(), metadata=metadata,
        )
        self.spans.append(s)
        return s

    def start_span(
        self,
        parent_id: str,
        kind: SpanKind,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        s = Span(
            id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name,
            started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata,
        )
        self.spans.append(s)
        return s

    def end_span(
        self, span_id: str, output: Any = None, error: str | None = None, usage: Any = None
    ) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.output = output
                s.error = error
                s.usage = usage

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(
        self, since: datetime | None = None, limit: int = 50, name: str | None = None
    ) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]

    async def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def feedback_loop(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "feedback.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    try:
        yield loop
    finally:
        await loop.close()


def _make_config(name: str, content: str, feedback_loop, tracer, grader, store=None):
    llm = FixedLLM(content)
    return AgentConfig(
        name=name,
        description="integration test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        store=store,
        feedback=feedback_loop,
        tracer=tracer,
        tools=ToolRegistry(),
        grader=grader,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentFeedbackLifecycle:
    async def test_result_registered_and_scored(self, feedback_loop):
        """run_agent with grader registers a result and attaches scores."""
        result = await run_agent(
            _make_config(
                "lifecycle-agent", "my agent output", feedback_loop,
                StubTracer(), FixedGrader("quality", 0.9, ScoreSource.HEURISTIC),
            ),
            "test input",
        )
        assert result.scores is not None
        assert result.scores[0].dimension == "quality"
        assert result.scores[0].value == pytest.approx(0.9)

        results = await feedback_loop.query(FeedbackQuery(limit=10))
        assert len(results) == 1
        assert results[0].content == "my agent output"
        assert results[0].scores[0].value == pytest.approx(0.9)

    async def test_metadata_agent_name_is_queryable(self, feedback_loop):
        """agent_name metadata stored by run_agent enables FeedbackQuery filtering."""
        await run_agent(
            _make_config(
                "filter-agent", "output A", feedback_loop,
                StubTracer(), FixedGrader("q", 0.8, ScoreSource.HEURISTIC),
            ),
            "input A",
        )
        by_agent = await feedback_loop.query(FeedbackQuery(agent_name="filter-agent"))
        assert len(by_agent) == 1
        assert by_agent[0].metadata["agent_name"] == "filter-agent"

        no_match = await feedback_loop.query(FeedbackQuery(agent_name="other-agent"))
        assert no_match == []

    async def test_metadata_model_is_queryable(self, feedback_loop):
        """model metadata stored by run_agent enables FeedbackQuery filtering."""
        llm = FixedLLM("output B")
        llm._model = "test-model-x"
        config = AgentConfig(
            name="model-agent",
            description="test",
            system_prompt="s",
            llm=llm,
            feedback=feedback_loop,
            tracer=StubTracer(),
            tools=ToolRegistry(),
            grader=FixedGrader("q", 0.7, ScoreSource.HEURISTIC),
        )
        await run_agent(config, "model input")

        by_model = await feedback_loop.query(FeedbackQuery(model="test-model-x"))
        assert len(by_model) == 1
        assert by_model[0].metadata["model"] == "test-model-x"

    async def test_get_signals_returns_scored_result(self, feedback_loop):
        """get_signals returns the agent output after run_agent completes."""
        await run_agent(
            _make_config(
                "signals-agent", "signals output", feedback_loop,
                StubTracer(), FixedGrader("q", 0.85, ScoreSource.HEURISTIC),
            ),
            "signals query",
        )
        signals = await feedback_loop.get_signals("signals query", limit=5)
        assert len(signals) >= 1
        assert any(s.content == "signals output" for s in signals)

    async def test_export_eval_set_includes_result(self, feedback_loop):
        """export_eval_set returns an EvalCase for the agent output."""
        await run_agent(
            _make_config(
                "export-agent", "export output", feedback_loop,
                StubTracer(), FixedGrader("q", 0.95, ScoreSource.HEURISTIC),
            ),
            "export input",
        )
        cases = await feedback_loop.export_eval_set()
        assert len(cases) >= 1
        case = next((c for c in cases if c.expected == "export output"), None)
        assert case is not None
        assert case.input == "export input"

    async def test_grade_span_has_feedback_result_id_and_canonical_scores(self, feedback_loop):
        """Grade span output includes feedback_result_id and scores=[{dimension,value}]."""
        tracer = StubTracer()
        await run_agent(
            _make_config(
                "span-agent", "span output", feedback_loop,
                tracer, FixedGrader("accuracy", 0.75, ScoreSource.HEURISTIC),
            ),
            "span input",
        )
        grade_spans = [s for s in tracer.spans if s.kind == SpanKind.GRADING]
        assert len(grade_spans) == 1
        output = grade_spans[0].output
        assert isinstance(output, dict)
        assert "feedback_result_id" in output
        assert isinstance(output["feedback_result_id"], str)
        assert len(output["feedback_result_id"]) > 0
        assert output["scores"] == [{"dimension": "accuracy", "value": pytest.approx(0.75)}]

    async def test_memory_id_is_separate_from_feedback_result_id(self, feedback_loop):
        """Memory store IDs and feedback result IDs are different identifiers."""
        memory_ids: list[str] = []

        class TrackingMemory(MemoryStore):
            async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
                memory_ids.append(f"mem-{len(memory_ids) + 1}")
                return memory_ids[-1]

            async def get(self, id: str) -> MemoryEntry | None:
                return None

            async def all(self) -> list[MemoryEntry]:
                return []

            async def delete(self, id: str) -> None:
                pass

            async def get_session(self, session_id: str) -> list[Message]:
                return []

            async def add_to_session(self, session_id: str, message: Message) -> None:
                pass

            async def clear(
                self, session_id: str | None = None, before: datetime | None = None
            ) -> None:
                pass

        config = AgentConfig(
            name="mem-agent",
            description="test",
            system_prompt="s",
            llm=FixedLLM("mem output"),
            store=TrackingMemory(),
            feedback=feedback_loop,
            tracer=StubTracer(),
            tools=ToolRegistry(),
            grader=FixedGrader("q", 0.8, ScoreSource.HEURISTIC),
        )
        await run_agent(config, "mem input")

        results = await feedback_loop.query(FeedbackQuery(limit=10))
        assert len(results) == 1
        # The feedback result ID must not be the memory store ID
        assert results[0].result_id not in memory_ids
        # The memory ID is surfaced in metadata as informational context
        assert results[0].metadata.get("memory_id") == memory_ids[0]

    async def test_result_without_grader_is_not_registered(self, feedback_loop):
        """run_agent without a grader does not register any feedback result."""
        await run_agent(
            AgentConfig(
                name="no-grader",
                description="test",
                system_prompt="s",
                llm=FixedLLM("output"),
                feedback=feedback_loop,
                tracer=StubTracer(),
                tools=ToolRegistry(),
            ),
            "input",
        )
        results = await feedback_loop.query(FeedbackQuery(limit=10))
        assert results == []
