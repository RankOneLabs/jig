"""End-to-end SQLite-backed lifecycle integration tests.

Proves the full eval and feedback loop: store_result → score → query →
get_signals → export_eval_set, for both the agent and pipeline paths.

Regression guard for: orphan FK enforcement, NaN rejection, and
filter-before-limit in export_eval_set.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from jig import AgentConfig, CompletionParams, LLMResponse, PipelineConfig, Score, ScoreSource, Step, Usage, run_agent, run_pipeline
from jig.core.types import (
    FeedbackQuery,
    Grader,
    LLMClient,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Deterministic fake embed (no Ollama required)
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
    def __init__(self, dimension: str, value: float) -> None:
        self._score = Score(dimension=dimension, value=value, source=ScoreSource.HEURISTIC)

    async def grade(self, input: Any, output: Any, context: dict[str, Any] | None = None) -> list[Score]:
        return [self._score]


class StubTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None, kind: SpanKind = SpanKind.AGENT_RUN) -> Span:
        s = Span(id="trace-0", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None, metadata: dict[str, Any] | None = None) -> Span:
        s = Span(
            id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name,
            started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata,
        )
        self.spans.append(s)
        return s

    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Any = None) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.output = output
                s.error = error
                s.usage = usage

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(self, since: datetime | None = None, limit: int = 50, name: str | None = None) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]

    async def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def feedback_db(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "lifecycle.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    try:
        yield loop
    finally:
        await loop.close()


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_lifecycle_end_to_end(feedback_db):
    """run_agent → result stored → query visible → get_signals visible → export_eval_set visible."""
    config = AgentConfig(
        name="e2e-agent",
        description="lifecycle test",
        system_prompt="You are a test agent.",
        llm=FixedLLM("final answer"),
        feedback=feedback_db,
        tracer=StubTracer(),
        tools=ToolRegistry(),
        grader=FixedGrader("quality", 0.85),
    )
    result = await run_agent(config, "user question")

    assert result.scores is not None
    assert result.scores[0].value == pytest.approx(0.85)

    # query() sees the result with agent_name metadata
    query_results = await feedback_db.query(FeedbackQuery(agent_name="e2e-agent", limit=5))
    assert len(query_results) == 1
    assert query_results[0].content == "final answer"
    assert query_results[0].scores[0].value == pytest.approx(0.85)
    assert query_results[0].metadata["kind"] == "agent_result"
    assert query_results[0].metadata["source"] == "run_agent"

    # get_signals() returns the stored result
    signals = await feedback_db.get_signals("user question", limit=5)
    assert any(s.content == "final answer" for s in signals)

    # export_eval_set() exports it as an EvalCase
    cases = await feedback_db.export_eval_set()
    assert any(c.expected == "final answer" and c.input == "user question" for c in cases)


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


async def _increment(ctx: dict[str, Any]) -> int:
    return ctx["input"] + 1


@pytest.mark.asyncio
async def test_pipeline_lifecycle_end_to_end(feedback_db):
    """run_pipeline step grading → result stored with step metadata → query → export."""
    tracer = StubTracer()
    result = await run_pipeline(
        PipelineConfig(
            name="e2e-pipeline",
            steps=[Step(name="increment", fn=_increment, grader=FixedGrader("accuracy", 0.75))],
            tracer=tracer,
            feedback=feedback_db,
        ),
        input=9,
    )

    assert result.step_scores["increment"][0].value == pytest.approx(0.75)

    # query() returns the step result with pipeline metadata
    query_results = await feedback_db.query(FeedbackQuery(limit=5))
    assert len(query_results) == 1
    meta = query_results[0].metadata
    assert meta["kind"] == "pipeline_step_result"
    assert meta["pipeline_name"] == "e2e-pipeline"
    assert meta["step_name"] == "increment"
    assert meta["trace_id"] == result.trace_id

    # export_eval_set() surfaces the graded step
    cases = await feedback_db.export_eval_set()
    assert len(cases) == 1
    assert cases[0].metadata["avg_score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Regression: orphan FK enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphan_score_raises_integrity_error(feedback_db):
    """score() with an unknown result_id must fail — FK enforcement is on."""
    with pytest.raises(sqlite3.IntegrityError):
        await feedback_db.score(
            "nonexistent-id",
            [Score(dimension="quality", value=0.9, source=ScoreSource.HEURISTIC)],
        )


# ---------------------------------------------------------------------------
# Regression: NaN rejected before DB write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nan_score_rejected(feedback_db):
    """NaN scores raise ValueError via validate_scores before any DB write."""
    rid = await feedback_db.store_result("content", "input", {})
    with pytest.raises(ValueError, match="NaN"):
        await feedback_db.score(rid, [Score(dimension="q", value=float("nan"), source=ScoreSource.HEURISTIC)])


# ---------------------------------------------------------------------------
# Regression: export_eval_set filter-before-limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_eval_set_filter_before_limit(feedback_db):
    """min_score filter is applied before limit; limit=2 returns 2 qualifying rows."""
    for i in range(3):
        rid = await feedback_db.store_result(f"high-{i}", "input", {})
        await feedback_db.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])
    for i in range(2):
        rid = await feedback_db.store_result(f"low-{i}", "input", {})
        await feedback_db.score(rid, [Score("q", 0.1, ScoreSource.HEURISTIC)])

    cases = await feedback_db.export_eval_set(min_score=0.5, limit=2)
    assert len(cases) == 2
    assert all("high" in c.expected for c in cases)
