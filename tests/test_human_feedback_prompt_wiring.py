"""Tests that run_agent wires HumanFeedbackPromptConfig end-to-end: when
enabled, it queries feedback.get_human_examples and appends the rendered
section to the system message; when disabled (the default), it does neither."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pytest

from jig import AgentConfig, CompletionParams, HumanFeedbackPromptConfig, LLMResponse, Score, ScoreSource, Usage, run_agent
from jig.core.types import LLMClient, Span, SpanKind, TracingLogger
from jig.feedback.loop import SQLiteFeedbackLoop
from jig.tools import ToolRegistry


async def _fake_embed(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32)


class CapturingLLM(LLMClient):
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_system: str | None = None

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.last_system = params.system
        return LLMResponse(
            content=self._content, tool_calls=None,
            usage=Usage(1, 1, cost=0.0), latency_ms=1, model="fixed",
        )


class StubTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN) -> Span:
        s = Span(id="trace-0", trace_id="t-0", kind=kind, name=name,
                  started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id, kind, name, input=None, metadata=None) -> Span:
        s = Span(id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name,
                  started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata)
        self.spans.append(s)
        return s

    def end_span(self, span_id, output=None, error=None, usage=None) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.output, s.error, s.usage = output, error, usage

    async def get_trace(self, trace_id):
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(self, since=None, limit=50, name=None):
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]

    async def flush(self) -> None:
        pass


@pytest.fixture
async def feedback_loop(tmp_path):
    loop = SQLiteFeedbackLoop(db_path=str(tmp_path / "feedback.db"))
    loop._embed = _fake_embed  # type: ignore[method-assign]
    try:
        yield loop
    finally:
        await loop.close()


def _make_config(feedback_loop, llm, human_feedback_prompt=None) -> AgentConfig:
    kwargs: dict[str, Any] = dict(
        name="wiring-agent",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        feedback=feedback_loop,
        tracer=StubTracer(),
        tools=ToolRegistry(),
        include_feedback_in_prompt=False,
    )
    if human_feedback_prompt is not None:
        kwargs["human_feedback_prompt"] = human_feedback_prompt
    return AgentConfig(**kwargs)


@pytest.mark.asyncio
class TestHumanFeedbackPromptWiring:
    async def test_disabled_by_default_omits_section_and_skips_query(self, feedback_loop):
        rid = await feedback_loop.store_result("qualifying output", "similar task", {})
        await feedback_loop.score(rid, [
            Score("plausibility", 0.9, ScoreSource.HUMAN, metadata={"note": "great"}),
        ])
        llm = CapturingLLM("done")
        config = _make_config(feedback_loop, llm)
        assert config.human_feedback_prompt.enabled is False

        await run_agent(config, "similar task")
        assert "Human-reviewed" not in llm.last_system

    async def test_enabled_injects_positive_section(self, feedback_loop):
        rid = await feedback_loop.store_result("qualifying output", "similar task", {})
        await feedback_loop.score(rid, [
            Score("plausibility", 0.9, ScoreSource.HUMAN, metadata={"note": "great result"}),
        ])
        llm = CapturingLLM("done")
        cfg = HumanFeedbackPromptConfig(enabled=True, dimensions=("plausibility",))
        config = _make_config(feedback_loop, llm, human_feedback_prompt=cfg)

        await run_agent(config, "similar task")
        assert "Human-reviewed positive examples" in llm.last_system
        assert "qualifying output" in llm.last_system
        assert "great result" in llm.last_system

    async def test_enabled_but_no_qualifying_examples_omits_section(self, feedback_loop):
        llm = CapturingLLM("done")
        cfg = HumanFeedbackPromptConfig(enabled=True, dimensions=("plausibility",))
        config = _make_config(feedback_loop, llm, human_feedback_prompt=cfg)

        await run_agent(config, "similar task")
        assert "Human-reviewed" not in llm.last_system

    async def test_legacy_and_human_paths_are_independent(self, feedback_loop):
        """Both can be enabled simultaneously; each renders its own section."""
        rid = await feedback_loop.store_result("legacy output", "similar task", {})
        await feedback_loop.score(rid, [Score("q", 0.9, ScoreSource.HEURISTIC)])
        rid2 = await feedback_loop.store_result("human output", "similar task", {})
        await feedback_loop.score(rid2, [
            Score("plausibility", 0.9, ScoreSource.HUMAN, metadata={"note": "n"}),
        ])
        llm = CapturingLLM("done")
        cfg = HumanFeedbackPromptConfig(enabled=True, dimensions=("plausibility",))
        config = AgentConfig(
            name="both-agent", description="test", system_prompt="sys",
            llm=llm, feedback=feedback_loop, tracer=StubTracer(), tools=ToolRegistry(),
            include_feedback_in_prompt=True, human_feedback_prompt=cfg,
        )
        await run_agent(config, "similar task")
        assert "Quality signals from past similar queries" in llm.last_system
        assert "Human-reviewed positive examples" in llm.last_system
