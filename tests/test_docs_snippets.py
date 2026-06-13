"""Drift guard: AgentConfig quick-start examples from README.md and jig-usage-guide.md.

These tests construct the two documented shapes so any stale field name (e.g.
the removed `memory=` kwarg) fails CI at construction with TypeError rather
than silently doing the wrong thing at runtime.

Sections mirrored:
- README.md "Quick start" (no-memory and local-memory examples)
- jig-usage-guide.md "Quick start — agent" (same two shapes)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from jig import AgentConfig, LLMResponse
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    LLMClient,
    MemoryEntry,
    MemoryStore,
    Retriever,
    Score,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


class _FakeLLM(LLMClient):
    async def complete(self, params: CompletionParams) -> LLMResponse:
        raise NotImplementedError


class _FakeMemory(MemoryStore, Retriever):
    async def add(self, content: Any, metadata: Any = None) -> str:
        return "m"

    async def get(self, id: str) -> Any:
        return None

    async def all(self) -> list:
        return []

    async def delete(self, id: str) -> None:
        pass

    async def retrieve(self, query: str, k: int = 5, context: Any = None) -> list:
        return []

    async def get_session(self, session_id: str) -> list:
        return []

    async def add_to_session(self, session_id: str, message: Any) -> None:
        pass

    async def clear(self, session_id: Any = None, before: Any = None) -> None:
        pass


class _FakeFeedback(FeedbackLoop):
    async def store_result(self, content: Any, input_text: Any, metadata: Any = None) -> str:
        return "r"

    async def score(self, result_id: str, scores: Any) -> None:
        pass

    async def get_signals(self, query: str, limit: int = 3, min_score: Any = None, source: Any = None) -> list:
        return []

    async def query(self, q: Any) -> list:
        return []

    async def export_eval_set(self, since: Any = None, min_score: Any = None, max_score: Any = None, limit: Any = None) -> list:
        return []


class _FakeTracer(TracingLogger):
    def start_trace(self, name: str, metadata: Any = None, kind: SpanKind = SpanKind.AGENT_RUN) -> Span:
        return Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now())

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None, metadata: Any = None) -> Span:
        return Span(id="s", trace_id="t", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)

    def end_span(self, span_id: str, output: Any = None, error: Any = None, usage: Any = None) -> None:
        pass

    async def get_trace(self, trace_id: str) -> list:
        return []

    async def list_traces(self, since: Any = None, limit: int = 50, name: Any = None) -> list:
        return []

    async def flush(self) -> None:
        pass


def test_no_memory_quickstart_constructs() -> None:
    """README.md 'Quick start' no-memory example: name/description/system_prompt/llm/feedback/tracer/tools."""
    config = AgentConfig(
        name="my-agent",
        description="A simple agent",
        system_prompt="You are a helpful assistant.",
        llm=_FakeLLM(),
        feedback=_FakeFeedback(),
        tracer=_FakeTracer(),
        tools=ToolRegistry(),
    )
    assert config.name == "my-agent"
    assert config.store is None
    assert config.retriever is None


def test_local_memory_quickstart_constructs() -> None:
    """README.md 'Quick start' local-memory example: adds store= and retriever= (store/retriever split)."""
    store = _FakeMemory()
    retriever = _FakeMemory()

    config = AgentConfig(
        name="my-agent",
        description="A simple agent",
        system_prompt="You are a helpful assistant.",
        llm=_FakeLLM(),
        store=store,
        retriever=retriever,
        feedback=_FakeFeedback(),
        tracer=_FakeTracer(),
        tools=ToolRegistry(),
    )
    assert config.store is store
    assert config.retriever is retriever


def test_legacy_memory_kwarg_rejected() -> None:
    """AgentConfig has no `memory=` field; passing it must raise TypeError."""
    with pytest.raises(TypeError):
        AgentConfig(  # type: ignore[call-arg]
            name="my-agent",
            description="A simple agent",
            system_prompt="You are a helpful assistant.",
            llm=_FakeLLM(),
            memory=_FakeMemory(),  # removed field — should raise
            feedback=_FakeFeedback(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )
