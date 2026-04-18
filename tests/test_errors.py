"""Tests for the typed error hierarchy and AgentResult.error wiring."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from jig import (
    AgentConfig,
    AgentError,
    AgentMaxLLMCallsError,
    AgentMaxLLMRetriesError,
    AgentSchemaNotCalledError,
    AgentSchemaValidationError,
    CompletionParams,
    JigBudgetError,
    JigError,
    JigLLMError,
    JigMemoryError,
    JigToolError,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    Usage,
    run_agent,
)
from jig.core.types import (
    AgentMemory,
    FeedbackLoop,
    LLMClient,
    MemoryEntry,
    ScoredResult,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


# --- Fakes ---


class FakeLLM(LLMClient):
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, params):
        resp = self._responses[self._call_count]
        self._call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeMemory(AgentMemory):
    async def add(self, content, metadata=None): return "m"
    async def query(self, query, limit=5, filter=None, session_id=None): return []
    async def get_session(self, session_id): return []
    async def add_to_session(self, session_id, message): pass
    async def clear(self, session_id=None, before=None): pass


class FakeFeedback(FeedbackLoop):
    async def score(self, result_id, scores): pass
    async def get_signals(self, query, limit=3, min_score=None, source=None): return []
    async def export_eval_set(self, since=None, min_score=None, max_score=None, limit=None): return []


class FakeTracer(TracingLogger):
    def __init__(self):
        self.spans: list[Span] = []

    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):
        s = Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id, kind, name, input=None):
        s = Span(id=f"s{len(self.spans)}", trace_id="t", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, input=input)
        self.spans.append(s)
        return s

    def end_span(self, span_id, output=None, error=None, usage=None):
        for s in self.spans:
            if s.id == span_id:
                s.ended_at = datetime.now()
                s.output = output
                s.error = error

    async def get_trace(self, trace_id): return []
    async def list_traces(self, since=None, limit=50, name=None): return []
    async def flush(self): pass


def _config(llm, **overrides: Any) -> AgentConfig:
    defaults: dict[str, Any] = dict(
        name="t",
        description="test",
        system_prompt="be brief",
        llm=llm,
        memory=FakeMemory(),
        feedback=FakeFeedback(),
        tracer=FakeTracer(),
        tools=ToolRegistry(),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# --- Error class tests ---


class TestErrorHierarchy:
    def test_all_errors_inherit_jig_error(self):
        assert issubclass(JigLLMError, JigError)
        assert issubclass(JigMemoryError, JigError)
        assert issubclass(JigToolError, JigError)
        assert issubclass(JigBudgetError, JigError)
        assert issubclass(AgentError, JigError)

    def test_agent_errors_inherit_agent_error(self):
        assert issubclass(AgentMaxLLMCallsError, AgentError)
        assert issubclass(AgentMaxLLMRetriesError, AgentError)
        assert issubclass(AgentSchemaValidationError, AgentError)
        assert issubclass(AgentSchemaNotCalledError, AgentError)


class TestJigMemoryError:
    def test_fields(self):
        e = JigMemoryError("query failed", source="local_memory", operation="query", retryable=True)
        assert e.source == "local_memory"
        assert e.operation == "query"
        assert e.retryable is True

    def test_retryable_defaults_false(self):
        e = JigMemoryError("x", source="honcho", operation="add")
        assert e.retryable is False


class TestJigToolError:
    def test_fields(self):
        underlying = ValueError("boom")
        e = JigToolError(
            "tool failed",
            tool_name="write_strategy",
            phase="execute",
            retryable=False,
            underlying=underlying,
        )
        assert e.tool_name == "write_strategy"
        assert e.phase == "execute"
        assert e.retryable is False
        assert e.underlying is underlying

    def test_phase_accepts_schema_execute_serialize(self):
        # All three conventional phases construct cleanly
        for phase in ("schema", "execute", "serialize"):
            JigToolError("x", tool_name="t", phase=phase)


class TestAgentErrorCategories:
    """Each subclass should have a stable, distinct category tag."""

    def test_distinct_categories(self):
        cats = {
            AgentMaxLLMCallsError(50).category,
            AgentMaxLLMRetriesError(3, "x").category,
            AgentSchemaValidationError(2, "x").category,
            AgentSchemaNotCalledError(2).category,
        }
        assert len(cats) == 4

    def test_max_llm_calls(self):
        e = AgentMaxLLMCallsError(50)
        assert e.category == "max_llm_calls"
        assert e.max_calls == 50
        assert e.context == {"max_calls": 50}

    def test_max_llm_retries(self):
        e = AgentMaxLLMRetriesError(3, "timeout")
        assert e.category == "max_llm_retries"
        assert e.retries == 3
        assert e.last_error == "timeout"

    def test_schema_validation(self):
        e = AgentSchemaValidationError(2, "missing field: strategy_types")
        assert e.category == "schema_validation_failed"
        assert e.retries == 2

    def test_schema_not_called(self):
        e = AgentSchemaNotCalledError(3)
        assert e.category == "schema_not_called"
        assert e.retries == 3


# --- Runner wiring tests ---


@pytest.mark.asyncio
class TestAgentResultError:
    async def test_happy_path_error_is_none(self):
        llm = FakeLLM([
            LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(_config(llm), "hi")
        assert result.error is None
        assert result.output == "ok"

    async def test_max_llm_calls_sets_error(self):
        tool_resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc", name="none", arguments={})],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        llm = FakeLLM([tool_resp] * 3)
        result = await run_agent(_config(llm, max_tool_calls=1, max_llm_calls=3), "go")

        assert isinstance(result.error, AgentMaxLLMCallsError)
        assert result.error.max_calls == 3
        assert result.error.category == "max_llm_calls"

    async def test_max_llm_retries_sets_error(self):
        err = JigLLMError("network down", "fake", retryable=True)
        llm = FakeLLM([err, err, err])
        result = await run_agent(_config(llm, max_llm_retries=2), "go")

        assert isinstance(result.error, AgentMaxLLMRetriesError)
        assert result.error.retries == 2
        assert "network down" in result.error.last_error

    async def test_schema_validation_failed_sets_error(self):
        class Out(BaseModel):
            strategy_types: list[str]

        bad = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="c", name="submit_output", arguments={"wrong_field": 1})],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        llm = FakeLLM([bad, bad, bad])
        result = await run_agent(
            _config(llm, output_schema=Out, max_parse_retries=1),
            "go",
        )

        assert isinstance(result.error, AgentSchemaValidationError)
        assert result.error.retries == 2  # attempts: initial + 1 retry = 2 total
        assert result.parsed is None

    async def test_schema_not_called_sets_error(self):
        class Out(BaseModel):
            value: int

        plain = LLMResponse(content="prose", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
        llm = FakeLLM([plain, plain, plain])
        result = await run_agent(
            _config(llm, output_schema=Out, max_parse_retries=1),
            "go",
        )

        assert isinstance(result.error, AgentSchemaNotCalledError)
        assert result.error.retries == 2
        assert result.parsed is None


@pytest.mark.asyncio
class TestTraceErrorCategoryTagging:
    async def test_trace_span_metadata_carries_category(self):
        """Root span output includes error_category for rollup queries."""
        err = JigLLMError("boom", "fake")
        llm = FakeLLM([err, err])
        tracer = FakeTracer()
        await run_agent(_config(llm, tracer=tracer, max_llm_retries=2), "go")

        root_span = tracer.spans[0]
        assert root_span.kind == SpanKind.AGENT_RUN
        assert isinstance(root_span.output, dict)
        assert root_span.output.get("error_category") == "max_llm_retries"
        assert root_span.error is not None  # stringified error

    async def test_trace_error_absent_on_success(self):
        llm = FakeLLM([
            LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        tracer = FakeTracer()
        await run_agent(_config(llm, tracer=tracer), "hi")

        root_span = tracer.spans[0]
        assert isinstance(root_span.output, dict)
        assert "error_category" not in root_span.output
        assert root_span.error is None
