"""Tests for the typed error hierarchy and AgentResult.error wiring."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from jig import (
    AgentAmbiguousTurnError,
    AgentConfig,
    AgentError,
    AgentLLMPermanentError,
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
    MemoryStore,
    Retriever,
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


class FakeMemory(MemoryStore, Retriever):
    async def add(self, content, metadata=None): return "m"
    async def get(self, id): return None
    async def all(self): return []
    async def delete(self, id): pass
    async def retrieve(self, query, k=5, context=None): return []
    async def get_session(self, session_id): return []
    async def add_to_session(self, session_id, message): pass
    async def clear(self, session_id=None, before=None): pass


class FakeFeedback(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None): return "r"
    async def score(self, result_id, scores): pass
    async def get_signals(self, query, limit=3, min_score=None, source=None): return []
    async def query(self, q): return []
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
        store=FakeMemory(), retriever=None,
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
        assert issubclass(AgentAmbiguousTurnError, AgentError)
        assert issubclass(AgentLLMPermanentError, AgentError)


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
            AgentAmbiguousTurnError(2).category,
            AgentLLMPermanentError("p", "m", 500).category,
        }
        assert len(cats) == 6

    def test_ambiguous_turn(self):
        e = AgentAmbiguousTurnError(2)
        assert e.category == "ambiguous_tool_turn"
        assert e.retries == 2

    def test_llm_permanent_error(self):
        e = AgentLLMPermanentError(
            provider="anthropic",
            message="invalid api key",
            status_code=401,
        )
        assert e.category == "llm_permanent_error"
        assert e.provider == "anthropic"
        assert e.status_code == 401
        assert "invalid api key" in str(e)

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

    async def test_ambiguous_turn_sets_error(self):
        """When model keeps combining submit_output with other tool calls."""
        class Out(BaseModel):
            strategy_types: list[str]

        mixed = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "x"}),
                ToolCall(id="t2", name="submit_output",
                         arguments={"strategy_types": ["x"]}),
            ],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        llm = FakeLLM([mixed, mixed, mixed])
        result = await run_agent(
            _config(llm, output_schema=Out, max_parse_retries=1),
            "go",
        )

        assert isinstance(result.error, AgentAmbiguousTurnError)
        assert result.error.category == "ambiguous_tool_turn"
        assert result.parsed is None

    async def test_llm_calls_counts_attempts_not_successes(self):
        """A flaky LLM must still hit max_llm_calls, not race past it."""
        # Mix successes and failures. With max_llm_calls=3, we should
        # terminate after 3 attempts regardless of how many succeeded.
        err = JigLLMError("transient", "fake", retryable=True)
        ok = LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
        # Sequence: err, err, ok — the 3rd attempt succeeds and returns
        llm = FakeLLM([err, err, ok])
        result = await run_agent(
            _config(llm, max_llm_calls=3, max_llm_retries=5),
            "go",
        )
        # Happy path: the 3rd attempt succeeded before hitting the cap
        assert result.error is None
        assert result.usage["llm_calls"] == 3

    async def test_llm_calls_cap_trips_on_persistent_failures(self):
        """All failures → max_llm_calls terminates before max_llm_retries."""
        err = JigLLMError("always broken", "fake", retryable=True)
        # max_llm_calls=2 < max_llm_retries=5, so the calls cap trips first
        llm = FakeLLM([err] * 5)
        result = await run_agent(
            _config(llm, max_llm_calls=2, max_llm_retries=5),
            "go",
        )
        # Attempts are counted; cap trips on the 3rd loop iteration
        assert result.usage["llm_calls"] == 2
        # Terminated via AgentMaxLLMCallsError (llm_calls cap) rather than
        # AgentMaxLLMRetriesError (consecutive errors cap), because the
        # calls cap is lower.
        assert isinstance(result.error, AgentMaxLLMCallsError)

    async def test_permanent_status_terminates_fast(self):
        """status_code in permanent set → one attempt, AgentLLMPermanentError."""
        err = JigLLMError(
            "invalid api key", "anthropic", status_code=401, retryable=False,
        )
        # Buffer many retries; if the runner ignored the permanent signal
        # it would consume several before hitting max_llm_retries
        llm = FakeLLM([err] * 5)
        result = await run_agent(
            _config(llm, max_llm_retries=5, max_llm_calls=10),
            "go",
        )

        assert isinstance(result.error, AgentLLMPermanentError)
        assert result.error.provider == "anthropic"
        assert result.error.status_code == 401
        # Only one attempt was made — budget preserved
        assert result.usage["llm_calls"] == 1

    async def test_transient_500_does_not_fast_fail(self):
        """500 with retryable=False should still use the retry path.

        Locks in the fix for the case where adapters conservatively default
        retryable=False on unknown server errors. We must NOT treat every
        non-retryable error as permanent — only known-permanent status codes
        (auth/bad-request/not-found) trigger the fast-fail path.
        """
        err = JigLLMError(
            "internal server error", "fake", status_code=500, retryable=False,
        )
        ok = LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
        # If runner fast-failed on 500, it would terminate after 1 call.
        # Correct behavior: retry and succeed on the 3rd attempt.
        llm = FakeLLM([err, err, ok])
        result = await run_agent(_config(llm, max_llm_retries=5), "go")

        assert result.error is None
        assert result.output == "ok"
        assert result.usage["llm_calls"] == 3

    async def test_unknown_error_without_status_uses_retry_path(self):
        """status_code=None → retry path, not fast-fail."""
        err = JigLLMError("generic failure", "fake", retryable=False)
        ok = LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
        llm = FakeLLM([err, ok])
        result = await run_agent(_config(llm, max_llm_retries=5), "go")

        assert result.error is None
        assert result.usage["llm_calls"] == 2

    async def test_retryable_llm_error_uses_retry_path(self):
        """retryable=True preserves existing consecutive-retries behavior."""
        err = JigLLMError("rate limit", "fake", retryable=True)
        ok = LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
        llm = FakeLLM([err, err, ok])
        result = await run_agent(_config(llm, max_llm_retries=5), "go")

        assert result.error is None
        assert result.output == "ok"
        assert result.usage["llm_calls"] == 3


@pytest.mark.asyncio
class TestTracerFinalizationOnException:
    """run_agent's try/finally ensures buffered tracers flush even when an
    exception propagates from memory.add / grading / etc. mid-run."""

    async def test_flush_called_when_memory_add_raises(self):
        class FlakyMemory(FakeMemory):
            async def add(self, content, metadata=None):
                raise RuntimeError("memory write failed")

        class FlushTrackingTracer(FakeTracer):
            def __init__(self):
                super().__init__()
                self.flush_count = 0

            async def flush(self):
                self.flush_count += 1

        tracer = FlushTrackingTracer()
        llm = FakeLLM([
            LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        config = AgentConfig(
            name="t",
            description="test",
            system_prompt="",
            llm=llm,
            store=FlakyMemory(), retriever=None,
            feedback=FakeFeedback(),
            tracer=tracer,
            tools=ToolRegistry(),
        )

        with pytest.raises(RuntimeError, match="memory write failed"):
            await run_agent(config, "hi")

        # flush must have run despite the memory exception propagating
        assert tracer.flush_count == 1
        # Root span must be ended
        root = tracer.spans[0]
        assert root.ended_at is not None

    async def test_finalization_swallows_end_span_errors(self):
        """A broken tracer.end_span must not shadow the original exception."""
        class FlakyMemory(FakeMemory):
            async def add(self, content, metadata=None):
                raise RuntimeError("original error")

        class BrokenEndSpanTracer(FakeTracer):
            def end_span(self, span_id, output=None, error=None, usage=None):
                if span_id == "t":  # root trace id in the fake
                    raise RuntimeError("tracer corrupt")
                super().end_span(span_id, output, error, usage)

        llm = FakeLLM([
            LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        config = AgentConfig(
            name="t",
            description="test",
            system_prompt="",
            llm=llm,
            store=FlakyMemory(), retriever=None,
            feedback=FakeFeedback(),
            tracer=BrokenEndSpanTracer(),
            tools=ToolRegistry(),
        )

        # Original exception bubbles up; tracer error is logged and swallowed
        with pytest.raises(RuntimeError, match="original error"):
            await run_agent(config, "hi")


@pytest.mark.asyncio
class TestTraceErrorCategoryTagging:
    async def test_trace_span_metadata_carries_category(self):
        """Root span output includes error_category for rollup queries."""
        # Retryable=True so the error feeds the max_llm_retries path
        # rather than the new permanent-error fast-fail.
        err = JigLLMError("boom", "fake", retryable=True)
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


@pytest.mark.asyncio
class TestTracerFlushDefault:
    """TracingLogger.flush is a no-op by default so run_agent can call it
    on any tracer implementation without AttributeError."""

    async def test_stdout_tracer_flush_default_is_noop(self):
        from jig.tracing.stdout import StdoutTracer

        tracer = StdoutTracer(color=False)
        # Should not raise — inherited default on TracingLogger
        await tracer.flush()

    async def test_run_agent_with_stdout_tracer_does_not_raise(self):
        """End-to-end sanity: run_agent calls tracer.flush() at the end; with
        the no-op default on TracingLogger, StdoutTracer works unchanged."""
        from jig.tracing.stdout import StdoutTracer

        llm = FakeLLM([
            LLMResponse(content="ok", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        # Must not raise AttributeError on tracer.flush()
        result = await run_agent(_config(llm, tracer=StdoutTracer(color=False)), "hi")
        assert result.output == "ok"
