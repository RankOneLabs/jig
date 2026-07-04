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
from jig.tracing.spans import span_guard


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

    def start_span(self, parent_id, kind, name, input=None, metadata=None):
        s = Span(id=f"s{len(self.spans)}", trace_id="t", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata)
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
    """Post-output bookkeeping failures are fail-soft: the valid output is
    returned and the tracer is always flushed. Pre-output failures still
    propagate so callers know the run produced no valid answer."""

    async def test_flush_called_when_memory_add_raises(self):
        """memory.add failure after valid output is fail-soft: result returned, tracer flushed."""
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

        # memory.add failure is fail-soft after a valid output: no exception raised
        result = await run_agent(config, "hi")
        assert result.output == "ok"
        assert result.error is None

        # flush must have run via _finalize_trace in the finally block
        assert tracer.flush_count >= 1
        # Root span must be ended
        root = tracer.spans[0]
        assert root.ended_at is not None

    async def test_finalization_swallows_end_span_errors(self):
        """Broken tracer.end_span and fail-soft memory error both leave the result intact."""
        class FlakyMemory(FakeMemory):
            async def add(self, content, metadata=None):
                raise RuntimeError("memory write failed")

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

        # Both memory.add and end_span failures are swallowed; valid output returned
        result = await run_agent(config, "hi")
        assert result.output == "ok"
        assert result.error is None


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


@pytest.mark.asyncio
class TestTerminatedRunSkipsBookkeeping:
    """Terminated runs (agent_error set) must not persist sentinel text to
    memory, session, or feedback so downstream queries stay clean."""

    async def test_terminated_run_skips_memory_add(self):
        """memory.add is not called when the run terminates (max_llm_calls)."""
        add_calls: list[str] = []

        class TrackingMemory(FakeMemory):
            async def add(self, content, metadata=None):
                add_calls.append(content)
                return "m"

        llm = FakeLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc", name="none", arguments={})],
                usage=Usage(1, 1), latency_ms=1, model="fake",
            )
        ] * 3)
        result = await run_agent(
            _config(llm, store=TrackingMemory(), max_tool_calls=1, max_llm_calls=2),
            "go",
        )
        assert isinstance(result.error, AgentMaxLLMCallsError)
        assert add_calls == [], "memory.add must not be called for terminated runs"

    async def test_terminated_run_skips_auto_grading(self):
        """grader.grade is not called when the run terminates."""
        from jig.core.types import Grader, Score, ScoreSource

        grade_calls: list[Any] = []

        class TrackingGrader(Grader):
            async def grade(self, input, output, context=None):
                grade_calls.append(output)
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        err = JigLLMError("down", "fake", retryable=True)
        llm = FakeLLM([err, err])
        result = await run_agent(
            _config(llm, max_llm_retries=2, grader=TrackingGrader()),
            "go",
        )
        assert isinstance(result.error, AgentMaxLLMRetriesError)
        assert grade_calls == [], "grader.grade must not be called for terminated runs"

    async def test_terminated_run_skips_session_append(self):
        """Session history is not updated when the run terminates.

        Sentinel text like '[agent terminated: ...]' must not appear in the
        session assistant content so subsequent retrievals don't see garbage.
        """
        session_writes: list[tuple[str, str]] = []  # (session_id, role)

        class TrackingMemory(FakeMemory):
            async def add_to_session(self, session_id, message):
                session_writes.append((session_id, message.role.value))

        err = JigLLMError("down", "fake", retryable=True)
        llm = FakeLLM([err, err])
        result = await run_agent(
            _config(
                llm,
                max_llm_retries=2,
                store=TrackingMemory(),
                session_id="test-session",
            ),
            "go",
        )
        assert isinstance(result.error, AgentMaxLLMRetriesError)
        assert session_writes == [], (
            "add_to_session must not be called for terminated runs; "
            f"got {session_writes}"
        )

    async def test_terminated_run_skips_feedback_scoring(self):
        """Feedback store_result/score are not called when the run terminates."""
        store_calls: list[str] = []
        score_calls: list[str] = []

        class TrackingFeedback(FakeFeedback):
            async def store_result(self, content, input_text, metadata=None):
                store_calls.append(content)
                return "r"

            async def score(self, result_id, scores):
                score_calls.append(result_id)

        from jig.core.types import Grader, Score, ScoreSource

        class FixedGrader(Grader):
            async def grade(self, input, output, context=None):
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        err = JigLLMError("down", "fake", retryable=True)
        llm = FakeLLM([err, err])
        result = await run_agent(
            _config(
                llm,
                max_llm_retries=2,
                feedback=TrackingFeedback(),
                grader=FixedGrader(),
            ),
            "go",
        )
        assert isinstance(result.error, AgentMaxLLMRetriesError)
        assert store_calls == [], "feedback.store_result must not be called for terminated runs"
        assert score_calls == [], "feedback.score must not be called for terminated runs"


@pytest.mark.asyncio
class TestPreOutputSpanLifecycle:
    """Pre-output spans must be closed even when the underlying operation raises."""

    async def test_span_guard_closes_success_without_explicit_finish(self):
        """The span lifecycle helper owns the success close by default."""
        tracer = FakeTracer()
        root = tracer.start_trace("root")

        with span_guard(tracer, root.id, SpanKind.MEMORY_QUERY, "implicit_success"):
            pass

        guarded = next(s for s in tracer.spans if s.name == "implicit_success")
        assert guarded.ended_at is not None
        assert guarded.error is None

    async def test_retriever_failure_closes_mem_span(self):
        """If retriever.retrieve raises, the memory query span is closed with error."""
        class BrokenRetriever(FakeMemory):
            async def retrieve(self, query, k=5, context=None):
                raise RuntimeError("retriever down")

        tracer = FakeTracer()
        llm = FakeLLM([])  # never reached
        config = AgentConfig(
            name="t",
            description="test",
            system_prompt="",
            llm=llm,
            store=None, retriever=BrokenRetriever(),
            feedback=FakeFeedback(),
            tracer=tracer,
            tools=ToolRegistry(),
        )

        with pytest.raises(RuntimeError, match="retriever down"):
            await run_agent(config, "hi")

        # All spans should be closed (no orphan open spans)
        open_spans = [s for s in tracer.spans if s.ended_at is None]
        assert open_spans == [], f"Orphan open spans: {[s.name for s in open_spans]}"

        mem_span = next(s for s in tracer.spans if s.name == "retrieve")
        assert mem_span.error is not None
        assert "RuntimeError" in mem_span.error


@pytest.mark.asyncio
class TestPostOutputFailSoft:
    """Post-output bookkeeping failures must not erase a valid output.

    Each secondary operation (memory store, session append, feedback store/score,
    grader) is wrapped in its own try/except after the valid output is produced.
    Failures are logged and the output is returned unchanged.
    """

    async def test_grading_failure_does_not_erase_output(self):
        """If grader.grade raises, run_agent still returns the valid output."""
        from jig.core.types import Grader, Score

        class BrokenGrader(Grader):
            async def grade(self, input, output, context=None) -> list[Score]:
                raise RuntimeError("grader exploded")

        llm = FakeLLM([
            LLMResponse(content="the answer", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(llm, grader=BrokenGrader()),
            "hi",
        )

        assert result.output == "the answer"
        assert result.error is None
        assert result.scores is None  # grading didn't complete

    async def test_session_append_failure_does_not_erase_output(self):
        """store.add_to_session failure after valid output is fail-soft."""

        class FlakyMemory(FakeMemory):
            async def add_to_session(self, session_id, message):
                raise RuntimeError("session DB locked")

        llm = FakeLLM([
            LLMResponse(content="hello", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(llm, store=FlakyMemory(), session_id="s1"),
            "hi",
        )

        assert result.output == "hello"
        assert result.error is None

    async def test_feedback_store_failure_does_not_erase_output(self):
        """feedback.store_result failure during grading is fail-soft."""
        from jig.core.types import Grader, Score, ScoreSource

        class FixedGrader(Grader):
            async def grade(self, input, output, context=None):
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        class BrokenFeedback(FakeFeedback):
            async def store_result(self, content, input_text, metadata=None):
                raise RuntimeError("feedback DB write failed")

        llm = FakeLLM([
            LLMResponse(content="result", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(llm, feedback=BrokenFeedback(), grader=FixedGrader()),
            "hi",
        )

        assert result.output == "result"
        assert result.error is None

    async def test_trace_flush_failure_does_not_erase_output(self):
        """tracer.flush failure after valid output is fail-soft."""

        class BrokenFlushTracer(FakeTracer):
            async def flush(self):
                raise RuntimeError("trace sink unavailable")

        llm = FakeLLM([
            LLMResponse(content="result", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(llm, tracer=BrokenFlushTracer()),
            "hi",
        )

        assert result.output == "result"
        assert result.error is None

    async def test_schema_not_called_skips_all_bookkeeping(self):
        """Schema-not-called termination does not persist sentinel output."""
        class Out(BaseModel):
            value: int

        calls: list[str] = []

        class TrackingMemory(FakeMemory):
            async def add(self, content, metadata=None):
                calls.append(f"memory:{content}")
                return "m"

            async def add_to_session(self, session_id, message):
                calls.append(f"session:{message.role.value}:{message.content}")

        class TrackingFeedback(FakeFeedback):
            async def store_result(self, content, input_text, metadata=None):
                calls.append(f"feedback:{content}")
                return "r"

            async def score(self, result_id, scores):
                calls.append(f"score:{result_id}")

        from jig.core.types import Grader, Score, ScoreSource

        class TrackingGrader(Grader):
            async def grade(self, input, output, context=None):
                calls.append(f"grade:{output}")
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        llm = FakeLLM([
            LLMResponse(content="plain", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(
                llm,
                output_schema=Out,
                max_parse_retries=0,
                store=TrackingMemory(),
                session_id="s1",
                feedback=TrackingFeedback(),
                grader=TrackingGrader(),
            ),
            "hi",
        )

        assert isinstance(result.error, AgentSchemaNotCalledError)
        assert calls == []

    async def test_schema_validation_failure_skips_all_bookkeeping(self):
        """Schema-validation termination does not persist sentinel output."""
        class Out(BaseModel):
            value: int

        calls: list[str] = []

        class TrackingMemory(FakeMemory):
            async def add(self, content, metadata=None):
                calls.append(f"memory:{content}")
                return "m"

            async def add_to_session(self, session_id, message):
                calls.append(f"session:{message.role.value}:{message.content}")

        class TrackingFeedback(FakeFeedback):
            async def store_result(self, content, input_text, metadata=None):
                calls.append(f"feedback:{content}")
                return "r"

            async def score(self, result_id, scores):
                calls.append(f"score:{result_id}")

        from jig.core.types import Grader, Score, ScoreSource

        class TrackingGrader(Grader):
            async def grade(self, input, output, context=None):
                calls.append(f"grade:{output}")
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        llm = FakeLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c", name="submit_output", arguments={"wrong": 1})],
                usage=Usage(1, 1),
                latency_ms=1,
                model="fake",
            ),
        ])
        result = await run_agent(
            _config(
                llm,
                output_schema=Out,
                max_parse_retries=0,
                store=TrackingMemory(),
                session_id="s1",
                feedback=TrackingFeedback(),
                grader=TrackingGrader(),
            ),
            "hi",
        )

        assert isinstance(result.error, AgentSchemaValidationError)
        assert calls == []


# ---------------------------------------------------------------------------
# Adapter error boundary tests
# ---------------------------------------------------------------------------


class TestGeminiAdapterErrorBoundaries:
    """GeminiClient error classification and nullable field handling.

    These tests mock the google-genai SDK so they run without the optional
    dependency installed. Each test targets a provider-specific edge case
    called out by the lifecycle-hardening spec.
    """

    @pytest.mark.asyncio
    async def test_none_usage_metadata_normalizes_to_zero(self, monkeypatch):
        """usage_metadata=None from Gemini API yields 0 tokens, not AttributeError.

        Gemini can omit usage metadata on certain request types (e.g., cached
        content). The adapter must treat it as 0/0 so Usage.input_tokens and
        output_tokens are always integers.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        # Stub out genai at the module level so GeminiClient skips the ImportError guard
        genai_stub = MagicMock()
        genai_types_stub = MagicMock()

        # GenerateContentConfig must be constructable
        genai_types_stub.GenerateContentConfig.return_value = MagicMock()
        genai_types_stub.Content = MagicMock
        genai_types_stub.Part = MagicMock

        candidate = MagicMock()
        candidate.content.parts = []  # no tool calls, no text
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = None  # the edge case

        genai_stub.Client.return_value.aio.models.generate_content = AsyncMock(return_value=response)

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.genai": genai_stub,
            "google.genai.types": genai_types_stub,
        }):
            with patch("jig.llm.google.genai", genai_stub), \
                 patch("jig.llm.google.genai_types", genai_types_stub):
                from jig.llm.google import GeminiClient
                client = GeminiClient.__new__(GeminiClient)
                client._client = genai_stub.Client()
                client._model = "gemini-test"

                params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
                result = await client.complete(params)

        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_none_token_counts_normalize_to_zero(self, monkeypatch):
        """Individual token count fields that are None become 0.

        prompt_token_count and candidates_token_count can both be None on
        empty/cached responses. The adapter normalizes via `or 0`.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        genai_stub = MagicMock()
        genai_types_stub = MagicMock()
        genai_types_stub.GenerateContentConfig.return_value = MagicMock()
        genai_types_stub.Content = MagicMock
        genai_types_stub.Part = MagicMock

        candidate = MagicMock()
        candidate.content.parts = []
        response = MagicMock()
        response.candidates = [candidate]
        # usage_metadata present but individual counts are None
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None

        genai_stub.Client.return_value.aio.models.generate_content = AsyncMock(return_value=response)

        with patch("jig.llm.google.genai", genai_stub), \
             patch("jig.llm.google.genai_types", genai_types_stub):
            from jig.llm.google import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._client = genai_stub.Client()
            client._model = "gemini-test"

            params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
            result = await client.complete(params)

        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_request_preparation_failure_raises_jig_llm_error(self):
        """A failure in _convert_messages wraps in JigLLMError, not a raw exception.

        Schema conversion or FunctionResponse validation can raise. The outer
        try/except in GeminiClient.complete() must catch these and re-raise as
        JigLLMError so the runner can classify the error correctly.
        """
        from unittest.mock import MagicMock, patch

        genai_stub = MagicMock()
        genai_types_stub = MagicMock()
        # Make GenerateContentConfig raise to trigger the request-prep error boundary
        genai_types_stub.GenerateContentConfig.side_effect = ValueError("nested schema unsupported")
        genai_types_stub.Content = MagicMock
        genai_types_stub.Part = MagicMock

        with patch("jig.llm.google.genai", genai_stub), \
             patch("jig.llm.google.genai_types", genai_types_stub):
            from jig.llm.google import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._client = genai_stub.Client()
            client._model = "gemini-test"

            params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
            with pytest.raises(JigLLMError) as exc_info:
                await client.complete(params)

        err = exc_info.value
        assert err.provider == "google"
        assert "Request preparation failed" in str(err) or "nested schema unsupported" in str(err)


class TestOllamaAdapterErrorBoundaries:
    """OllamaClient transport error classification.

    Verifies that httpx transport errors are wrapped in JigLLMError
    so the runner can classify and route them correctly.
    """

    @pytest.mark.asyncio
    async def test_httpx_read_error_becomes_jig_llm_error_retryable(self):
        """httpx.ReadError (transport-level) is classified as retryable JigLLMError.

        ReadError is a subclass of TransportError — it occurs when the TCP
        connection drops mid-response. The adapter wraps it as retryable so
        the runner's consecutive-errors path handles transient Ollama restarts.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        ollama_stub = MagicMock()
        ollama_stub.AsyncClient = MagicMock

        with patch("jig.llm.ollama._ollama", ollama_stub), \
             patch("jig.llm.ollama.OllamaAsyncClient", ollama_stub.AsyncClient):
            from jig.llm.ollama import OllamaClient
            client = OllamaClient.__new__(OllamaClient)
            client._client = MagicMock()
            client._model = "llama3.1"

            # httpx.ReadError is a TransportError — simulates dropped connection
            read_error = httpx.ReadError("connection reset", request=MagicMock())
            client._client.chat = AsyncMock(side_effect=read_error)

            params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
            with pytest.raises(JigLLMError) as exc_info:
                await client.complete(params)

        err = exc_info.value
        assert err.provider == "ollama"
        assert err.retryable is True

    @pytest.mark.asyncio
    async def test_httpx_status_error_preserves_status_code(self):
        """httpx.HTTPStatusError status code is preserved in JigLLMError.

        When Ollama returns a non-2xx response, the status code must flow
        through so the runner's permanent-error detection can act on 4xx.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        ollama_stub = MagicMock()

        with patch("jig.llm.ollama._ollama", ollama_stub), \
             patch("jig.llm.ollama.OllamaAsyncClient", ollama_stub.AsyncClient):
            from jig.llm.ollama import OllamaClient
            client = OllamaClient.__new__(OllamaClient)
            client._client = MagicMock()
            client._model = "llama3.1"

            mock_response = MagicMock()
            mock_response.status_code = 503
            http_err = httpx.HTTPStatusError(
                "503 service unavailable",
                request=MagicMock(),
                response=mock_response,
            )
            client._client.chat = AsyncMock(side_effect=http_err)

            params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
            with pytest.raises(JigLLMError) as exc_info:
                await client.complete(params)

        err = exc_info.value
        assert err.provider == "ollama"
        assert err.status_code == 503
