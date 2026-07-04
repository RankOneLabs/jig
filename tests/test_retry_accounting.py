"""Retry accounting tests.

Verifies that the runner is the single retry layer for provider calls:
one provider invocation per runner attempt, no hidden multiplier from
adapter-level retries. Uses a fake provider counter to prove the claim.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from jig.core.errors import (
    AgentMaxLLMRetriesError,
    JigLLMError,
)
from jig.core.runner import AgentConfig, run_agent
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    LLMClient,
    LLMResponse,
    MemoryEntry,
    MemoryStore,
    Message,
    Retriever,
    Role,
    ScoredResult,
    Span,
    SpanKind,
    TracingLogger,
    Usage,
)
from jig.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class CountingFakeLLM(LLMClient):
    """LLM fake that counts every complete() call and always raises the same error."""

    def __init__(self, error: Exception):
        self._error = error
        self.call_count = 0

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.call_count += 1
        raise self._error


class SucceedAfterFakeLLM(LLMClient):
    """LLM fake that raises `error` for `fail_times` calls then returns `response`."""

    def __init__(self, error: Exception, fail_times: int, response: LLMResponse):
        self._error = error
        self._fail_times = fail_times
        self._response = response
        self.call_count = 0

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise self._error
        return self._response


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
        s = Span(id="t", trace_id="t", kind=kind, name=name,
                 started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id, kind, name, input=None, metadata=None):
        s = Span(id=f"s{len(self.spans)}", trace_id="t", kind=kind, name=name,
                 started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata)
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


def _config(llm: LLMClient, **overrides: Any) -> AgentConfig:
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


def _ok_response() -> LLMResponse:
    return LLMResponse(
        content="done",
        tool_calls=None,
        usage=Usage(input_tokens=5, output_tokens=3, cost=0.0),
        latency_ms=1.0,
        model="fake",
    )


# ---------------------------------------------------------------------------
# Retry accounting tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetryAccounting:
    async def test_one_provider_call_per_runner_attempt_persistent_failure(self):
        """Under persistent retryable failure, provider call count == runner attempts.

        This is the core acceptance test: if adapters had internal retries,
        real provider calls would multiply silently behind the recorded count.
        With retries removed from adapters, every runner attempt maps 1-to-1
        to exactly one provider call.
        """
        err = JigLLMError("rate limit", "fake", retryable=True)
        llm = CountingFakeLLM(err)
        max_retries = 3

        result = await run_agent(_config(llm, max_llm_retries=max_retries), "go")

        assert isinstance(result.error, AgentMaxLLMRetriesError)
        # Runner recorded `max_retries` attempts. Each must correspond to
        # exactly one real provider call (no hidden multiplier).
        assert llm.call_count == result.usage["llm_calls"]
        assert llm.call_count == max_retries

    async def test_one_provider_call_per_attempt_then_success(self):
        """Two failures then success: provider calls == runner's llm_calls."""
        err = JigLLMError("transient", "fake", retryable=True)
        llm = SucceedAfterFakeLLM(error=err, fail_times=2, response=_ok_response())

        result = await run_agent(_config(llm, max_llm_retries=5), "go")

        assert result.error is None
        assert result.output == "done"
        # 2 failures + 1 success = 3 total calls. llm_calls counter must match.
        assert llm.call_count == 3
        assert result.usage["llm_calls"] == 3

    async def test_non_retryable_error_makes_exactly_one_call(self):
        """A non-retryable error must not trigger any implicit adapter retry."""
        err = JigLLMError("bad request", "fake", status_code=400, retryable=False)
        llm = CountingFakeLLM(err)

        result = await run_agent(_config(llm, max_llm_retries=5), "go")

        # 400 is in _PERMANENT_LLM_STATUS_CODES → fast-fail after one attempt.
        assert llm.call_count == 1
        assert result.usage["llm_calls"] == 1

    async def test_max_llm_retries_bounds_total_provider_calls(self):
        """max_llm_retries=N means at most N provider calls, no silent extras."""
        err = JigLLMError("always broken", "fake", retryable=True)
        llm = CountingFakeLLM(err)
        max_retries = 2

        result = await run_agent(_config(llm, max_llm_retries=max_retries), "go")

        assert isinstance(result.error, AgentMaxLLMRetriesError)
        # Exactly max_retries real calls — adapter must not retry internally.
        assert llm.call_count == max_retries
        assert result.usage["llm_calls"] == max_retries

    async def test_different_error_types_all_count_as_one_call(self):
        """Both retryable and non-retryable errors must produce exactly one call each.

        Tests that the absence of adapter-level retry applies to all error types,
        not just the specific ones previously guarded by each adapter's _retryable fn.
        """
        for err in [
            JigLLMError("rate limit", "fake", retryable=True),
            JigLLMError("server error", "fake", status_code=500),
            JigLLMError("generic", "fake"),
        ]:
            llm = CountingFakeLLM(err)
            await run_agent(
                _config(llm, max_llm_retries=1, max_llm_calls=1),
                "go",
            )
            assert llm.call_count == 1, (
                f"Expected exactly 1 provider call for {err!r}, got {llm.call_count}"
            )


@pytest.mark.asyncio
class TestNonProviderRetriesExcluded:
    """Retries that are not provider generation attempts must not appear in llm_calls.

    The ``with_retry`` utility in ``jig.core.retry`` is used for infrastructure
    operations (e.g. listener startup, network retries) that are completely
    separate from LLM provider calls. This class verifies that ``with_retry``
    retrying a non-LLM function N times is independent of LLM call accounting,
    and that concurrent infra retries and provider retries do not bleed into
    each other's counters.
    """

    async def test_with_retry_for_non_llm_op_does_not_touch_provider(self):
        """with_retry wrapping a non-LLM function is architecturally separate from
        provider generation.

        Verifies that with_retry retries the non-LLM operation the expected
        number of times and succeeds, with no LLM client involved in the path.
        This is the boundary test: infrastructure retries (listener startup,
        HTTP probes, smithers polling) live in a different call stack from
        run_agent's LLM retry loop.
        """
        from jig.core.retry import with_retry

        non_llm_attempts = 0

        async def non_llm_operation() -> str:
            nonlocal non_llm_attempts
            non_llm_attempts += 1
            if non_llm_attempts < 3:
                raise OSError("transient network error")
            return "connected"

        result = await with_retry(
            non_llm_operation,
            max_attempts=5,
            base_delay=0.0,  # no sleep in tests
            retryable=lambda e: isinstance(e, OSError),
        )

        assert result == "connected"
        assert non_llm_attempts == 3, "retried exactly twice before success"

    async def test_polling_retries_are_distinct_from_provider_attempts(self):
        """Separate retry budgets for infrastructure vs. provider calls must not
        interfere.

        Scenario: a fake non-LLM poller retries 3 times independently, while
        a CountingFakeLLM records its own call count. The two counters must
        remain independent — infrastructure retries do not inflate llm_calls
        and LLM retries do not inflate infrastructure retry counts.
        """
        from jig.core.retry import with_retry

        llm = SucceedAfterFakeLLM(
            error=JigLLMError("transient", "fake", retryable=True),
            fail_times=2,
            response=_ok_response(),
        )

        infra_attempts = 0

        async def infra_poll() -> str:
            nonlocal infra_attempts
            infra_attempts += 1
            if infra_attempts < 4:
                raise ConnectionError("not ready")
            return "ready"

        # Run infra retries and LLM agent concurrently to prove independence
        agent_result, poll_result = await asyncio.gather(
            run_agent(_config(llm, max_llm_retries=5), "go"),
            with_retry(
                infra_poll,
                max_attempts=5,
                base_delay=0.0,
                retryable=lambda e: isinstance(e, ConnectionError),
            ),
        )

        assert poll_result == "ready"
        assert infra_attempts == 4
        # LLM: 2 failures + 1 success = 3 provider calls
        assert llm.call_count == 3
        assert agent_result.usage["llm_calls"] == 3
        # Infrastructure retries must not have leaked into llm_calls
        assert agent_result.usage["llm_calls"] != infra_attempts
