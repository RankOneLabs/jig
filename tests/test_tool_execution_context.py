from __future__ import annotations

import asyncio

from datetime import datetime
from typing import Any

import pytest

from jig import (
    AgentConfig,
    LLMResponse,
    Score,
    Span,
    SpanKind,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    TracingLogger,
    Usage,
    current_tool_context,
    run_agent,
)


class _FakeTracer(TracingLogger):
    def __init__(self) -> None:
        self._spans: dict[str, Span] = {}
        self._counter = 0

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        self._counter += 1
        span = Span(
            id=f"root-{self._counter}",
            trace_id=f"trace-{self._counter}",
            kind=kind,
            name=name,
            started_at=datetime.now(),
            metadata=metadata,
        )
        self._spans[span.id] = span
        return span

    def start_span(
        self,
        parent_id: str,
        kind: SpanKind,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        self._counter += 1
        parent = self._spans.get(parent_id)
        trace_id = parent.trace_id if parent else parent_id
        span = Span(
            id=f"span-{self._counter}",
            trace_id=trace_id,
            kind=kind,
            name=name,
            started_at=datetime.now(),
            parent_id=parent_id,
            input=input,
            metadata=metadata,
        )
        self._spans[span.id] = span
        return span

    def end_span(
        self,
        span_id: str,
        output: Any = None,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        span = self._spans.get(span_id)
        if span is None:
            return
        span.output = output
        span.error = error
        span.usage = usage
        span.ended_at = datetime.now()

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [span for span in self._spans.values() if span.trace_id == trace_id]

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        spans = [span for span in self._spans.values() if span.kind == SpanKind.AGENT_RUN]
        if name is not None:
            spans = [span for span in spans if span.name == name]
        return spans[:limit]


class _FakeFeedback:
    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: Any = None,
    ):
        return []

    async def score(self, result_id: str, scores: list[Score]) -> None:
        return None

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "result-1"

    async def query(self, q: Any):
        return []

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ):
        return []


class _LegacyTool(Tool):
    def __init__(self) -> None:
        self.seen_contexts: list[ToolExecutionContext | None] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="legacy_tool",
            description="legacy",
            parameters={"type": "object"},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        self.seen_contexts.append(current_tool_context.get())
        return "legacy-ok"


class _ContextTool(Tool):
    def __init__(self) -> None:
        self.seen_contexts: list[ToolExecutionContext | None] = []
        self.seen_current: list[ToolExecutionContext | None] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="context_tool",
            description="context",
            parameters={"type": "object"},
        )

    async def execute_with_context(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> str:
        self.seen_contexts.append(context)
        self.seen_current.append(current_tool_context.get())
        return "context-ok"

    async def execute(self, args: dict[str, Any]) -> str:
        return "unexpected"


class _DispatchTool(Tool):
    dispatch = True

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dispatch_tool",
            description="dispatch",
            parameters={"type": "object"},
        )

    @property
    def dispatch_fn_ref(self) -> str | None:
        return "run_fake_dispatch"

    def dispatch_payload_extra(
        self,
        context: ToolExecutionContext | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "trace_id": context.trace_id if context else None,
            "argument_count": len(arguments or {}),
        }

    async def execute(self, args: dict[str, Any]) -> str:
        return "unused"


class _ReorderedDispatchTool(_DispatchTool):
    def dispatch_payload_extra(
        self,
        arguments: dict[str, Any] | None = None,
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        return {
            "hook": "reordered",
            "trace_id": context.trace_id if context else None,
            "argument_count": len(arguments or {}),
        }


class _KeywordOnlyDispatchTool(_DispatchTool):
    def dispatch_payload_extra(
        self,
        *,
        arguments: dict[str, Any] | None = None,
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        return {
            "hook": "keyword-only",
            "trace_id": context.trace_id if context else None,
            "argument_count": len(arguments or {}),
        }


class _AsyncDispatchTool(_DispatchTool):
    async def dispatch_payload_extra(self, context=None, arguments=None):
        await asyncio.sleep(0)
        return {"async_hook": True, "trace_id": context.trace_id if context else None}


class _NeverFinishingDispatchTool(_DispatchTool):
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def dispatch_payload_extra(self, context=None, arguments=None):
        try:
            await asyncio.Future()
        finally:
            self.cancelled.set()


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, params: Any) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tool-call-1", name="legacy_tool", arguments={})],
                usage=Usage(input_tokens=1, output_tokens=1),
                latency_ms=0.0,
                model="fake",
            )
        return LLMResponse(
            content="done",
            tool_calls=None,
            usage=Usage(input_tokens=1, output_tokens=1),
            latency_ms=0.0,
            model="fake",
        )


@pytest.mark.asyncio
async def test_legacy_execute_sees_current_tool_context() -> None:
    tool = _LegacyTool()
    registry = ToolRegistry([tool])
    context = ToolExecutionContext(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id="root-span",
        tool_call_id="call-1",
    )

    token = current_tool_context.set(context)
    try:
        result = await registry.execute(ToolCall(id="call-1", name="legacy_tool", arguments={}))
    finally:
        current_tool_context.reset(token)

    assert result.output == "legacy-ok"
    assert tool.seen_contexts == [context]


@pytest.mark.asyncio
async def test_execute_with_context_receives_tool_context() -> None:
    tool = _ContextTool()
    registry = ToolRegistry([tool])
    context = ToolExecutionContext(
        trace_id="trace-2",
        span_id="span-2",
        parent_span_id="root-span",
        tool_call_id="call-2",
        metadata={"tool_name": "context_tool"},
    )

    token = current_tool_context.set(context)
    try:
        result = await registry.execute(ToolCall(id="call-2", name="context_tool", arguments={}))
    finally:
        current_tool_context.reset(token)

    assert result.output == "context-ok"
    assert tool.seen_contexts == [context]
    assert tool.seen_current == [context]


@pytest.mark.asyncio
async def test_run_agent_sets_and_resets_current_tool_context() -> None:
    tracer = _FakeTracer()
    tool = _LegacyTool()
    registry = ToolRegistry([tool])
    llm = _FakeLLM()

    config = AgentConfig(
        name="context-agent",
        description="test",
        system_prompt="system",
        llm=llm,
        feedback=_FakeFeedback(),
        tracer=tracer,
        tools=registry,
        include_feedback_in_prompt=False,
        include_memory_in_prompt=False,
    )

    result = await run_agent(config, "hello")

    assert result.output == "done"
    assert tool.seen_contexts and tool.seen_contexts[0] is not None
    seen = tool.seen_contexts[0]
    assert seen is not None
    assert seen.trace_id == result.trace_id
    assert seen.tool_call_id == "tool-call-1"
    assert seen.span_id.startswith("span-")
    spans = await tracer.get_trace(result.trace_id)
    agent_span = next(span for span in spans if span.kind == SpanKind.AGENT_RUN)
    assert agent_span.id != result.trace_id
    assert seen.parent_span_id == agent_span.id
    assert current_tool_context.get() is None


@pytest.mark.asyncio
async def test_dispatch_payload_uses_current_tool_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _DispatchTool()
    registry = ToolRegistry([tool], dispatch_url="http://127.0.0.1:9", execute_timeout=1.9)
    context = ToolExecutionContext(
        trace_id="trace-3",
        span_id="span-3",
        parent_span_id="root-span",
        tool_call_id="call-3",
    )

    captured: dict[str, Any] = {}

    async def fake_dispatch_run(
        fn_ref: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        captured["fn_ref"] = fn_ref
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        captured["context"] = current_tool_context.get()
        return "dispatch-ok"

    monkeypatch.setattr("jig.dispatch.run", fake_dispatch_run)

    token = current_tool_context.set(context)
    try:
        result = await registry.execute(
            ToolCall(id="call-3", name="dispatch_tool", arguments={"x": 1})
        )
    finally:
        current_tool_context.reset(token)

    assert result.output == "dispatch-ok"
    assert captured["fn_ref"] == "run_fake_dispatch"
    assert captured["payload"] == {
        "x": 1,
        "trace_id": "trace-3",
        "argument_count": 1,
    }
    assert captured["kwargs"]["dispatch_url"] == "http://127.0.0.1:9"
    assert captured["kwargs"]["timeout_seconds"] == 2
    assert captured["kwargs"]["trace_context"] == {
        "trace_id": "trace-3",
        "parent_span_id": "span-3",
    }
    assert captured["context"] == context


@pytest.mark.asyncio
async def test_async_dispatch_payload_extra_is_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_dispatch_run(fn_ref, payload=None, **kwargs):
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr("jig.dispatch.run", fake_dispatch_run)
    context = ToolExecutionContext(
        trace_id="trace-async", span_id="span", parent_span_id="root", tool_call_id="call"
    )
    token = current_tool_context.set(context)
    try:
        result = await ToolRegistry([_AsyncDispatchTool()]).execute(
            ToolCall(id="call", name="dispatch_tool", arguments={"x": 1})
        )
    finally:
        current_tool_context.reset(token)

    assert result.error is None
    assert captured["payload"] == {"x": 1, "async_hook": True, "trace_id": "trace-async"}


@pytest.mark.asyncio
async def test_async_dispatch_hook_timeout_cancels_without_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _NeverFinishingDispatchTool()
    dispatch_calls = 0

    async def fake_dispatch_run(fn_ref, payload=None, **kwargs):
        nonlocal dispatch_calls
        dispatch_calls += 1
        return "unexpected"

    monkeypatch.setattr("jig.dispatch.run", fake_dispatch_run)
    result = await ToolRegistry([tool], execute_timeout=0.01).execute(
        ToolCall(id="call", name="dispatch_tool", arguments={})
    )

    assert result.error == "TimeoutError: Dispatched tool dispatch_tool timed out after 0.01s"
    assert tool.cancelled.is_set()
    assert dispatch_calls == 0


@pytest.mark.asyncio
async def test_dispatch_payload_extra_matches_context_and_arguments_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, Any] | None] = []

    async def fake_dispatch_run(
        fn_ref: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        captured_payloads.append(payload)
        return "dispatch-ok"

    monkeypatch.setattr("jig.dispatch.run", fake_dispatch_run)
    context = ToolExecutionContext(
        trace_id="trace-4",
        span_id="span-4",
        parent_span_id="root-span",
        tool_call_id="call-4",
    )

    token = current_tool_context.set(context)
    try:
        for tool in (_ReorderedDispatchTool(), _KeywordOnlyDispatchTool()):
            registry = ToolRegistry([tool])
            result = await registry.execute(
                ToolCall(id="call-4", name="dispatch_tool", arguments={"x": 1})
            )
            assert result.output == "dispatch-ok"
    finally:
        current_tool_context.reset(token)

    assert captured_payloads == [
        {
            "x": 1,
            "hook": "reordered",
            "trace_id": "trace-4",
            "argument_count": 1,
        },
        {
            "x": 1,
            "hook": "keyword-only",
            "trace_id": "trace-4",
            "argument_count": 1,
        },
    ]
