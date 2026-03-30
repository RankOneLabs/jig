from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from jig import (
    AgentConfig,
    AgentResult,
    CompletionParams,
    EvalCase,
    LLMResponse,
    MemoryEntry,
    Message,
    Role,
    Score,
    ScoredResult,
    ScoreSource,
    Span,
    SpanKind,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    run_agent,
)
from jig.core.types import AgentMemory, FeedbackLoop, Grader, LLMClient, Tool, TracingLogger
from jig.tools import ToolRegistry


# --- Fakes ---


class FakeLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, params: CompletionParams) -> LLMResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class FakeMemory(AgentMemory):
    def __init__(self) -> None:
        self.stored: list[tuple[str, dict]] = []
        self.sessions: dict[str, list[Message]] = {}

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        self.stored.append((content, metadata or {}))
        return f"mem-{len(self.stored)}"

    async def query(self, query: str, limit: int = 5, filter: dict[str, Any] | None = None, session_id: str | None = None) -> list[MemoryEntry]:
        return []

    async def get_session(self, session_id: str) -> list[Message]:
        return self.sessions.get(session_id, [])

    async def add_to_session(self, session_id: str, message: Message) -> None:
        self.sessions.setdefault(session_id, []).append(message)

    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        pass


class FakeFeedback(FeedbackLoop):
    def __init__(self) -> None:
        self.scored: list[tuple[str, list[Score]]] = []

    async def score(self, result_id: str, scores: list[Score]) -> None:
        self.scored.append((result_id, scores))

    async def get_signals(self, query: str, limit: int = 3, min_score: float | None = None, source: ScoreSource | None = None) -> list[ScoredResult]:
        return []

    async def export_eval_set(self, since: datetime | None = None, min_score: float | None = None, max_score: float | None = None, limit: int | None = None) -> list[EvalCase]:
        return []


class FakeTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None, kind: SpanKind = SpanKind.AGENT_RUN) -> Span:
        s = Span(id="trace-0", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None) -> Span:
        s = Span(id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, input=input)
        self.spans.append(s)
        return s

    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Any = None) -> None:
        for s in self.spans:
            if s.id == span_id:
                s.ended_at = datetime.now()
                s.output = output
                s.error = error
                s.usage = usage

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(self, since: datetime | None = None, limit: int = 50, name: str | None = None) -> list[Span]:
        return [s for s in self.spans if s.kind == SpanKind.AGENT_RUN]

    async def flush(self) -> None:
        pass


class EchoTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="echo", description="Echoes input", parameters={"type": "object", "properties": {"text": {"type": "string"}}})

    async def execute(self, args: dict[str, Any]) -> str:
        return args.get("text", "")


# --- Tests ---


async def test_simple_completion():
    llm = FakeLLM([
        LLMResponse(content="Hello!", tool_calls=None, usage=Usage(10, 5), latency_ms=100, model="fake"),
    ])
    result = await run_agent(
        AgentConfig(
            name="test",
            description="test agent",
            system_prompt="You are a test agent.",
            llm=llm,
            memory=FakeMemory(),
            feedback=FakeFeedback(),
            tracer=FakeTracer(),
            tools=ToolRegistry(),
        ),
        "Hi",
    )
    assert result.output == "Hello!"
    assert result.usage["llm_calls"] == 1
    assert result.usage["tool_calls"] == 0


async def test_tool_loop():
    llm = FakeLLM([
        LLMResponse(
            content="Let me echo that.",
            tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "world"})],
            usage=Usage(10, 20),
            latency_ms=50,
            model="fake",
        ),
        LLMResponse(content="The echo said: world", tool_calls=None, usage=Usage(30, 10), latency_ms=50, model="fake"),
    ])
    result = await run_agent(
        AgentConfig(
            name="test",
            description="test agent",
            system_prompt="You are a test agent.",
            llm=llm,
            memory=FakeMemory(),
            feedback=FakeFeedback(),
            tracer=FakeTracer(),
            tools=ToolRegistry([EchoTool()]),
        ),
        "Echo world",
    )
    assert result.output == "The echo said: world"
    assert result.usage["llm_calls"] == 2
    assert result.usage["tool_calls"] == 1


async def test_max_tool_calls_enforced():
    # LLM always wants to call tools — should be capped
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "x"})],
        usage=Usage(5, 5),
        latency_ms=10,
        model="fake",
    )
    final = LLMResponse(content="Done", tool_calls=None, usage=Usage(5, 5), latency_ms=10, model="fake")
    llm = FakeLLM([tool_response, tool_response, tool_response, final])

    result = await run_agent(
        AgentConfig(
            name="test",
            description="test agent",
            system_prompt="test",
            llm=llm,
            memory=FakeMemory(),
            feedback=FakeFeedback(),
            tracer=FakeTracer(),
            tools=ToolRegistry([EchoTool()]),
            max_tool_calls=2,
        ),
        "go",
    )
    assert result.usage["tool_calls"] == 2


async def test_session_persistence():
    memory = FakeMemory()
    llm = FakeLLM([
        LLMResponse(content="I remember.", tool_calls=None, usage=Usage(10, 5), latency_ms=50, model="fake"),
    ])
    await run_agent(
        AgentConfig(
            name="test",
            description="test agent",
            system_prompt="test",
            llm=llm,
            memory=memory,
            feedback=FakeFeedback(),
            tracer=FakeTracer(),
            tools=ToolRegistry(),
            session_id="sess-1",
        ),
        "Remember me",
    )
    assert len(memory.sessions["sess-1"]) == 2
    assert memory.sessions["sess-1"][0].role == Role.USER
    assert memory.sessions["sess-1"][1].role == Role.ASSISTANT


async def test_auto_grading():
    class AlwaysPerfect(Grader):
        async def grade(self, input: str, output: str, context: dict[str, Any] | None = None) -> list[Score]:
            return [Score(dimension="quality", value=1.0, source=ScoreSource.HEURISTIC)]

    feedback = FakeFeedback()
    llm = FakeLLM([
        LLMResponse(content="Great answer", tool_calls=None, usage=Usage(10, 5), latency_ms=50, model="fake"),
    ])
    result = await run_agent(
        AgentConfig(
            name="test",
            description="test agent",
            system_prompt="test",
            llm=llm,
            memory=FakeMemory(),
            feedback=feedback,
            tracer=FakeTracer(),
            tools=ToolRegistry(),
            grader=AlwaysPerfect(),
        ),
        "test input",
    )
    assert result.scores is not None
    assert result.scores[0].value == 1.0
    assert len(feedback.scored) == 1


async def test_tool_registry_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute(ToolCall(id="x", name="nonexistent", arguments={}))
    assert result.error == "Unknown tool: nonexistent"


async def test_tool_registry_error_handling():
    class BadTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="bad", description="Always fails", parameters={})

        async def execute(self, args: dict[str, Any]) -> str:
            raise ValueError("boom")

    registry = ToolRegistry([BadTool()])
    result = await registry.execute(ToolCall(id="x", name="bad", arguments={}))
    assert result.error == "boom"


async def test_completion_params_provider_params():
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="hi")],
        system="be nice",
        temperature=0.5,
        provider_params={"top_k": 40},
    )
    assert params.provider_params == {"top_k": 40}
    assert params.system == "be nice"
    assert params.temperature == 0.5
