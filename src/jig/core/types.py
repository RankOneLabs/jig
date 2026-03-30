from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolResult:
    call_id: str
    output: str
    error: str | None = None


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cost: float | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: Usage
    latency_ms: float
    model: str


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class CompletionParams:
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    provider_params: dict[str, Any] | None = None


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None
    created_at: datetime = field(default_factory=datetime.now)


class ScoreSource(str, Enum):
    LLM_JUDGE = "llm_judge"
    HEURISTIC = "heuristic"
    HUMAN = "human"
    GROUND_TRUTH = "ground_truth"


@dataclass
class Score:
    dimension: str
    value: float
    source: ScoreSource


@dataclass
class ScoredResult:
    result_id: str
    content: str
    scores: list[Score]
    avg_score: float
    metadata: dict[str, Any]
    created_at: datetime


@dataclass
class EvalCase:
    input: str
    expected: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SpanKind(str, Enum):
    AGENT_RUN = "agent_run"
    PIPELINE_RUN = "pipeline_run"
    PIPELINE_STEP = "pipeline_step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_QUERY = "memory_query"
    GRADING = "grading"


@dataclass
class Span:
    id: str
    trace_id: str
    kind: SpanKind
    name: str
    started_at: datetime
    parent_id: str | None = None
    input: Any = None
    output: Any = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None
    usage: Usage | None = None


# --- Abstract interfaces ---


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, params: CompletionParams) -> LLMResponse: ...

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not implemented for this provider")
        yield  # noqa: unreachable — makes this a generator


class AgentMemory(ABC):
    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str: ...

    @abstractmethod
    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryEntry]: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> list[Message]: ...

    @abstractmethod
    async def add_to_session(self, session_id: str, message: Message) -> None: ...

    @abstractmethod
    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None: ...


class FeedbackLoop(ABC):
    @abstractmethod
    async def score(self, result_id: str, scores: list[Score]) -> None: ...

    @abstractmethod
    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]: ...

    @abstractmethod
    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]: ...


class Grader(ABC):
    @abstractmethod
    async def grade(
        self,
        input: Any,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> list[Score]: ...


class TracingLogger(ABC):
    @abstractmethod
    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span: ...

    @abstractmethod
    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None) -> Span: ...

    @abstractmethod
    def end_span(self, span_id: str, output: Any = None, error: str | None = None, usage: Usage | None = None) -> None: ...

    @abstractmethod
    async def get_trace(self, trace_id: str) -> list[Span]: ...

    @abstractmethod
    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]: ...


class Tool(ABC):
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str: ...
