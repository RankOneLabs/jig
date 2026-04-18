from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


@dataclass
class FeedbackQuery:
    """Filter spec for :meth:`FeedbackLoop.query`.

    All fields are optional; unset fields are no-ops. When ``similar_to``
    is provided, the backend orders candidates by embedding similarity
    before applying metadata filters. Without it, results are ordered by
    ``created_at`` descending.

    ``tags`` matches any overlap with the stored list in
    ``metadata["tags"]``. ``agent_name`` and ``model`` match exact keys
    in ``metadata`` — set those keys at ``store_result`` time for them
    to be filterable.
    """

    similar_to: str | None = None
    agent_name: str | None = None
    model: str | None = None
    tags: list[str] | None = None
    min_score: float | None = None
    max_age: timedelta | None = None
    limit: int = 10

    def __post_init__(self) -> None:
        # Reject bool explicitly: ``bool`` is an ``int`` subclass so
        # ``isinstance(True, int)`` would silently accept it as limit=1.
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit < 1
        ):
            raise ValueError(
                f"FeedbackQuery.limit must be a positive int, got {self.limit!r}"
            )


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


@dataclass
class TraceContext:
    """Carries trace identity across a dispatch boundary.

    Serialized into smithers job payloads so workers can start their
    spans as children of the caller's span. Phase 7+8 establishes the
    protocol (payload propagation); phase 9 wires the reader side on
    workers so the spans actually reparent.
    """

    trace_id: str
    parent_span_id: str

    def to_dict(self) -> dict[str, str]:
        return {"trace_id": self.trace_id, "parent_span_id": self.parent_span_id}

    @classmethod
    def from_dict(cls, data: Any) -> TraceContext | None:
        # Accept ``Any`` rather than ``dict`` because this is usually
        # called on JSON straight off the wire, which can arrive as null,
        # a string, a list, etc. when a worker misbehaves. Return None
        # for everything that isn't a well-formed mapping of strings.
        if not isinstance(data, dict):
            return None
        tid = data.get("trace_id")
        pid = data.get("parent_span_id")
        if not isinstance(tid, str) or not isinstance(pid, str):
            return None
        return cls(trace_id=tid, parent_span_id=pid)


# --- Abstract interfaces ---


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, params: CompletionParams) -> LLMResponse: ...

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not implemented for this provider")
        yield  # noqa: unreachable — makes this a generator

    async def aclose(self) -> None:
        """Release any resources held by the client.

        Default is a no-op. Providers that manage their own connections
        (e.g. ``DispatchClient`` holds an ``httpx.AsyncClient``) should
        override this to close them. Safe to call multiple times.
        """
        return None


class MemoryStore(ABC):
    """Storage for agent memory entries and session history.

    Retrieval strategy is a separate concern — see :class:`Retriever`.
    Backends where storage and retrieval are inseparable (a managed
    service, for instance) can implement both protocols on one class.
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str: ...

    @abstractmethod
    async def get(self, id: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def all(self) -> list[MemoryEntry]: ...

    @abstractmethod
    async def delete(self, id: str) -> None: ...

    # Session history. Pragmatically lives on the store since sessions
    # and memory entries share a backing resource in every concrete
    # implementation we have.
    @abstractmethod
    async def get_session(self, session_id: str) -> list[Message]: ...

    @abstractmethod
    async def add_to_session(self, session_id: str, message: Message) -> None: ...

    @abstractmethod
    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None: ...


class Retriever(ABC):
    """Swappable retrieval strategy — the axis sweeps iterate over.

    Dense vectors, BM25, hybrid, reranking — all slot in here. Can be
    composed (``RerankingRetriever(HybridRetriever(...))``) or wrap a
    managed service that owns both storage and retrieval. Every call
    produces a :attr:`SpanKind.MEMORY_QUERY` span in the trace with
    the query and retrieved-id scores.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]: ...


class FeedbackLoop(ABC):
    @abstractmethod
    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist an agent result and return its ID.

        The returned ID is the key ``score`` writes against. Metadata
        keys that ``FeedbackQuery`` filters on — ``agent_name``,
        ``model``, ``tags`` — should be populated here so later queries
        can match them.
        """
        ...

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
    async def query(self, q: FeedbackQuery) -> list[ScoredResult]:
        """Find prior results matching a :class:`FeedbackQuery`.

        Combines embedding similarity (when ``similar_to`` is set) with
        metadata filters. Returns up to ``q.limit`` results ranked by
        similarity (or recency when no similarity query is given).
        """
        ...

    @abstractmethod
    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]: ...


class Grader[T](ABC):
    """Score an agent output along one or more dimensions.

    Parameterized over ``T``: the type of ``output`` passed to :meth:`grade`.
    When an agent declares ``output_schema=MyModel``, ``T`` is ``MyModel`` and
    graders receive a validated pydantic instance. Without a schema, ``T`` is
    ``str`` (the raw final content). Graders that want the raw string even
    when a schema is set can read it from ``context["raw_output"]``.
    """

    @abstractmethod
    async def grade(
        self,
        input: Any,
        output: T,
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

    async def flush(self) -> None:
        """Flush any buffered spans to the backing store.

        Default is a no-op. Backends that buffer writes (notably
        :class:`SQLiteTracer`) should override this. Safe for callers to
        invoke on any tracer — ``run_agent`` always calls it at the end
        of a run.
        """
        return None


class Tool(ABC):
    # Route execution through ``jig.dispatch.run`` instead of calling
    # ``execute()`` locally. Subclasses override to True for work that
    # belongs on a smithers worker (backtests, embeddings, reindexes).
    # :attr:`dispatch_fn_ref` must be set when this is True.
    dispatch: bool = False

    @property
    def dispatch_fn_ref(self) -> str | None:
        """Entry-point identifier the smithers worker resolves.

        Format: ``"package.module:function"``. The worker looks this up
        via the ``jig.smithers_fn`` entry-point group. Only required
        (and only used) when :attr:`dispatch` is True.
        """
        return None

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str: ...
