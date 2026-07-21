from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Literal


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
    # Optional declaration of argument paths (dot-separated for nested
    # dicts) that jointly identify "the same real-world entity" across
    # calls to this tool, for replay alignment. ``None`` means the tool
    # author has not declared an identity; see jig.replay.align.
    identity_fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.identity_fields is None:
            return
        if not isinstance(self.identity_fields, list):
            raise ValueError(
                f"ToolDefinition.identity_fields must be a list[str] or "
                f"None, got {self.identity_fields!r}"
            )
        if len(self.identity_fields) == 0:
            raise ValueError(
                "ToolDefinition.identity_fields must not be an empty list "
                "(omit it, or pass None, to declare no identity)"
            )
        seen: set[str] = set()
        for path in self.identity_fields:
            if not isinstance(path, str):
                raise ValueError(
                    f"ToolDefinition.identity_fields entries must be str, "
                    f"got {path!r}"
                )
            if path == "":
                raise ValueError(
                    "ToolDefinition.identity_fields entries must not be "
                    "empty strings"
                )
            if any(segment == "" for segment in path.split(".")):
                raise ValueError(
                    f"ToolDefinition.identity_fields entry {path!r} has an "
                    f"empty dot segment"
                )
            if path in seen:
                raise ValueError(
                    f"ToolDefinition.identity_fields entry {path!r} is "
                    f"repeated"
                )
            seen.add(path)


@dataclass
class CompletionParams:
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    provider_params: dict[str, Any] | None = None
    # Portable OpenAI-compatible structured-output contract:
    # {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}.
    # Adapters forward it unchanged when they speak the OpenAI-compatible
    # request shape, translate it when the backend needs a different form
    # (Ollama), or reject it before making a request when unsupported.
    # Never normalized or mutated here — interpretation belongs to adapters.
    response_format: dict[str, Any] | None = None


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
    metadata: dict[str, Any] | None = None


@dataclass
class EffectiveScore:
    """The single score a consumer should trust for one (result, dimension).

    Resolved from the append-only ``scores`` history: the newest ``human``
    row if any exists, else the newest ``heuristic`` row, else absent.
    Carries its own provenance (``source``, ``created_at``, ``metadata``)
    separately from the full history so callers can see *why* this row
    won — the history itself is never mutated to produce this.
    """

    dimension: str
    value: float
    source: ScoreSource
    created_at: datetime
    metadata: dict[str, Any] | None = None


@dataclass
class EffectiveScoreFilter:
    """An inclusive [min_value, max_value] gate on one dimension's effective score.

    Both bounds are optional and inclusive when set. A result missing an
    effective score for ``dimension`` fails the filter — absence is never
    treated as satisfying a quality requirement.
    """

    dimension: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True, kw_only=True)
class HumanFeedbackPromptConfig:
    """Opt-in policy for injecting human-graded exemplars into agent prompts.

    Separate from the legacy ``AgentConfig.include_feedback_in_prompt`` /
    ``feedback.get_signals`` path — that path can inject any source
    (heuristic, LLM-judge, human) above a single ``min_score`` and carries
    no safety gate. This path only ever surfaces *human*-sourced effective
    scores, split into labeled positive/negative sections, gated by
    ``eligibility_filters`` before a result can enter either section.

    Disabled by default (``enabled=False``): every field below is inert
    until a caller opts in, so existing agents see no behavior change.
    """

    enabled: bool = False
    # Dimensions considered for positive/negative classification. Required
    # non-empty when enabled — there is no "all dimensions" wildcard, so a
    # caller must name exactly which rubric axes it wants surfaced.
    dimensions: tuple[str, ...] = ()
    # Inclusive thresholds: an effective *human* score >= positive_threshold
    # qualifies its dimension as a positive signal; <= negative_threshold
    # qualifies it as negative. Values strictly between are not injected.
    positive_threshold: float = 0.75
    negative_threshold: float = 0.25
    # Per-section cap on the number of examples rendered.
    positive_limit: int = 2
    negative_limit: int = 2
    # Combined character budget for both sections' rendered example bodies.
    total_character_budget: int = 6000
    # AND-combined gate on effective scores, applied before similarity
    # ranking and before positive/negative classification. A result that
    # fails this can never enter either section, regardless of how it
    # would otherwise classify — this is the safe-exemplar guard (e.g. a
    # plausibility floor) and is not itself a classification dimension.
    eligibility_filters: tuple[EffectiveScoreFilter, ...] = ()

    def __post_init__(self) -> None:
        if self.enabled and not self.dimensions:
            raise ValueError(
                "HumanFeedbackPromptConfig.dimensions must be non-empty when enabled=True"
            )
        for name, value in (
            ("positive_threshold", self.positive_threshold),
            ("negative_threshold", self.negative_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ValueError(f"HumanFeedbackPromptConfig.{name} must be in [0.0, 1.0], got {value!r}")
        if self.positive_threshold < self.negative_threshold:
            raise ValueError(
                "HumanFeedbackPromptConfig.positive_threshold must be >= negative_threshold "
                f"(got positive_threshold={self.positive_threshold}, "
                f"negative_threshold={self.negative_threshold})"
            )
        for name, value in (
            ("positive_limit", self.positive_limit),
            ("negative_limit", self.negative_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"HumanFeedbackPromptConfig.{name} must be a non-negative int, got {value!r}")
        if (
            isinstance(self.total_character_budget, bool)
            or not isinstance(self.total_character_budget, int)
            or self.total_character_budget < 0
        ):
            raise ValueError(
                "HumanFeedbackPromptConfig.total_character_budget must be a non-negative int, "
                f"got {self.total_character_budget!r}"
            )


@dataclass(frozen=True)
class HumanExampleDimension:
    """One threshold-crossing dimension that explains an example's classification."""

    dimension: str
    value: float
    note: str | None


@dataclass(frozen=True)
class HumanExample:
    """A task-similar, human-graded exemplar qualified for prompt injection."""

    result_id: str
    input_text: str
    output: str
    classification: Literal["positive", "negative"]
    # Every selected dimension that crossed the threshold matching
    # ``classification`` — never a mix of both polarities on one example.
    dimensions: list[HumanExampleDimension]


@dataclass(frozen=True)
class HumanExampleSet:
    """Result of :meth:`FeedbackLoop.get_human_examples` — already deduplicated,
    ranked, and capped to each section's configured limit."""

    positive: list[HumanExample]
    negative: list[HumanExample]


@dataclass
class ScoredResult:
    result_id: str
    content: str
    scores: list[Score]
    avg_score: float
    metadata: dict[str, Any]
    created_at: datetime
    # Populated only when the originating FeedbackQuery opted into effective
    # resolution (``resolve_effective=True`` or ``effective_filters`` is set);
    # keyed by dimension. None when not requested.
    effective_scores: dict[str, EffectiveScore] | None = None


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
    # When set, only scores from this source are considered per result;
    # results whose filtered score list is empty are dropped entirely.
    source: ScoreSource | None = None
    # Opt-in: when True, each returned ScoredResult.effective_scores is
    # populated (human-over-heuristic, newest-wins resolution per
    # dimension). Legacy callers that leave this False see no change in
    # shape or ranking. Implied True whenever effective_filters is set.
    resolve_effective: bool = False
    # Opt-in: inclusive per-dimension gates on the *effective* score,
    # combined with AND. Applied within the backend's bounded candidate
    # window before the final limit; aggressive filters can therefore return
    # fewer than ``limit`` matches. A result missing an effective score for a
    # named dimension fails that filter.
    effective_filters: list[EffectiveScoreFilter] | None = None

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


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Context available while Jig is executing a tool call."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    tool_call_id: str
    metadata: dict[str, Any] | None = None


current_tool_context: ContextVar[ToolExecutionContext | None] = ContextVar(
    "current_tool_context",
    default=None,
)


# --- Abstract interfaces ---


class LLMClient(ABC):
    # Whether this client can honor CompletionParams.response_format (the
    # OpenAI-compatible json_schema envelope) without silently dropping or
    # downgrading it. Checked by the runner's native structured-output mode
    # before the first completion call, so an incapable client fails fast
    # instead of burning an LLM round-trip. Adapters that forward or
    # translate response_format set this True; everything else keeps the
    # False default.
    supports_response_format: bool = False

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

    async def get_human_examples(
        self,
        task_input: str,
        config: HumanFeedbackPromptConfig,
    ) -> HumanExampleSet:
        """Task-similar human-graded positive/negative exemplars for prompts.

        Concrete method (not abstract) with a no-op default returning an
        empty set — backends that can't rank by embedding similarity (e.g.
        :class:`~jig.feedback.null.NullFeedbackLoop`) need no override, and
        callers that opt into :class:`HumanFeedbackPromptConfig` against
        such a backend degrade to "no examples" rather than erroring.
        Backends that support it (:class:`~jig.feedback.loop.SQLiteFeedbackLoop`)
        override this with real retrieval.
        """
        return HumanExampleSet(positive=[], negative=[])


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
    def start_span(
        self,
        parent_id: str,
        kind: SpanKind,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span: ...

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

    def dispatch_payload_extra(
        self,
        context: ToolExecutionContext | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extra payload fields to send to dispatched workers."""
        return {}

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str: ...

    async def execute_with_context(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> str:
        """Execute with optional Jig context.

        Subclasses can override this when they need span or tool-call
        metadata. The default preserves the original ``execute(args)``
        contract for existing tools.
        """
        return await self.execute(args)
