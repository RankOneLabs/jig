"""
============================================================
jig — Rank One Labs Agent Framework
Five interfaces. One run function. Everything swappable.
============================================================

A jig holds the pieces while you work on them.
It doesn't have opinions about your pieces — just the shapes
of the plugs between them.

Language: Python 3.12+
Package manager: uv
"""

# ============================================================
# DEPENDENCIES
# ============================================================
#
# Core (always installed):
#   httpx           — async HTTP client for all LLM adapters
#   aiosqlite       — async SQLite for tracing, feedback, local memory
#   pydantic        — data validation for all types below
#   numpy           — cosine similarity for local memory search
#
# LLM adapters (install per-provider):
#   anthropic       — Anthropic SDK (Messages API)
#   openai          — OpenAI SDK (Chat Completions API)
#   ollama          — Ollama Python client (local models via homelab)
#
# Memory adapters (install per-backend):
#   honcho          — Honcho SDK (sessions + metamessages)
#   zep-python      — Zep SDK (memory + search)
#
# Embeddings (for local memory):
#   ollama          — reuse for embeddings (nomic-embed-text via RTX 5000 Pro)
#
# Eval (development):
#   promptfoo       — installed globally via npm, not a Python dep
#                     agents exposed as promptfoo Python providers
#
# pyproject.toml dependencies section:
#
# [project]
# dependencies = [
#     "httpx>=0.27",
#     "aiosqlite>=0.20",
#     "pydantic>=2.0",
#     "numpy>=1.26",
# ]
#
# [project.optional-dependencies]
# anthropic = ["anthropic>=0.40"]
# openai = ["openai>=1.50"]
# ollama = ["ollama>=0.4"]
# honcho = ["honcho>=0.1"]
# zep = ["zep-python>=2.0"]
# all = ["jig[anthropic,openai,ollama,honcho,zep]"]
# ============================================================


from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable
import uuid
import time


# ============================================================
# CORE TYPES
# ============================================================

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
    cost: float | None = None  # USD, if calculable


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: Usage
    latency_ms: float
    model: str


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None  # relevance score from query
    created_at: datetime = field(default_factory=datetime.now)


class ScoreSource(str, Enum):
    LLM_JUDGE = "llm_judge"
    HEURISTIC = "heuristic"
    HUMAN = "human"
    GROUND_TRUTH = "ground_truth"


@dataclass
class Score:
    dimension: str       # e.g. "relevance", "signal_to_noise", "actionable"
    value: float         # 0.0 - 1.0
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
    """Promptfoo-compatible eval case for export."""
    input: str
    expected: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SpanKind(str, Enum):
    AGENT_RUN = "agent_run"
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
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class CompletionParams:
    """Universal params every adapter understands (typed core)
    plus an escape hatch for provider-specific knobs."""
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    # Escape hatch — passed through to the underlying SDK
    # e.g. {"top_k": 40} for Anthropic, {"response_format": {...}} for OpenAI,
    #      {"num_ctx": 8192} for Ollama
    provider_params: dict[str, Any] | None = None


# ============================================================
# 1. LLMClient
# ============================================================
# Thin wrapper. Swap providers in config, never in agent code.
# Each adapter receives native API responses and returns LLMResponse.

class LLMClient(ABC):
    """
    Contract:
      complete() takes CompletionParams (typed core + provider escape hatch)
      Returns jig LLMResponse with normalized usage, timing, and tool calls
      All provider-specific shapes are translated inside the adapter

      Adapters MUST use typed fields from CompletionParams.
      Adapters SHOULD forward provider_params to the underlying SDK call.
      Adapters MUST silently ignore unknown keys in provider_params.
    """

    @abstractmethod
    async def complete(self, params: CompletionParams) -> LLMResponse:
        ...

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        raise NotImplementedError("Streaming not implemented for this provider")
        yield  # make it a generator


# ---- Anthropic Adapter ----
# SDK: anthropic
#
# RECEIVES from Anthropic Messages API:
#   response.content          -> list[ContentBlock] where each is TextBlock or ToolUseBlock
#   response.usage            -> { "input_tokens": int, "output_tokens": int }
#   response.model            -> str
#   ToolUseBlock.id           -> str
#   ToolUseBlock.name         -> str
#   ToolUseBlock.input        -> dict
#
# TRANSLATES TO jig LLMResponse:
#   content      <- join all TextBlock.text with newlines
#   tool_calls   <- [ToolCall(id=block.id, name=block.name, arguments=block.input) for ToolUseBlock]
#   usage        <- Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
#                         cost=compute_cost(model, input_tokens, output_tokens))
#   latency_ms   <- wall clock time of the API call
#   model        <- response.model
#
# SENDS to Anthropic (from CompletionParams):
#   params.system                   -> system parameter (string, not in messages list)
#   params.messages (role=USER)     -> {"role": "user", "content": str}
#   params.messages (role=ASSISTANT)-> {"role": "assistant", "content": [TextBlock] + [ToolUseBlock...]}
#   params.messages (role=TOOL)     -> {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": str}]}
#   params.tools                    -> {"name": str, "description": str, "input_schema": parameters}
#   params.temperature              -> temperature parameter
#   params.max_tokens               -> max_tokens parameter
#   params.provider_params          -> forwarded as **kwargs (e.g. top_k, top_p, metadata)
#
# ERROR HANDLING:
#   anthropic.RateLimitError   -> retry with exponential backoff (max 3 attempts)
#   anthropic.APIError         -> wrap in JigLLMError with original status code
#   anthropic.AuthenticationError -> raise immediately, don't retry


# ---- OpenAI Adapter ----
# SDK: openai
#
# RECEIVES from OpenAI Chat Completions API:
#   response.choices[0].message.content     -> str | None
#   response.choices[0].message.tool_calls  -> list[ChatCompletionMessageToolCall] | None
#   ChatCompletionMessageToolCall.id        -> str
#   ChatCompletionMessageToolCall.function.name       -> str
#   ChatCompletionMessageToolCall.function.arguments  -> str (JSON string, must parse)
#   response.usage.prompt_tokens            -> int
#   response.usage.completion_tokens        -> int
#   response.model                          -> str
#
# TRANSLATES TO jig LLMResponse:
#   content      <- response.choices[0].message.content or ""
#   tool_calls   <- [ToolCall(id=tc.id, name=tc.function.name,
#                     arguments=json.loads(tc.function.arguments)) for tc in tool_calls]
#   usage        <- Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
#                         cost=compute_cost(model, prompt_tokens, completion_tokens))
#   latency_ms   <- wall clock time
#   model        <- response.model
#
# SENDS to OpenAI (from CompletionParams):
#   params.system                   -> {"role": "system", "content": str} prepended to messages
#   params.messages (role=USER)     -> {"role": "user", "content": str}
#   params.messages (role=ASSISTANT)-> {"role": "assistant", "content": str, "tool_calls": [...]}
#   params.messages (role=TOOL)     -> {"role": "tool", "tool_call_id": id, "content": str}
#   params.tools                    -> {"type": "function", "function": {"name": str, "description": str, "parameters": parameters}}
#   params.temperature              -> temperature parameter
#   params.max_tokens               -> max_tokens parameter
#   params.provider_params          -> forwarded as **kwargs (e.g. response_format, seed, logprobs)
#
# ERROR HANDLING:
#   openai.RateLimitError      -> retry with exponential backoff (max 3 attempts)
#   openai.APIError            -> wrap in JigLLMError
#   openai.AuthenticationError -> raise immediately


# ---- Ollama Adapter ----
# SDK: ollama
#
# RECEIVES from Ollama chat API:
#   response["message"]["content"]          -> str
#   response["message"]["tool_calls"]       -> list[dict] | None
#   tool_call["function"]["name"]           -> str
#   tool_call["function"]["arguments"]      -> dict (already parsed, not JSON string)
#   response.get("eval_count", 0)           -> int (output tokens)
#   response.get("prompt_eval_count", 0)    -> int (input tokens)
#
# TRANSLATES TO jig LLMResponse:
#   content      <- response["message"]["content"]
#   tool_calls   <- [ToolCall(id=generate_uuid(), name=tc["function"]["name"],
#                     arguments=tc["function"]["arguments"]) for tc in tool_calls]
#                   NOTE: Ollama does not provide tool call IDs — generate them
#   usage        <- Usage(input_tokens=prompt_eval_count, output_tokens=eval_count, cost=None)
#                   NOTE: cost is always None for local models
#   latency_ms   <- wall clock time
#   model        <- config model string
#
# SENDS to Ollama (from CompletionParams):
#   params.system                   -> {"role": "system", "content": str} prepended to messages
#   params.messages (role=USER)     -> {"role": "user", "content": str}
#   params.messages (role=ASSISTANT)-> {"role": "assistant", "content": str}
#   params.messages (role=TOOL)     -> {"role": "tool", "content": str}
#   params.tools                    -> {"type": "function", "function": {"name": str, "description": str, "parameters": parameters}}
#   params.temperature              -> temperature parameter (options.temperature)
#   params.max_tokens               -> num_predict option
#   params.provider_params          -> forwarded as **kwargs (e.g. num_ctx, top_k, repeat_penalty)
#
# ERROR HANDLING:
#   ConnectionError            -> retry with backoff (homelab might be waking up)
#   ollama.ResponseError       -> wrap in JigLLMError


# ============================================================
# 2. AgentMemory
# ============================================================
# Backend-agnostic memory. Each adapter receives native API
# responses and returns MemoryEntry[].

class AgentMemory(ABC):
    """
    Contract:
      add() stores content, returns jig memory ID (string)
      query() returns MemoryEntry[] sorted by relevance score descending
      getSession()/addToSession() manage conversational history as Message[]
      All backend-specific shapes are translated inside the adapter
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        ...

    @abstractmethod
    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> list[Message]:
        ...

    @abstractmethod
    async def add_to_session(self, session_id: str, message: Message) -> None:
        ...

    @abstractmethod
    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        ...


# ---- Local Memory Adapter (SQLite + Ollama Embeddings) ----
# Dependencies: aiosqlite, ollama, numpy
#
# EMBEDDING MODEL: nomic-embed-text via Ollama on homelab (RTX 5000 Pro)
#   Dimensions: 768
#   Call: ollama.embed(model="nomic-embed-text", input=text) -> {"embeddings": [[float, ...]]}
#
# STORAGE: Two SQLite tables
#   memories(id TEXT PK, content TEXT, metadata JSON, embedding BLOB, created_at TEXT)
#   sessions(session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls JSON, created_at TEXT)
#
# add():
#   1. Generate embedding via Ollama: ollama.embed(model, content) -> float[768]
#   2. Serialize embedding as numpy bytes
#   3. INSERT into memories table
#   4. Return generated UUID
#
# query():
#   1. Generate embedding for query string via Ollama
#   2. SELECT all embeddings from memories (with optional metadata filter)
#   3. Compute cosine similarity: numpy.dot(query_emb, row_emb) / (norm(q) * norm(r))
#   4. Sort by similarity descending, take top `limit`
#   5. Return as MemoryEntry[] with score = cosine similarity
#
# get_session():
#   SELECT from sessions WHERE session_id = ? ORDER BY created_at ASC
#   Return as Message[]
#
# add_to_session():
#   INSERT into sessions table
#
# NOTE: For small-to-medium memory stores (< 100k entries), brute-force cosine
# similarity is fine. If this becomes a bottleneck, add sqlite-vss or hnswlib.


# ---- Honcho Adapter ----
# SDK: honcho
#
# RECEIVES from Honcho API:
#   Session object:
#     session.id              -> str (UUID)
#     session.metadata        -> dict
#   Message/Metamessage objects:
#     message.content         -> str
#     message.is_user         -> bool
#     metamessage.content     -> str
#     metamessage.metadata    -> dict
#   collection.query():
#     results                 -> list[Document] where Document has .content, .metadata, .distance
#
# TRANSLATES TO jig types:
#   add():
#     Creates a Honcho metamessage or document in a collection
#     Returns Honcho document ID as jig memory ID string
#   query():
#     Calls collection.query(query_text, top_k=limit)
#     Document.content    -> MemoryEntry.content
#     Document.metadata   -> MemoryEntry.metadata
#     1.0 - Document.distance -> MemoryEntry.score (Honcho uses distance, jig uses similarity)
#     Document.id         -> MemoryEntry.id
#   get_session():
#     Calls honcho.apps.users.sessions.messages.list(app_id, user_id, session_id)
#     message.is_user=True  -> Message(role=USER)
#     message.is_user=False -> Message(role=ASSISTANT)
#   add_to_session():
#     Calls honcho.apps.users.sessions.messages.create(...)
#     Message.role == USER -> is_user=True
#     Message.role == ASSISTANT -> is_user=False
#
# ERROR HANDLING:
#   honcho.AuthenticationError -> raise immediately
#   honcho.NotFoundError       -> return empty list (session may not exist yet)


# ---- Zep Adapter ----
# SDK: zep-python
#
# RECEIVES from Zep API:
#   Memory search:
#     SearchResult.message    -> dict with "content", "role"
#     SearchResult.score      -> float (relevance, higher = better)
#     SearchResult.metadata   -> dict
#   Session messages:
#     Message.content         -> str
#     Message.role            -> str ("human", "ai", "tool")
#     Message.metadata        -> dict
#
# TRANSLATES TO jig types:
#   add():
#     Calls zep.memory.add(session_id, messages=[...])
#     Returns generated UUID as jig memory ID
#   query():
#     Calls zep.memory.search(session_id, query, limit=limit)
#     SearchResult.message["content"] -> MemoryEntry.content
#     SearchResult.score              -> MemoryEntry.score (no inversion needed)
#     SearchResult.metadata           -> MemoryEntry.metadata
#   get_session():
#     Calls zep.memory.get(session_id)
#     Zep "human"  -> Message(role=USER)
#     Zep "ai"     -> Message(role=ASSISTANT)
#     Zep "tool"   -> Message(role=TOOL)
#   add_to_session():
#     Calls zep.memory.add(session_id, messages=[...])
#     jig USER      -> Zep role="human"
#     jig ASSISTANT  -> Zep role="ai"
#
# ERROR HANDLING:
#   zep.NotFoundError -> create session on first access, then retry


# ============================================================
# 3. FeedbackLoop
# ============================================================
# Two speeds: fast (per-request context enrichment) and
# slow (batch export to promptfoo eval sets).

class FeedbackLoop(ABC):
    """
    Contract:
      score() stores Score[] against a result ID
      get_signals() returns historical quality signals for similar queries
      export_eval_set() produces promptfoo-compatible EvalCase[] for batch eval
    """

    @abstractmethod
    async def score(self, result_id: str, scores: list[Score]) -> None:
        ...

    @abstractmethod
    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        ...

    @abstractmethod
    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        ...


# ---- SQLite FeedbackLoop Implementation ----
# Dependencies: aiosqlite, numpy (reuses local memory embedding for similarity)
#
# STORAGE: Two SQLite tables
#   results(id TEXT PK, content TEXT, input TEXT, metadata JSON, embedding BLOB, created_at TEXT)
#   scores(result_id TEXT FK, dimension TEXT, value REAL, source TEXT, created_at TEXT)
#
# score():
#   INSERT into scores for each Score in the list
#
# get_signals():
#   1. Embed query via Ollama (same model as local memory)
#   2. SELECT results + their scores
#   3. Cosine similarity to find relevant past results
#   4. Filter by min_score and source if specified
#   5. Return as ScoredResult[] sorted by similarity
#
# export_eval_set():
#   SELECT results + scores with filters (since, min_score, max_score)
#   Transform each into EvalCase:
#     result.input   -> EvalCase.input
#     result.content -> EvalCase.expected (the output that got scored)
#     result.metadata + avg score -> EvalCase.metadata
#   Output format is directly consumable by promptfoo YAML test loader


# ============================================================
# 4. Graders
# ============================================================
# Produce Score[] from (input, output) pairs. Swappable per agent.

class Grader(ABC):
    """
    Contract:
      grade() receives the agent input and output (both strings)
      Returns Score[] — one or more dimensions rated 0.0 to 1.0
      Each Score includes its source so the feedback loop knows
      how much to trust it
    """

    @abstractmethod
    async def grade(
        self,
        input: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        ...


# ---- LLM Judge Grader ----
# Uses a separate LLM call to evaluate the output.
#
# SENDS to LLM (via jig LLMClient):
#   System prompt defining grading rubric and dimensions
#   User message with the input/output pair to evaluate
#   Requests JSON response: {"scores": [{"dimension": str, "value": float, "reasoning": str}]}
#
# RECEIVES from LLM:
#   JSON with scores array
#
# TRANSLATES TO jig Score[]:
#   Each item -> Score(dimension=item["dimension"], value=item["value"], source=ScoreSource.LLM_JUDGE)
#
# CONFIG:
#   llm: LLMClient        — which model to use as judge (can differ from agent's model)
#   dimensions: list[str]  — what to grade on, e.g. ["relevance", "completeness", "accuracy"]
#   rubric: str            — grading instructions appended to system prompt


# ---- Heuristic Grader ----
# Pattern matching and simple checks. Free, instant.
#
# CONFIG:
#   checks: list of {name: str, pattern: regex | callable, weight: float}
#
# grade():
#   For each check:
#     If pattern is regex: score = 1.0 if match found else 0.0
#     If pattern is callable: score = callable(input, output) returning 0.0-1.0
#   Return Score(dimension=check.name, value=score, source=ScoreSource.HEURISTIC)


# ---- Ground Truth Grader ----
# Compares output against known-correct answers. Best for trading bot.
#
# CONFIG:
#   comparator: Callable[[str, str], float]  — takes (output, expected) returns similarity 0.0-1.0
#
# grade():
#   Requires context["expected"] to be set
#   Calls comparator(output, context["expected"])
#   Returns Score(dimension="correctness", value=result, source=ScoreSource.GROUND_TRUTH)


# ---- Composite Grader ----
# Runs multiple graders and merges their Score lists.
#
# CONFIG:
#   graders: list[Grader]
#
# grade():
#   results = await asyncio.gather(*[g.grade(input, output, context) for g in graders])
#   Return flattened list of all Score objects


# ============================================================
# 5. TracingLogger
# ============================================================
# Structured logging with parent-child spans.

class TracingLogger(ABC):
    """
    Contract:
      start_trace() creates a root span, returns Span with generated id and trace_id
      start_span() creates a child span under a parent
      end_span() closes a span with output (or error)
      Spans are queryable by trace_id for debugging and dashboard
    """

    @abstractmethod
    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> Span:
        ...

    @abstractmethod
    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None) -> Span:
        ...

    @abstractmethod
    def end_span(self, span_id: str, output: Any = None, error: str | None = None) -> None:
        ...

    @abstractmethod
    async def get_trace(self, trace_id: str) -> list[Span]:
        ...

    @abstractmethod
    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        ...


# ---- SQLite Tracer ----
# Dependencies: aiosqlite
#
# STORAGE: One table
#   spans(id TEXT PK, trace_id TEXT, parent_id TEXT, kind TEXT, name TEXT,
#         input JSON, output JSON, started_at TEXT, ended_at TEXT,
#         duration_ms REAL, metadata JSON, error TEXT,
#         usage_input_tokens INT, usage_output_tokens INT, usage_cost REAL)
#
# start_trace():
#   Generate trace_id and span_id (both UUIDs)
#   INSERT span with kind=AGENT_RUN, parent_id=NULL
#   Return Span object
#
# start_span():
#   Generate span_id, inherit trace_id from parent
#   INSERT span with parent_id set
#   Return Span object
#
# end_span():
#   UPDATE span SET ended_at, duration_ms, output, error WHERE id = span_id
#
# get_trace():
#   SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC
#
# list_traces():
#   SELECT * FROM spans WHERE kind = 'agent_run' (root spans only)
#   Filter by since, name. ORDER BY started_at DESC LIMIT limit


# ---- Stdout Tracer ----
# Pretty-prints spans to console. For development/debugging.
#
# start_trace(): print "[TRACE] {name} started" with color
# start_span(): print "  [{kind}] {name}" indented by depth
# end_span(): print "  [{kind}] {name} completed in {duration_ms}ms"
#              if error: print in red
#              if usage: print token counts and cost
# get_trace()/list_traces(): not supported, raise NotImplementedError


# ============================================================
# 6. ToolRegistry
# ============================================================

class Tool(ABC):
    """
    Contract:
      definition provides the JSON Schema that gets sent to the LLM
      execute() takes parsed arguments and returns a string result
      On error, return ToolResult with error set — don't raise
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str:
        ...


class ToolRegistry:
    """
    Registry holds tools by name. execute() resolves and calls.
    If a tool is not found, returns ToolResult with error.
    If a tool raises, catches and returns ToolResult with error.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in (tools or []):
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(call_id=call.id, output="", error=f"Unknown tool: {call.name}")
        try:
            output = await tool.execute(call.arguments)
            return ToolResult(call_id=call.id, output=output)
        except Exception as e:
            return ToolResult(call_id=call.id, output="", error=str(e))


# ============================================================
# 7. Agent Config + Runner
# ============================================================

@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str | Callable[[], str | Awaitable[str]]

    # Wiring — all injected, all swappable
    llm: LLMClient
    memory: AgentMemory
    feedback: FeedbackLoop
    tracer: TracingLogger
    tools: ToolRegistry

    # Optional grader for auto-scoring outputs
    grader: Grader | None = None

    # Behavior
    max_tool_calls: int = 10
    include_memory_in_prompt: bool = True
    include_feedback_in_prompt: bool = True
    session_id: str | None = None


@dataclass
class AgentResult:
    output: str
    trace_id: str
    usage: dict  # {total_input_tokens, total_output_tokens, total_cost, llm_calls, tool_calls}
    scores: list[Score] | None
    duration_ms: float


async def run_agent(config: AgentConfig, input: str) -> AgentResult:
    """
    The entire execution model in one function.

    Flow:
      1. Start trace
      2. Resolve system prompt (static string or callable)
      3. Query memory for relevant context
      4. Query feedback for quality signals from past similar runs
      5. Assemble messages: system (with memory + signals injected) + session history + user input
      6. LLM call + tool execution loop (max_tool_calls enforced)
      7. Store output in memory
      8. Auto-grade if grader configured, store scores in feedback
      9. Close trace, return AgentResult
    """
    start = time.time()

    # 1. Start trace
    trace = config.tracer.start_trace(config.name, {"input": input})

    # 2. Resolve system prompt
    if callable(config.system_prompt):
        result = config.system_prompt()
        system_prompt = await result if hasattr(result, "__await__") else result
    else:
        system_prompt = config.system_prompt

    # 3. Query memory
    memory_context: list[MemoryEntry] = []
    if config.include_memory_in_prompt:
        mem_span = config.tracer.start_span(trace.id, SpanKind.MEMORY_QUERY, "query_memory", {"query": input})
        memory_context = await config.memory.query(input, limit=5, session_id=config.session_id)
        config.tracer.end_span(mem_span.id, [e.content for e in memory_context])

    # 4. Query feedback signals
    feedback_signals: list[ScoredResult] = []
    if config.include_feedback_in_prompt:
        fb_span = config.tracer.start_span(trace.id, SpanKind.MEMORY_QUERY, "query_feedback", {"query": input})
        feedback_signals = await config.feedback.get_signals(input, limit=3, min_score=0.7)
        config.tracer.end_span(fb_span.id, [s.content[:100] for s in feedback_signals])

    # 5. Assemble messages (system prompt is separate, not in messages list)
    system_message = _build_system_message(system_prompt, memory_context, feedback_signals)
    messages: list[Message] = []
    if config.session_id:
        history = await config.memory.get_session(config.session_id)
        messages.extend(history)
    messages.append(Message(role=Role.USER, content=input))

    # 6. LLM call + tool loop
    total_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
    }
    tool_call_count = 0
    final_output = ""

    while True:
        llm_span = config.tracer.start_span(trace.id, SpanKind.LLM_CALL, "completion")
        params = CompletionParams(
            messages=messages,
            system=system_message,
            tools=config.tools.list() or None,
        )
        response = await config.llm.complete(params)
        config.tracer.end_span(llm_span.id, {"content": response.content[:200], "tool_calls": len(response.tool_calls or [])})

        total_usage["total_input_tokens"] += response.usage.input_tokens
        total_usage["total_output_tokens"] += response.usage.output_tokens
        total_usage["total_cost"] += response.usage.cost or 0.0
        total_usage["llm_calls"] += 1

        # No tool calls — done
        if not response.tool_calls:
            final_output = response.content
            break

        # Execute tool calls
        messages.append(Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        for call in response.tool_calls:
            if tool_call_count >= config.max_tool_calls:
                messages.append(Message(
                    role=Role.TOOL,
                    content="Max tool calls reached. Provide final answer.",
                    tool_call_id=call.id,
                ))
                break

            tool_span = config.tracer.start_span(trace.id, SpanKind.TOOL_CALL, call.name, call.arguments)
            result = await config.tools.execute(call)
            config.tracer.end_span(tool_span.id, result.output[:500], error=result.error)

            messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=call.id))
            tool_call_count += 1
            total_usage["tool_calls"] += 1

    # 7. Store in memory
    result_id = await config.memory.add(final_output, {
        "agent": config.name,
        "input": input,
        "trace_id": trace.trace_id,
    })

    if config.session_id:
        await config.memory.add_to_session(config.session_id, Message(role=Role.USER, content=input))
        await config.memory.add_to_session(config.session_id, Message(role=Role.ASSISTANT, content=final_output))

    # 8. Auto-grade
    scores: list[Score] | None = None
    if config.grader:
        grade_span = config.tracer.start_span(trace.id, SpanKind.GRADING, "auto_grade", {"input": input})
        scores = await config.grader.grade(input, final_output)
        await config.feedback.score(result_id, scores)
        config.tracer.end_span(grade_span.id, [{"dim": s.dimension, "val": s.value} for s in scores])

    # 9. Close trace
    duration = (time.time() - start) * 1000
    config.tracer.end_span(trace.id, {"output": final_output[:200], "scores": scores})

    return AgentResult(
        output=final_output,
        trace_id=trace.trace_id,
        usage=total_usage,
        scores=scores,
        duration_ms=duration,
    )


def _build_system_message(
    system_prompt: str,
    memory: list[MemoryEntry],
    signals: list[ScoredResult],
) -> str:
    """Assemble system prompt with memory context and feedback signals."""
    prompt = system_prompt

    if memory:
        prompt += "\n\n## Relevant context from memory\n"
        for entry in memory:
            prompt += f"- {entry.content}\n"

    if signals:
        prompt += "\n\n## Quality signals from past similar queries\n"
        for signal in signals:
            score_str = ", ".join(f"{s.dimension}: {s.value:.2f}" for s in signal.scores)
            prompt += f"- [{score_str}] {signal.content[:200]}\n"

    return prompt


# ============================================================
# ERRORS
# ============================================================

class JigError(Exception):
    """Base error for all jig errors."""
    pass

class JigLLMError(JigError):
    """LLM adapter error. Wraps provider-specific errors."""
    def __init__(self, message: str, provider: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

class JigMemoryError(JigError):
    """Memory adapter error."""
    pass

class JigToolError(JigError):
    """Tool execution error."""
    pass


# ============================================================
# FILE LAYOUT
# ============================================================
#
# jig/
# ├── src/
# │   └── jig/
# │       ├── __init__.py           # Public API exports
# │       ├── core/
# │       │   ├── __init__.py
# │       │   ├── types.py          # All dataclasses, enums, ABCs above
# │       │   ├── runner.py         # run_agent function
# │       │   ├── prompt.py         # _build_system_message
# │       │   └── errors.py         # JigError hierarchy
# │       ├── llm/
# │       │   ├── __init__.py
# │       │   ├── anthropic.py      # Anthropic adapter
# │       │   ├── openai.py         # OpenAI adapter
# │       │   └── ollama.py         # Ollama adapter (homelab)
# │       ├── memory/
# │       │   ├── __init__.py
# │       │   ├── local.py          # SQLite + Ollama embeddings
# │       │   ├── honcho.py         # Honcho adapter
# │       │   └── zep.py            # Zep adapter
# │       ├── feedback/
# │       │   ├── __init__.py
# │       │   ├── loop.py           # SQLite-backed FeedbackLoop
# │       │   ├── llm_judge.py      # LLM-as-judge grader
# │       │   ├── heuristic.py      # Pattern-based grader
# │       │   ├── ground_truth.py   # Known-answer grader (trading)
# │       │   └── composite.py      # Merges multiple graders
# │       ├── tracing/
# │       │   ├── __init__.py
# │       │   ├── sqlite.py         # SQLite tracer
# │       │   └── stdout.py         # Console tracer (dev)
# │       └── tools/
# │           ├── __init__.py
# │           ├── registry.py       # ToolRegistry (concrete, not abstract)
# │           └── common/           # Reusable tools (web search, etc)
# ├── agents/
# │   ├── scout.py                  # Scout config + tools
# │   └── trading.py                # Trading bot config + tools
# ├── evals/
# │   ├── scout.yaml                # Promptfoo eval config for Scout
# │   └── trading.yaml              # Promptfoo eval config for trading
# ├── pyproject.toml
# └── README.md
