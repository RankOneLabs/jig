from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ValidationError

from jig.core.errors import (
    AgentAmbiguousTurnError,
    AgentError,
    AgentLLMPermanentError,
    AgentMaxLLMCallsError,
    AgentMaxLLMRetriesError,
    AgentNativeOutputError,
    AgentSchemaNotCalledError,
    AgentSchemaValidationError,
    JigLLMError,
    UnsupportedResponseFormatError,
)
from jig.core.grading import grade_and_record
from jig.core.prompt import build_human_feedback_section, build_system_message
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    Grader,
    HumanExampleSet,
    HumanFeedbackPromptConfig,
    LLMClient,
    MemoryEntry,
    MemoryStore,
    Message,
    Retriever,
    Role,
    Score,
    ScoreSource,
    ScoredResult,
    SpanKind,
    ToolCall,
    ToolExecutionContext,
    ToolDefinition,
    TracingLogger,
    current_tool_context,
)
from jig.tools.registry import ToolRegistry
from jig.tracing.spans import span_guard

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3

# Reserved tool name the runner injects when output_schema is set. The agent
# loop terminates when the model calls this tool; its arguments are validated
# against the schema to produce AgentResult.parsed.
SUBMIT_OUTPUT_TOOL = "submit_output"

# Reserved keys the runner writes into the AGENT_RUN root span's ``output``
# dict, in addition to the pre-existing ``output`` (200-char preview) and
# ``scores``. ``output_kind`` is written unconditionally on every finalized
# run and marks the trace as understanding complete-output capture — its
# absence (not merely a missing/false value) is how jig.replay.diff tells a
# trace recorded before this feature ("preview_only_output") apart from a
# modern trace that simply produced no structured value
# ("structured_output_unavailable"), e.g. a plain-text run or a schema run
# that never validated. The other three keys are only written together, and
# only when a validated structured value was produced.
ROOT_OUTPUT_KIND_KEY = "output_kind"
ROOT_OUTPUT_COMPLETE_KEY = "output_complete"
ROOT_OUTPUT_SHA256_KEY = "output_sha256"
ROOT_OUTPUT_BYTE_LENGTH_KEY = "output_byte_length"

# "legacy" injects the synthetic submit_output tool and validates its
# arguments (the long-standing behavior). "native" omits that tool, converts
# output_schema into a strict response_format, and parses the schema-
# constrained terminal assistant content directly. "native_two_phase" runs
# every working turn unconstrained (tools offered, no response_format); the
# first no-tool-call turn triggers one additional schema-constrained,
# tool-free finalize call whose content is the terminal result — the schema
# can never bias a turn on which tools are offered. No "auto": callers pick
# explicitly so benchmark attribution and production guarantees stay knowable.
StructuredOutputMode = Literal["legacy", "native", "native_two_phase"]

# HTTP status codes that indicate a definitely-permanent LLM failure.
# 400 / 422 — malformed request; 401 / 403 — auth problem; 404 — bad
# model or endpoint. Retrying these is pure budget burn — fail fast.
# Unknown errors (status_code=None) and transient server errors (5xx)
# continue to use the existing max_llm_retries retry path: we only
# want to short-circuit when we're confident the next call would fail
# identically.
_PERMANENT_LLM_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


@dataclass(frozen=True, kw_only=True)
class AgentConfig[T]:
    """Immutable agent configuration.

    Keyword-only construction. Frozen so configs can be shared across
    concurrent runs without aliasing bugs. Derive variants with
    :meth:`with_` rather than mutating — the generic ``T`` is preserved.
    """

    name: str
    description: str
    system_prompt: str | Callable[[], str | Awaitable[str]]

    llm: LLMClient
    feedback: FeedbackLoop
    tracer: TracingLogger
    tools: ToolRegistry

    # Memory split: store owns persistence + session history; retriever
    # owns the strategy for pulling context into the prompt. Both are
    # optional — agents that don't need memory leave them as None.
    store: MemoryStore | None = None
    retriever: Retriever | None = None

    grader: Grader[T] | None = None
    max_tool_calls: int = 10
    # Absolute cap on LLM calls per run. ``max_tool_calls`` caps tool
    # *execution*, but a model that keeps emitting tool_use blocks after
    # being told "max reached" would otherwise loop forever. This bounds
    # the outer loop so a misbehaving model can't burn unbounded spend.
    max_llm_calls: int = 50
    max_llm_retries: int = _MAX_LLM_RETRIES
    include_memory_in_prompt: bool = True
    include_feedback_in_prompt: bool = True
    session_id: str | None = None

    # --- Feedback signal query ---
    # Parameters for the ``feedback.get_signals`` call made when
    # ``include_feedback_in_prompt`` is set. Explicit fields make the
    # previously-hardcoded run_agent query parameters configurable per
    # config rather than fixed at limit=3, min_score=0.7.
    feedback_limit: int = 3
    feedback_min_score: float = 0.7
    feedback_source: ScoreSource | None = None

    # --- Human-reviewed feedback prompt injection (opt-in) ---
    # Independent of the legacy feedback_signals path above: when enabled,
    # queries feedback.get_human_examples for task-similar, human-only
    # positive/negative exemplars and appends them to the system message
    # under labeled, delimited headings. Disabled by default.
    human_feedback_prompt: HumanFeedbackPromptConfig = HumanFeedbackPromptConfig()

    # --- Structured output ---
    # Pydantic model the agent should produce. When set, the runner injects a
    # ``submit_output`` tool with the model's JSON schema; the loop ends when
    # the model calls it. Invalid args trigger a retry up to
    # ``max_parse_retries`` times before the loop gives up and leaves
    # ``AgentResult.parsed`` as None.
    output_schema: type[T] | None = None
    max_parse_retries: int = 2
    # "legacy" (default), "native", or "native_two_phase" — see
    # StructuredOutputMode. Only meaningful when output_schema is set;
    # both native modes require it.
    structured_output_mode: StructuredOutputMode = "legacy"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentConfig.name must be non-empty.")
        if self.max_tool_calls <= 0:
            raise ValueError(
                f"max_tool_calls must be positive, got {self.max_tool_calls}."
            )
        if self.max_llm_calls <= 0:
            raise ValueError(
                f"max_llm_calls must be positive, got {self.max_llm_calls}."
            )
        if self.max_llm_retries <= 0:
            raise ValueError(
                f"max_llm_retries must be positive, got {self.max_llm_retries}."
            )
        if self.max_parse_retries < 0:
            raise ValueError(
                f"max_parse_retries must be non-negative, got {self.max_parse_retries}."
            )
        if self.structured_output_mode not in ("legacy", "native", "native_two_phase"):
            raise ValueError(
                f"structured_output_mode must be 'legacy', 'native', or "
                f"'native_two_phase', got {self.structured_output_mode!r}."
            )
        if (
            self.structured_output_mode in ("native", "native_two_phase")
            and self.output_schema is None
        ):
            raise ValueError(
                f"structured_output_mode={self.structured_output_mode!r} "
                f"requires output_schema to be set."
            )
        if (
            isinstance(self.feedback_limit, bool)
            or not isinstance(self.feedback_limit, int)
            or self.feedback_limit < 1
        ):
            raise ValueError(
                f"feedback_limit must be a positive int, got {self.feedback_limit!r}."
            )

    def with_(self, **overrides: Any) -> AgentConfig[T]:
        """Return a new config with the given fields replaced.

        Preserves the generic parameter ``T`` so typed configs stay typed
        across variants::

            base = AgentConfig[StrategyOutput](..., grader=explore_grader)
            refined = base.with_(llm=other_client, max_tool_calls=15, grader=strict_grader)

        Any unknown field names raise ``TypeError`` (from ``dataclasses.replace``);
        ``__post_init__`` validation runs on the new instance.
        """
        return dataclasses.replace(self, **overrides)


@dataclass
class AgentResult[T]:
    output: str
    trace_id: str
    usage: dict[str, Any]
    scores: list[Score] | None
    duration_ms: float
    parsed: T | None = None
    # Structured termination reason. ``None`` on successful completion;
    # a typed :class:`AgentError` (one of the category-tagged subclasses)
    # when the runner gave up. Prefer this over string-matching
    # ``output`` for ``[agent terminated: ...]`` markers.
    error: AgentError | None = None


def _validate_output_schema(schema: type) -> None:
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"output_schema must be a pydantic BaseModel subclass, got {schema!r}"
        )


def _normalize_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Close a JSON schema for strict decoding, recursively.

    Strict decoders (OpenAI-style ``"strict": true`` on function calls and
    ``json_schema`` response formats) reject any object node that permits
    unknown keys or leaves a declared property optional.
    ``model_json_schema()`` guarantees neither: nested ``$defs`` stay open
    and defaulted fields are omitted from ``required``. Every object node
    gets ``additionalProperties: false`` (overriding an explicit ``true``
    from ``extra="allow"``) and a ``required`` listing every property.
    A schema-valued ``additionalProperties`` (pydantic's ``dict[str, X]``)
    is normalized in place rather than clobbered — backends that can't
    decode it should fail loudly, not silently receive an empty-object
    constraint.
    """
    out = dict(schema)
    for key in ("$defs", "properties"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = {
                name: _normalize_strict_schema(sub) if isinstance(sub, dict) else sub
                for name, sub in value.items()
            }
    for key in ("items", "contains", "propertyNames", "not"):
        if isinstance(out.get(key), dict):
            out[key] = _normalize_strict_schema(out[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = [
                _normalize_strict_schema(sub) if isinstance(sub, dict) else sub
                for sub in value
            ]
    if out.get("type") == "object" or "properties" in out:
        if isinstance(out.get("additionalProperties"), dict):
            out["additionalProperties"] = _normalize_strict_schema(
                out["additionalProperties"]
            )
        else:
            out["additionalProperties"] = False
        if "properties" in out:
            out["required"] = list(out["properties"])
    return out


def _build_submit_output_tool(schema: type[BaseModel]) -> ToolDefinition:
    parameters = _normalize_strict_schema(schema.model_json_schema())
    return ToolDefinition(
        name=SUBMIT_OUTPUT_TOOL,
        description=(
            "Submit your final answer. Call this exactly once when you have "
            "your result — do not produce a free-form text response as your "
            "final answer."
        ),
        parameters=parameters,
        strict=True,
    )


def _build_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic output_schema into the portable response_format
    envelope: ``{"type": "json_schema", "json_schema": {"name", "schema",
    "strict"}}``. ``name`` is the schema class name — deterministic and
    stable across runs of the same agent config.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": _normalize_strict_schema(schema.model_json_schema()),
            "strict": True,
        },
    }


def _append_schema_instruction(system_message: str) -> str:
    return (
        f"{system_message}\n\n"
        f"When you have your final answer, call the `{SUBMIT_OUTPUT_TOOL}` "
        f"tool with your result matching the provided schema. Do not produce "
        f"a free-form text response as your final answer — always finish by "
        f"calling `{SUBMIT_OUTPUT_TOOL}`."
    )


def _append_two_phase_instruction(system_message: str) -> str:
    return (
        f"{system_message}\n\n"
        f"When your work is complete, reply without tool calls; you will "
        f"then be asked for a final structured summary."
    )


def _serialize_config_snapshot(config: AgentConfig[Any]) -> dict[str, Any]:
    """Capture the state fields of an :class:`AgentConfig` for replay.

    Live objects (``llm``, ``tracer``, ``tools``, ``store``, ``retriever``,
    ``feedback``, ``grader``) are skipped — replay's caller supplies those
    fresh. Everything else is kept so :func:`jig.replay` can reconstruct
    an equivalent config.

    ``system_prompt`` is stored verbatim when it's a string; if a callable
    was used, we stash a sentinel and let replay force the caller to
    supply a new one via ``config_override``.

    ``output_schema`` is stored as its fully-qualified ``module:ClassName``
    so phase 11 replay can re-import it.
    """
    if callable(config.system_prompt):
        system_prompt: str | None = None
        system_prompt_is_callable = True
    else:
        system_prompt = config.system_prompt
        system_prompt_is_callable = False

    schema_fqn: str | None = None
    if config.output_schema is not None:
        schema_fqn = f"{config.output_schema.__module__}:{config.output_schema.__qualname__}"

    return {
        "agent_name": config.name,
        "description": config.description,
        "system_prompt": system_prompt,
        "system_prompt_is_callable": system_prompt_is_callable,
        "max_tool_calls": config.max_tool_calls,
        "max_llm_calls": config.max_llm_calls,
        "max_llm_retries": config.max_llm_retries,
        "max_parse_retries": config.max_parse_retries,
        "include_memory_in_prompt": config.include_memory_in_prompt,
        "include_feedback_in_prompt": config.include_feedback_in_prompt,
        "session_id": config.session_id,
        "output_schema": schema_fqn,
        "structured_output_mode": config.structured_output_mode,
        "feedback_limit": config.feedback_limit,
        "feedback_min_score": config.feedback_min_score,
        "feedback_source": config.feedback_source.value if config.feedback_source else None,
        "human_feedback_prompt_enabled": config.human_feedback_prompt.enabled,
        "human_feedback_prompt_dimensions": list(config.human_feedback_prompt.dimensions),
        # The LLMClient itself isn't JSON-serializable (live object,
        # gets dropped to ``null`` by _safe_json), so stamp the model
        # slug alongside it. Every shipped adapter sets ``self._model``;
        # custom adapters that don't will just record None.
        "model_id": _resolve_model_id(config.llm),
    }


def _reject_non_canonical(value: Any) -> None:
    """Recursively validate a value is safe for canonical JSON encoding.

    Raises ``ValueError`` for non-finite floats, non-string object keys, or
    any value outside JSON's native null / bool / finite-number / string /
    array / string-keyed-object surface. Silently coercing these (e.g. via
    ``str()`` or ``default=str``) would make the canonical hash lossy and
    non-reproducible across runs.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite float in structured output (value type: "
                f"{type(value).__name__})"
            )
        return
    if isinstance(value, list):
        for item in value:
            _reject_non_canonical(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"non-string object key in structured output (key type: "
                    f"{type(key).__name__})"
                )
            _reject_non_canonical(item)
        return
    raise ValueError(
        f"unsupported value in structured output (value type: "
        f"{type(value).__name__})"
    )


def _canonical_output_hash(value: Any) -> tuple[str, int]:
    """Return ``(sha256_hexdigest, utf8_byte_length)`` of ``value``'s
    canonical JSON encoding.

    Canonical means: UTF-8, keys sorted, compact separators, Unicode
    preserved verbatim (``ensure_ascii=False``), array order preserved,
    non-finite floats rejected (``allow_nan=False`` backstops
    :func:`_reject_non_canonical`, which already raises first with a
    stable message). This is the sole definition of output equality for
    :func:`jig.replay.diff.trace_diff` — the 200-character root preview is
    presentation-only and must never be hashed or compared for equality.
    """
    _reject_non_canonical(value)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _resolve_model_id(client: LLMClient | None) -> str | None:
    """Walk past LLMClient wrappers (e.g. BudgetedLLMClient) to find ``_model``.

    Wrappers store the wrapped client at ``_inner`` by convention. Without
    this unwrap, traces record ``model_id=None`` whenever a budget tracker
    or similar decorator sits in front of the real adapter — exactly the
    observability hole this stamping is meant to close.
    """
    seen: set[int] = set()
    current: Any = client
    while current is not None and id(current) not in seen:
        model = getattr(current, "_model", None)
        if model is not None:
            return model
        seen.add(id(current))
        current = getattr(current, "_inner", None)
    return None


async def _finalize_trace(
    tracer: TracingLogger,
    trace_span: Any,
    final_output: str,
    scores: list[Score] | None,
    agent_error: AgentError | None,
    complete_output: Any | None,
    output_kind: str,
) -> None:
    """Close the root span and flush the tracer, best-effort.

    Called from ``run_agent``'s ``finally`` so buffered tracers (notably
    ``SQLiteTracer``) don't drop failure traces when an exception propagates
    mid-run. Exceptions from ``end_span`` or ``flush`` are logged and
    swallowed so they don't shadow the original error.

    ``complete_output`` is the post-Pydantic, JSON-native structured value
    (``BaseModel.model_dump(mode="json")``) when the run produced a
    validated ``submit_output`` result, or ``None`` otherwise (no schema,
    or the run terminated before validating one). When present, its
    canonical SHA-256 and UTF-8 byte length are persisted alongside it on
    the AGENT_RUN root — see ``ROOT_OUTPUT_*_KEY`` — so
    :func:`jig.replay.diff.trace_diff` can compare complete output instead
    of the 200-char ``output`` preview. The accepted ``submit_output``
    TOOL_CALL span's raw arguments are not used for this: Pydantic can add
    defaults or coerce types the model never sent, so the raw call
    arguments are not reliably byte-identical to the validated result.
    Canonicalization failure (a non-finite float, an unsupported runtime
    value) is logged and treated the same as "no complete output" — it
    must not abort trace finalization.
    """
    trace_output: dict[str, Any] = {
        "output": final_output[:200],
        "scores": scores,
        ROOT_OUTPUT_KIND_KEY: output_kind,
    }
    if complete_output is not None:
        try:
            output_hash, output_len = _canonical_output_hash(complete_output)
        except Exception:
            # Canonicalization failure (non-finite float, unsupported
            # value, or an unexpected error from json.dumps/recursion
            # such as TypeError/OverflowError/RecursionError) must not
            # abort finalization — treat it the same as "no complete
            # output" and still close/flush the trace.
            logger.exception(
                "failed to canonicalize structured output for trace persistence"
            )
        else:
            trace_output[ROOT_OUTPUT_COMPLETE_KEY] = complete_output
            trace_output[ROOT_OUTPUT_SHA256_KEY] = output_hash
            trace_output[ROOT_OUTPUT_BYTE_LENGTH_KEY] = output_len
    if agent_error is not None:
        trace_output["error_category"] = agent_error.category
    try:
        tracer.end_span(
            trace_span.id,
            trace_output,
            error=str(agent_error) if agent_error is not None else None,
        )
    except Exception:
        logger.exception("tracer.end_span failed during finalization")
    try:
        await tracer.flush()
    except Exception:
        logger.exception("tracer.flush failed during finalization")


async def run_agent[T](config: AgentConfig[T], input: str) -> AgentResult[T]:
    start = time.time()

    if config.output_schema is not None:
        _validate_output_schema(config.output_schema)
    output_kind = "structured" if config.output_schema is not None else "text"

    # Fail fast, before the trace even starts: a native-mode agent on a
    # client that hasn't declared response_format support must never
    # silently run unconstrained or fall back to legacy. This is a
    # caller-side contract violation, not a provider failure, so it
    # propagates as a raised exception rather than an AgentResult.error.
    if (
        config.structured_output_mode in ("native", "native_two_phase")
        and not config.llm.supports_response_format
    ):
        raise UnsupportedResponseFormatError(
            f"LLM client {type(config.llm).__name__} does not declare "
            f"response_format support; cannot run agent {config.name!r} in "
            f"structured_output_mode={config.structured_output_mode!r}."
        )

    # 1. Start trace. Snapshot the config alongside the input so phase 11
    # replay can reconstruct an equivalent AgentConfig without the caller
    # having to supply state fields that were already recorded.
    trace = config.tracer.start_trace(
        config.name,
        {"input": input, "config": _serialize_config_snapshot(config)},
        kind=SpanKind.AGENT_RUN,
    )

    # State hoisted so the finally block can close the trace even if an
    # exception propagates from memory.add / grading / etc. mid-run.
    final_output = ""
    parsed: T | None = None
    structured_complete: Any | None = None
    scores: list[Score] | None = None
    agent_error: AgentError | None = None
    total_usage: dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
    }

    try:
        # 2. Resolve system prompt
        if callable(config.system_prompt):
            result = config.system_prompt()
            system_prompt = await result if hasattr(result, "__await__") else result
        else:
            system_prompt = config.system_prompt

        # 3. Retrieve memory context via the swappable Retriever
        memory_context: list[MemoryEntry] = []
        if config.include_memory_in_prompt and config.retriever is not None:
            with span_guard(
                config.tracer, trace.id, SpanKind.MEMORY_QUERY, "retrieve",
                input={"query": input},
            ) as mem_span:
                logger.debug("retriever.retrieve start (k=5)")
                memory_context = await config.retriever.retrieve(input, k=5)
                logger.debug("retriever.retrieve done (hits=%d)", len(memory_context))
                # Span output carries retrieved-id scores so downstream
                # analytics can group "which retriever picked what?" without
                # replaying the corpus.
                mem_span.finish(
                    {
                        "retrieved": [
                            {"id": e.id, "score": e.score, "preview": e.content[:120]}
                            for e in memory_context
                        ],
                    },
                )

        # 4. Query feedback signals
        feedback_signals: list[ScoredResult] = []
        if config.include_feedback_in_prompt:
            with span_guard(
                config.tracer, trace.id, SpanKind.MEMORY_QUERY, "query_feedback",
                input={"query": input},
            ) as fb_span:
                logger.debug(
                    "feedback.get_signals start (limit=%d, min_score=%s, source=%s)",
                    config.feedback_limit, config.feedback_min_score, config.feedback_source,
                )
                feedback_signals = await config.feedback.get_signals(
                    input,
                    limit=config.feedback_limit,
                    min_score=config.feedback_min_score,
                    source=config.feedback_source,
                )
                logger.debug("feedback.get_signals done (signals=%d)", len(feedback_signals))
                fb_span.finish([s.content[:100] for s in feedback_signals])

        # 4b. Query human-reviewed feedback examples (opt-in, separate from
        # the legacy signals above).
        human_examples: HumanExampleSet | None = None
        if config.human_feedback_prompt.enabled:
            with span_guard(
                config.tracer, trace.id, SpanKind.MEMORY_QUERY, "query_human_feedback",
                input={"query": input},
            ) as hf_span:
                human_examples = await config.feedback.get_human_examples(
                    input, config.human_feedback_prompt,
                )
                hf_span.finish({
                    "positive": len(human_examples.positive),
                    "negative": len(human_examples.negative),
                })

        # 5. Assemble messages (system prompt is separate, not in messages list)
        system_message = build_system_message(system_prompt, memory_context, feedback_signals)
        if human_examples is not None:
            system_message += build_human_feedback_section(
                human_examples, config.human_feedback_prompt.total_character_budget,
            )
        is_legacy_structured = (
            config.output_schema is not None and config.structured_output_mode == "legacy"
        )
        is_native_structured = (
            config.output_schema is not None and config.structured_output_mode == "native"
        )
        is_two_phase_structured = (
            config.output_schema is not None
            and config.structured_output_mode == "native_two_phase"
        )
        if is_legacy_structured:
            system_message = _append_schema_instruction(system_message)
        elif is_two_phase_structured:
            system_message = _append_two_phase_instruction(system_message)
        messages: list[Message] = []
        if config.session_id and config.store is not None:
            history = await config.store.get_session(config.session_id)
            messages.extend(history)
        messages.append(Message(role=Role.USER, content=input))

        # Build the tool list. Legacy mode adds the synthetic submit_output
        # tool; native mode relies on response_format instead and leaves the
        # tool list to ordinary working tools only.
        user_tools = config.tools.list()
        extra_tools: list[ToolDefinition] = []
        if is_legacy_structured:
            if any(t.name == SUBMIT_OUTPUT_TOOL for t in user_tools):
                raise ValueError(
                    f"Tool name {SUBMIT_OUTPUT_TOOL!r} is reserved by the runner "
                    f"when output_schema is set. Rename the user tool."
                )
            extra_tools.append(_build_submit_output_tool(config.output_schema))
        tools_for_llm = (user_tools + extra_tools) or None

        response_format = (
            _build_response_format(config.output_schema) if is_native_structured else None
        )
        # Two-phase keeps working turns unconstrained; this envelope is
        # attached only to the finalize call.
        finalize_response_format = (
            _build_response_format(config.output_schema)
            if is_two_phase_structured else None
        )

        # 6. LLM call + tool loop
        tool_call_count = 0
        consecutive_llm_errors = 0
        parse_retries = 0
        # native_two_phase only: set when the model's first no-tool-call turn
        # has requested the schema-constrained, tool-free finalize call.
        finalize_pending = False

        while True:
            if total_usage["llm_calls"] >= config.max_llm_calls:
                agent_error = AgentMaxLLMCallsError(config.max_llm_calls)
                final_output = f"[agent terminated: {agent_error}]"
                break

            # Count every attempted round-trip so the cap applies to failures
            # too, not just successes. A flaky LLM would otherwise be able to
            # race past max_llm_calls via the consecutive-errors retry path.
            total_usage["llm_calls"] += 1
            # span_guard ensures the LLM span is closed even if llm.complete
            # raises a non-JigLLMError exception (e.g. RuntimeError from a
            # buggy adapter). JigLLMError is still handled explicitly inside
            # the block so the runner can record the right error string and
            # choose break vs. continue.
            with span_guard(
                config.tracer, trace.id, SpanKind.LLM_CALL, "completion",
                metadata={"model": _resolve_model_id(config.llm)},
            ) as llm_span:
                params = CompletionParams(
                    messages=messages,
                    system=system_message,
                    tools=None if finalize_pending else tools_for_llm,
                    response_format=(
                        finalize_response_format if finalize_pending
                        else response_format
                    ),
                )

                try:
                    logger.debug(
                        "llm.complete start (call %d/%d, messages=%d, tools=%d)",
                        total_usage["llm_calls"],
                        config.max_llm_calls,
                        len(messages),
                        len(params.tools) if params.tools else 0,
                    )
                    response = await config.llm.complete(params)
                    logger.debug(
                        "llm.complete done (latency_ms=%s, tool_calls=%d)",
                        getattr(response, "latency_ms", "?"),
                        len(response.tool_calls or []),
                    )
                except JigLLMError as e:
                    llm_span.finish(error=str(e))
                    # Fast-fail only on known-permanent errors (auth, bad
                    # request, not-found). Unknown errors and 5xx go through
                    # the retry path — adapters default `retryable=False`
                    # broadly, so reading that flag would collapse transient
                    # server errors into immediate termination.
                    if e.status_code in _PERMANENT_LLM_STATUS_CODES:
                        agent_error = AgentLLMPermanentError(
                            provider=e.provider,
                            message=str(e),
                            status_code=e.status_code,
                        )
                        final_output = f"[agent terminated: {agent_error}]"
                        logger.warning(
                            "Permanent LLM error (status %s); terminating: %s",
                            e.status_code, e,
                        )
                        break
                    consecutive_llm_errors += 1
                    logger.warning(
                        "LLM call failed (%d/%d): %s",
                        consecutive_llm_errors, config.max_llm_retries, e,
                    )
                    if consecutive_llm_errors >= config.max_llm_retries:
                        agent_error = AgentMaxLLMRetriesError(consecutive_llm_errors, str(e))
                        final_output = f"[agent terminated: {agent_error}]"
                        break
                    # Inject an error message so the model can see it on next turn
                    messages.append(Message(
                        role=Role.USER,
                        content=f"[system: LLM call failed: {e}. Please continue.]",
                    ))
                    continue

                consecutive_llm_errors = 0
                llm_span.finish(
                    {"content": response.content[:200], "tool_calls": len(response.tool_calls or [])},
                    usage=response.usage,
                )

            total_usage["total_input_tokens"] += response.usage.input_tokens
            total_usage["total_output_tokens"] += response.usage.output_tokens
            total_usage["total_cost"] += response.usage.cost or 0.0

            # --- Structured-output termination path ---
            # When a schema is set, submit_output must be the ONLY tool call in
            # its turn. Otherwise, silently terminating on submit_output would
            # drop the other tool executions the model just requested — bugs
            # that look like "my RAG lookup result vanished."
            submit_calls: list[ToolCall] = []
            other_tool_calls: list[ToolCall] = []
            if is_legacy_structured and response.tool_calls:
                for call in response.tool_calls:
                    if call.name == SUBMIT_OUTPUT_TOOL:
                        submit_calls.append(call)
                    else:
                        other_tool_calls.append(call)

            # Ambiguous turn: submit_output combined with other tool calls, or
            # emitted more than once. Ask the model to retry with submit_output
            # alone. Counts against max_parse_retries — same failure budget as
            # invalid args, since both are "model didn't follow the schema
            # contract."
            if submit_calls and (len(submit_calls) > 1 or other_tool_calls):
                parse_retries += 1
                logger.info(
                    "submit_output emitted alongside other tool calls "
                    "(%d/%d)", parse_retries, config.max_parse_retries,
                )
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                for call in response.tool_calls:
                    if call.name == SUBMIT_OUTPUT_TOOL:
                        content = (
                            f"Ambiguous turn: when calling `{SUBMIT_OUTPUT_TOOL}`,"
                            f" it must be the only tool call. Retry with "
                            f"`{SUBMIT_OUTPUT_TOOL}` alone."
                        )
                    else:
                        content = (
                            f"[skipped: `{SUBMIT_OUTPUT_TOOL}` must be the only "
                            f"call in its turn. Run this tool in a separate turn "
                            f"before submitting.]"
                        )
                    messages.append(
                        Message(role=Role.TOOL, content=content, tool_call_id=call.id)
                    )
                if parse_retries > config.max_parse_retries:
                    agent_error = AgentAmbiguousTurnError(parse_retries)
                    final_output = f"[agent terminated: {agent_error}]"
                    break
                continue

            submit_call: ToolCall | None = submit_calls[0] if submit_calls else None

            if submit_call is not None:
                with span_guard(
                    config.tracer, trace.id, SpanKind.TOOL_CALL, SUBMIT_OUTPUT_TOOL,
                    input=submit_call.arguments,
                ) as extract_span:
                    try:
                        parsed = config.output_schema.model_validate(submit_call.arguments)  # type: ignore[union-attr]
                    except ValidationError as ve:
                        parse_retries += 1
                        extract_span.finish(error=str(ve))
                        logger.info(
                            "submit_output validation failed (%d/%d): %s",
                            parse_retries, config.max_parse_retries, ve,
                        )
                        # Record the assistant's attempt and feed the validation
                        # errors back so the model can correct. Do not execute any
                        # other tools in this turn — the model is in retry mode.
                        messages.append(
                            Message(
                                role=Role.ASSISTANT,
                                content=response.content,
                                tool_calls=response.tool_calls,
                            )
                        )
                        # Every tool call in this turn must receive a response
                        # (providers reject assistant-with-tool_calls followed by
                        # user text without matching tool messages in between).
                        for call in response.tool_calls:
                            if call.name == SUBMIT_OUTPUT_TOOL:
                                content = (
                                    f"Validation failed: {ve.error_count()} error(s).\n"
                                    f"{ve}\n"
                                    f"Call {SUBMIT_OUTPUT_TOOL} again with corrected arguments."
                                )
                            else:
                                content = (
                                    f"[skipped: submit_output retry in progress — "
                                    f"please call {SUBMIT_OUTPUT_TOOL} with valid "
                                    f"arguments]"
                                )
                            messages.append(
                                Message(role=Role.TOOL, content=content, tool_call_id=call.id)
                            )
                        if parse_retries > config.max_parse_retries:
                            agent_error = AgentSchemaValidationError(parse_retries, str(ve))
                            final_output = f"[agent terminated: {agent_error}]"
                            break
                        continue

                    # Validation succeeded — finalize.
                    extract_span.finish(submit_call.arguments)
                    final_output = parsed.model_dump_json()
                    # Pydantic can add unset-field defaults or coerce types
                    # the model never sent, so the raw submit_output
                    # arguments recorded above are not reliable complete
                    # evidence — persist the validated value itself.
                    structured_complete = parsed.model_dump(mode="json")
                    messages.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=response.content,
                            tool_calls=response.tool_calls,
                        )
                    )
                    break

            # --- Two-phase structured-output paths ---
            # Working turns run unconstrained. The first no-tool-call turn
            # requests one schema-constrained, tool-free finalize call; that
            # call's content is the terminal result, parsed with the same
            # decode-time-enforcement rationale as native mode.
            if is_two_phase_structured and finalize_pending and response.tool_calls:
                # tools=None was sent on the finalize call — a tool call here
                # is a provider anomaly. Nudge within the parse budget, then
                # fail closed (same budget as legacy's ambiguous turns).
                parse_retries += 1
                logger.info(
                    "tool call emitted on finalize turn (%d/%d)",
                    parse_retries, config.max_parse_retries,
                )
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                for call in response.tool_calls:
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=(
                                "[skipped: finalize turn — respond with the "
                                "final structured output only, no tool calls]"
                            ),
                            tool_call_id=call.id,
                        )
                    )
                if parse_retries > config.max_parse_retries:
                    agent_error = AgentAmbiguousTurnError(parse_retries)
                    final_output = f"[agent terminated: {agent_error}]"
                    break
                continue

            if is_two_phase_structured and not response.tool_calls:
                if not finalize_pending:
                    finalize_pending = True
                    messages.append(
                        Message(role=Role.ASSISTANT, content=response.content)
                    )
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "Produce your final structured output now, "
                                "matching the required schema."
                            ),
                        )
                    )
                    continue
                assert config.output_schema is not None  # guaranteed by __post_init__
                try:
                    parsed = config.output_schema.model_validate_json(response.content)
                except (ValidationError, ValueError) as ve:
                    agent_error = AgentNativeOutputError(str(ve))
                    final_output = f"[agent terminated: {agent_error}]"
                    break
                final_output = parsed.model_dump_json()
                # Same rationale as the native path below: persist the
                # validated value, not the raw content.
                structured_complete = parsed.model_dump(mode="json")
                break

            # --- Native structured-output termination path ---
            # A schema-constrained turn with no tool calls is the terminal
            # result. Parsed once, no retry: response_format is decode-time
            # enforcement, so a violation here is a provider bug or schema
            # drift, not a correctable model mistake — see
            # AgentNativeOutputError.
            if is_native_structured and not response.tool_calls:
                assert config.output_schema is not None  # guaranteed by __post_init__
                try:
                    parsed = config.output_schema.model_validate_json(response.content)
                except (ValidationError, ValueError) as ve:
                    agent_error = AgentNativeOutputError(str(ve))
                    final_output = f"[agent terminated: {agent_error}]"
                    break
                final_output = parsed.model_dump_json()
                # Same rationale as the legacy submit_output path: persist
                # the validated value, not the raw content, as complete
                # evidence (Pydantic may add unset-field defaults).
                structured_complete = parsed.model_dump(mode="json")
                break

            # --- Plain-text termination path (no schema, or legacy schema set
            # but model didn't call submit_output this turn). ---
            if not response.tool_calls:
                if config.output_schema is not None:
                    # Model ignored the schema instruction. Nudge and retry, up
                    # to max_parse_retries attempts.
                    parse_retries += 1
                    if parse_retries > config.max_parse_retries:
                        # Fail closed: the caller explicitly asked for a typed
                        # output, so returning the model's free-form content
                        # would make a non-compliant run look successful. Leave
                        # parsed as None and attach a structured AgentError.
                        agent_error = AgentSchemaNotCalledError(parse_retries)
                        final_output = f"[agent terminated: {agent_error}]"
                        break
                    messages.append(
                        Message(role=Role.ASSISTANT, content=response.content)
                    )
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=(
                                f"You must call the `{SUBMIT_OUTPUT_TOOL}` tool "
                                f"with your final answer. Do not respond with "
                                f"plain text."
                            ),
                        )
                    )
                    continue
                final_output = response.content
                break

            # --- Execute user-tool calls (no submit_output in this turn). ---
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                if tool_call_count >= config.max_tool_calls:
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content="Max tool calls reached. Provide final answer.",
                            tool_call_id=call.id,
                        )
                    )
                    continue

                tool_span = config.tracer.start_span(
                    trace.id, SpanKind.TOOL_CALL, call.name, call.arguments
                )
                tool_context = ToolExecutionContext(
                    trace_id=trace.trace_id,
                    span_id=tool_span.id,
                    parent_span_id=tool_span.parent_id,
                    tool_call_id=call.id,
                    metadata={"tool_name": call.name},
                )
                token = current_tool_context.set(tool_context)
                try:
                    result = await config.tools.execute(call)
                finally:
                    current_tool_context.reset(token)
                # Full tool output is persisted (no truncation). Phase 11
                # replay substitutes recorded tool outputs into a rerun of
                # the agent loop — truncation would silently corrupt that
                # substitution and hide real tool behavior.
                config.tracer.end_span(
                    tool_span.id, result.output, error=result.error,
                )

                # Surface tool errors to the model so it can react
                if result.error:
                    content = f"[tool error: {result.error}]"
                    if result.output:
                        content = f"{result.output}\n{content}"
                else:
                    content = result.output

                messages.append(
                    Message(role=Role.TOOL, content=content, tool_call_id=call.id)
                )
                tool_call_count += 1
                total_usage["tool_calls"] += 1

        # 7. Post-output bookkeeping: only for successful (non-terminated) runs.
        # Terminated runs (agent_error set) skip all persistence so sentinel text
        # does not pollute memory, session history, or feedback signals. After a
        # valid output is produced, each bookkeeping operation is fail-soft:
        # exceptions are logged and the valid output is still returned.
        memory_id: str | None = None
        if agent_error is None:
            if config.store is not None:
                try:
                    memory_id = await config.store.add(
                        final_output,
                        {"agent": config.name, "input": input, "trace_id": trace.trace_id},
                    )
                except Exception:
                    logger.exception("memory.add failed after successful run (non-fatal)")

            if config.session_id and config.store is not None:
                try:
                    await config.store.add_to_session(
                        config.session_id, Message(role=Role.USER, content=input)
                    )
                    await config.store.add_to_session(
                        config.session_id, Message(role=Role.ASSISTANT, content=final_output)
                    )
                except Exception:
                    logger.exception("store.add_to_session failed after successful run (non-fatal)")

            # 8. Auto-grade: register the output as a feedback result first so
            # scores reference a real result row that query/get_signals/export can
            # join against. Memory IDs and trace IDs are separate namespaces and
            # must not be used as the feedback result ID.
            if config.grader:
                # Flush so trajectory graders can read pre-grade spans via
                # get_trace. SQLiteTracer retains unended spans (notably the
                # root) so _finalize_trace's end_span still works.
                await config.tracer.flush()
                grade_output: Any = parsed if parsed is not None else final_output
                fb_meta: dict[str, Any] = {
                    "kind": "agent_result",
                    "agent_name": config.name,
                    "source": "run_agent",
                    "trace_id": trace.trace_id,
                }
                model_id = _resolve_model_id(config.llm)
                if model_id is not None:
                    fb_meta["model"] = model_id
                if config.session_id is not None:
                    fb_meta["session_id"] = config.session_id
                if memory_id is not None:
                    fb_meta["memory_id"] = memory_id
                outcome = await grade_and_record(
                    tracer=config.tracer,
                    parent_span_id=trace.id,
                    span_name="auto_grade",
                    grader=config.grader,
                    grade_input=input,
                    grade_output=grade_output,
                    grade_context={"raw_output": final_output, "trace_id": trace.trace_id},
                    feedback=config.feedback,
                    feedback_content=final_output,
                    feedback_input_text=input,
                    feedback_metadata=fb_meta,
                )
                scores = outcome.scores
    finally:
        # Always close the root span + flush the tracer, even if an
        # exception propagates mid-run. Buffered tracers (SQLiteTracer)
        # would otherwise drop the failure trace.
        await _finalize_trace(
            config.tracer, trace, final_output, scores, agent_error,
            structured_complete, output_kind,
        )

    return AgentResult(
        output=final_output,
        trace_id=trace.trace_id,
        usage=total_usage,
        scores=scores,
        duration_ms=(time.time() - start) * 1000,
        parsed=parsed,
        error=agent_error,
    )
