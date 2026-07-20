"""Typed error hierarchy.

Three layers:

- ``JigError`` — base for everything jig raises.
- Provider/component errors (``JigLLMError``, ``JigMemoryError``,
  ``JigToolError``, ``JigBudgetError``) — raised by adapters and utilities.
  They carry structured fields so retry logic and diagnostic rollups can
  act on them without string matching.
- ``AgentError`` — terminal agent-loop conditions. Set on
  ``AgentResult.error`` when ``run_agent`` gives up. Each subclass has a
  ``category`` tag that flows into trace span metadata so you can query
  "how often did Haiku hit max_llm_calls vs schema_not_called?" without
  parsing text markers.
"""
from __future__ import annotations

from typing import Any, Literal

ToolErrorPhase = Literal["schema", "execute", "serialize"]


class JigError(Exception):
    pass


# ---------------------------------------------------------------------------
# Provider / component errors
# ---------------------------------------------------------------------------


class JigLLMError(JigError):
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        # Diagnostic metadata set by the adapter — indicates whether the
        # provider considers the error transient. The runner does NOT use
        # this flag to decide retry behavior; it routes via status_code
        # (permanent on 400/401/403/404/422) and consecutive-error counts.
        # Callers inspecting retryable are reading provider intent, not a
        # runner retry contract.
        self.retryable = retryable


class UnsupportedResponseFormatError(ValueError):
    """A CompletionParams.response_format value the adapter cannot honor.

    Raised for two distinct cases: the adapter has no structured-output
    support at all, or the value doesn't match the portable
    ``{"type": "json_schema", "json_schema": {"schema": {...}}}`` shape
    a supporting adapter requires. Deliberately a ``ValueError`` subclass
    rather than ``JigError`` — this is a caller-side contract violation
    (bad request shape / unsupported capability), not a provider or
    network failure, and callers should be able to catch it distinctly
    from ``JigLLMError`` without it being swallowed by broader jig error
    handling.
    """


class JigMemoryError(JigError):
    """Raised by memory backends (store, retriever, session)."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        operation: str,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.source = source        # e.g. "local_memory", "honcho", "zep"
        self.operation = operation  # e.g. "query", "add", "get_session"
        # Provider-local diagnostic hint — not a runner retry contract.
        # The runner's fail-soft bookkeeping catches all exceptions after a
        # valid output; callers must not rely on this flag to drive retries.
        self.retryable = retryable


class JigToolError(JigError):
    """Raised by tool execution or schema validation.

    ``phase`` marks *where* in the tool lifecycle the error occurred:
    ``"schema"`` (arg validation before execute), ``"execute"`` (the
    tool's own code), or ``"serialize"`` (encoding the result back).
    The ``Literal`` typing keeps the contract honest at the type-checker
    level without adding runtime overhead.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        phase: ToolErrorPhase,
        retryable: bool = False,
        underlying: BaseException | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.phase: ToolErrorPhase = phase
        self.retryable = retryable
        self.underlying = underlying


class JigBudgetError(JigError):
    """Raised when a :class:`BudgetTracker` observes spend exceeding its limit."""

    def __init__(self, message: str, spent_usd: float, limit_usd: float):
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


# ---------------------------------------------------------------------------
# Agent-loop terminal errors
# ---------------------------------------------------------------------------


class AgentError(JigError):
    """Base for terminal conditions in ``run_agent``.

    Not raised from ``run_agent`` — set on ``AgentResult.error`` and made
    available to callers that want structured termination reasons.
    ``category`` is a stable string tag used for span metadata and rollups.
    """

    category: str = "agent_error"

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.context: dict[str, Any] = context


class AgentMaxLLMCallsError(AgentError):
    """Hard cap on LLM round-trips per run was exceeded."""

    category = "max_llm_calls"

    def __init__(self, max_calls: int):
        super().__init__(
            f"Exceeded max_llm_calls ({max_calls})",
            max_calls=max_calls,
        )
        self.max_calls = max_calls


class AgentMaxLLMRetriesError(AgentError):
    """Too many consecutive LLM errors; run_agent gave up."""

    category = "max_llm_retries"

    def __init__(self, retries: int, last_error: str):
        super().__init__(
            f"Agent terminated after {retries} consecutive LLM errors; last: {last_error}",
            retries=retries,
            last_error=last_error,
        )
        self.retries = retries
        self.last_error = last_error


class AgentSchemaValidationError(AgentError):
    """Model called submit_output but its args failed pydantic validation
    more than ``max_parse_retries`` times."""

    category = "schema_validation_failed"

    def __init__(self, retries: int, last_error: str):
        super().__init__(
            f"submit_output validation failed {retries} times; last: {last_error}",
            retries=retries,
            last_error=last_error,
        )
        self.retries = retries
        self.last_error = last_error


class AgentSchemaNotCalledError(AgentError):
    """Model ignored the schema instruction and never called submit_output."""

    category = "schema_not_called"

    def __init__(self, retries: int):
        super().__init__(
            f"Model did not call submit_output within {retries} retries",
            retries=retries,
        )
        self.retries = retries


class GradeParseError(JigError):
    """Raised when a judge cannot parse the LLM response into valid scores.

    Distinct from a low-quality answer: the judge infrastructure failed,
    not the model under evaluation. Callers must not treat this as a
    score of 0.0 or any other numeric value.
    """


class AgentAmbiguousTurnError(AgentError):
    """Model kept emitting submit_output alongside other tool calls.

    The runner rejects these turns (terminating would silently drop the
    other requested tool executions). Repeated ambiguity past
    ``max_parse_retries`` gives up.
    """

    category = "ambiguous_tool_turn"

    def __init__(self, retries: int):
        super().__init__(
            f"Model combined submit_output with other tool calls {retries} times",
            retries=retries,
        )
        self.retries = retries


class AgentNativeOutputError(AgentError):
    """Native structured-output mode: the terminal assistant content failed
    JSON parsing or pydantic validation against ``output_schema``.

    Unlike the legacy ``submit_output`` path, native mode does not retry —
    decode-time response_format constraints should make this unreachable in
    ordinary operation, so a violation here signals a provider bug or a
    schema-translation defect rather than a correctable model mistake.
    """

    category = "native_structured_output_failed"

    def __init__(self, message: str):
        super().__init__(
            f"Native structured-output parsing failed: {message}",
            last_error=message,
        )
        self.last_error = message


class AgentLLMPermanentError(AgentError):
    """Runner terminated because the LLM raised a known-permanent error.

    The runner treats specific HTTP status codes as permanent (auth, invalid
    model, bad request). It does not use ``JigLLMError.retryable`` as an
    authoritative retry contract.
    """

    category = "llm_permanent_error"

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(
            f"Permanent LLM error from {provider}: {message}",
            provider=provider,
            status_code=status_code,
        )
        self.provider = provider
        self.status_code = status_code
        self.last_error = message


class AgentBudgetError(AgentError):
    """Budget cap was reached; this run was not admitted or could not complete.

    Non-retryable by definition — re-submitting the same call under the same
    budget will fail again. Sweep workers catch :class:`JigBudgetError` and
    convert it into this structured result so budget exhaustion becomes a
    per-case outcome rather than a worker-killing exception.
    """

    category = "budget_exhausted"

    def __init__(self, message: str, *, spent_usd: float, limit_usd: float):
        super().__init__(message, spent_usd=spent_usd, limit_usd=limit_usd)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
