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

from typing import Any


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
        self.retryable = retryable


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
        self.retryable = retryable


class JigToolError(JigError):
    """Raised by tool execution or schema validation."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        phase: str,
        retryable: bool = False,
        underlying: BaseException | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.phase = phase          # one of "schema" | "execute" | "serialize"
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
