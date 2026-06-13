"""Private shared utilities for LLM adapters.

Not exported from jig.llm — adapters import directly via ``from jig.llm._common import ...``.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from jig.core.errors import JigLLMError
from jig.core.types import CompletionParams, ToolDefinition


def start_timer() -> Callable[[], float]:
    """Return a callable that yields elapsed milliseconds since this call.

    Uses monotonic clock internally; the returned unit (ms float) is identical
    to the ``time.time()``-based pattern it replaces.
    """
    t = time.monotonic()
    return lambda: (time.monotonic() - t) * 1000


def merge_completion_kwargs(kwargs: dict[str, Any], params: CompletionParams) -> None:
    """Apply temperature, max_tokens, and provider_params to kwargs in-place.

    provider_params is merged last so caller-supplied values win over defaults.
    Only sets temperature/max_tokens when those fields are not None; pre-set
    defaults (e.g. Anthropic's ``max_tokens or 4096``) are left untouched when
    ``params.temperature`` or ``params.max_tokens`` is None.
    """
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.max_tokens is not None:
        kwargs["max_tokens"] = params.max_tokens
    if params.provider_params:
        kwargs.update(params.provider_params)


def openai_tool_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Serialize ToolDefinition list to the OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def wrap_llm_error(e: Exception, provider_label: str) -> JigLLMError:
    """Wrap an exception in JigLLMError, extracting status_code via getattr.

    Only for providers whose status handling uses ``getattr(e, 'status_code', None)``.
    Providers with custom status extraction (google, ollama, dispatch) keep their
    own error wrapping.
    """
    status = getattr(e, "status_code", None)
    return JigLLMError(str(e), provider_label, status_code=status)
