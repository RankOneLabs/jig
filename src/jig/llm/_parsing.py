from __future__ import annotations

import json
from typing import Any

from jig.core.errors import JigLLMError


def parse_tool_arguments(raw: Any, provider: str) -> dict[str, Any]:
    """Normalize provider-returned tool-call arguments to a dict.

    OpenAI returns arguments as a JSON string; Anthropic/Ollama/Google typically
    return dicts. A few Ollama models occasionally return a string. This handles
    both uniformly and raises a retryable JigLLMError on malformed input so
    callers get consistent error semantics.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise JigLLMError(
                f"Malformed tool call arguments: {e}. Raw: {raw!r}",
                provider,
                retryable=True,
            ) from e
        if not isinstance(parsed, dict):
            raise JigLLMError(
                f"Tool call arguments must be a JSON object, got {type(parsed).__name__}. Raw: {raw!r}",
                provider,
                retryable=True,
            )
        return parsed
    raise JigLLMError(
        f"Unexpected tool call arguments type: {type(raw).__name__}",
        provider,
    )
