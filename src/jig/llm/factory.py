"""Prefix-routed LLM client factory.

``from_model("claude-sonnet-4-5")`` → ``AnthropicClient``
``from_model("gpt-5-mini")``        → ``OpenAIClient``
``from_model("gemini-2.5-pro")``    → ``GeminiClient``
``from_model("ollama/llama3.1")``   → ``OllamaClient`` (name stripped)
``from_model("dispatch/llama-70b")``→ ``DispatchClient`` (name stripped)

Overrides pass through to the adapter constructor (``host=``, ``dispatch_url=``,
``api_key=``, etc.).
"""
from __future__ import annotations

from typing import Any

from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)


def from_model(model: str, **overrides: Any) -> LLMClient:
    """Return an ``LLMClient`` suitable for the given model name."""
    if model.startswith("claude-"):
        from jig.llm.anthropic import AnthropicClient
        return AnthropicClient(model=model, **overrides)

    # o-series: accept bare "o1"/"o3"/"o4" and their suffixed variants
    # ("o3-mini", "o4-mini-2025-04-16"), but not unrelated names that happen
    # to share a prefix ("o11-mini" is not an OpenAI model).
    if (
        model.startswith(("gpt-", "chatgpt-"))
        or model in {"o1", "o3", "o4"}
        or model.startswith(("o1-", "o3-", "o4-"))
    ):
        from jig.llm.openai import OpenAIClient
        return OpenAIClient(model=model, **overrides)

    if model.startswith("gemini-"):
        from jig.llm.google import GeminiClient
        return GeminiClient(model=model, **overrides)

    if model.startswith("ollama/"):
        provider_model = model[len("ollama/"):]
        if not provider_model:
            raise ValueError("Model name required after 'ollama/' prefix.")
        from jig.llm.ollama import OllamaClient
        return OllamaClient(model=provider_model, **overrides)

    if model.startswith("dispatch/"):
        provider_model = model[len("dispatch/"):]
        if not provider_model:
            raise ValueError("Model name required after 'dispatch/' prefix.")
        from jig.llm.dispatch import DispatchClient
        return DispatchClient(model=provider_model, **overrides)

    raise ValueError(
        f"No provider matches model '{model}'. "
        f"Use a prefixed name (claude-*, gpt-*, o1/o3/o4-*, gemini-*, "
        f"ollama/<name>, dispatch/<name>)."
    )


def _coerce_message(m: Message | dict[str, Any]) -> Message:
    if isinstance(m, Message):
        return m
    if not isinstance(m, dict):
        raise ValueError(
            f"Each message must be a Message or dict, got {type(m).__name__}."
        )
    if "role" not in m:
        raise ValueError("Message dict must include a 'role' field.")
    role = m["role"]
    if isinstance(role, Role):
        pass
    elif isinstance(role, str):
        try:
            role = Role(role)
        except ValueError as e:
            valid = [r.value for r in Role]
            raise ValueError(
                f"Invalid message role {m['role']!r}; expected one of {valid}."
            ) from e
    else:
        valid = [r.value for r in Role]
        raise ValueError(
            f"Invalid message role type {type(role).__name__}; "
            f"expected str or Role (one of {valid})."
        )
    return Message(
        role=role,
        content=m.get("content", "") or "",
        tool_call_id=m.get("tool_call_id"),
        tool_calls=_coerce_tool_calls(m.get("tool_calls")),
    )


def _coerce_tool_calls(raw: Any) -> list[ToolCall] | None:
    """Coerce a list that may contain ToolCall or dict into list[ToolCall].

    ``Message.tool_calls`` is typed ``list[ToolCall]``; raw dicts would
    pass silently through the dataclass constructor and break downstream
    ``.name`` / ``.arguments`` access in the runner and adapters.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(
            f"tool_calls must be a list, got {type(raw).__name__}."
        )
    result: list[ToolCall] = []
    for i, tc in enumerate(raw):
        if isinstance(tc, ToolCall):
            result.append(tc)
        elif isinstance(tc, dict):
            missing = [k for k in ("id", "name") if k not in tc]
            if missing:
                raise ValueError(
                    f"tool_calls[{i}] missing required field(s): {missing}."
                )
            result.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments") or {},
                )
            )
        else:
            raise ValueError(
                f"tool_calls[{i}] must be ToolCall or dict, got {type(tc).__name__}."
            )
    return result


async def complete(
    model: str,
    messages: list[Message] | list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[ToolDefinition] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider_params: dict[str, Any] | None = None,
    client_kwargs: dict[str, Any] | None = None,
) -> LLMResponse:
    """One-shot completion: build a client, run one request, return the response.

    Convenience for call sites that don't need to hold a client reference.
    Accepts ``list[Message]`` or a list of ``{"role": ..., "content": ...}``
    dicts. ``client_kwargs`` forwards to the adapter constructor (e.g.
    ``{"host": "http://mcbain:11434"}`` for Ollama).
    """
    # Validate and coerce input before constructing the client — gives useful
    # errors even when the provider SDK isn't installed.
    coerced = [_coerce_message(m) for m in messages]
    client = from_model(model, **(client_kwargs or {}))
    params = CompletionParams(
        messages=coerced,
        system=system,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        provider_params=provider_params,
    )
    try:
        return await client.complete(params)
    finally:
        # Release connections held by clients that own their transport
        # (notably DispatchClient's httpx.AsyncClient). No-op on adapters
        # whose SDKs self-manage pooling (Anthropic, OpenAI, Gemini).
        await client.aclose()
