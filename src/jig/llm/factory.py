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
    ToolDefinition,
)


def from_model(model: str, **overrides: Any) -> LLMClient:
    """Return an ``LLMClient`` suitable for the given model name."""
    if model.startswith("claude-"):
        from jig.llm.anthropic import AnthropicClient
        return AnthropicClient(model=model, **overrides)

    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
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
    if "role" not in m:
        raise ValueError("Message dict must include a 'role' field.")
    role = m["role"]
    if isinstance(role, str):
        try:
            role = Role(role)
        except ValueError as e:
            valid = [r.value for r in Role]
            raise ValueError(
                f"Invalid message role {m['role']!r}; expected one of {valid}."
            ) from e
    return Message(
        role=role,
        content=m.get("content", "") or "",
        tool_call_id=m.get("tool_call_id"),
        tool_calls=m.get("tool_calls"),
    )


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
    return await client.complete(params)
