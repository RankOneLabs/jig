from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from jig.core.errors import JigLLMError
from jig.core.retry import with_retry
from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
)

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o", **client_kwargs: Any):
        if openai is None:
            raise ImportError("Install openai: pip install 'jig[openai]'")
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model

    def _convert_messages(self, params: CompletionParams) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if params.system:
            messages.append({"role": "system", "content": params.system})
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.TOOL:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                messages.append({"role": msg.role.value, "content": msg.content})
        return messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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

    async def complete(self, params: CompletionParams) -> LLMResponse:
        messages = self._convert_messages(params)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if params.tools:
            kwargs["tools"] = self._convert_tools(params.tools)
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        if params.provider_params:
            kwargs.update(params.provider_params)

        start = time.time()

        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        def _retryable(e: Exception) -> bool:
            if openai is None:
                return False
            return isinstance(e, openai.RateLimitError)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except Exception as e:
            if openai and isinstance(e, openai.AuthenticationError):
                raise
            raise JigLLMError(str(e), "openai") from e

        latency_ms = (time.time() - start) * 1000
        choice = response.choices[0].message

        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.tool_calls
            ]

        return LLMResponse(
            content=choice.content or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            latency_ms=latency_ms,
            model=response.model,
        )

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        messages = self._convert_messages(params)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if params.tools:
            kwargs["tools"] = self._convert_tools(params.tools)
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        if params.provider_params:
            kwargs.update(params.provider_params)

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
