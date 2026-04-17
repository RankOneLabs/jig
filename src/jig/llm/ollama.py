from __future__ import annotations

import time
import uuid
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
from jig.llm._parsing import parse_tool_arguments

try:
    import ollama as _ollama
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    _ollama = None  # type: ignore[assignment]
    OllamaAsyncClient = None  # type: ignore[assignment, misc]


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1", host: str | None = None):
        if OllamaAsyncClient is None:
            raise ImportError("Install ollama: pip install 'jig[ollama]'")
        self._client = OllamaAsyncClient(host=host)
        self._model = model

    def _convert_messages(self, params: CompletionParams) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if params.system:
            messages.append({"role": "system", "content": params.system})
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
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

        options: dict[str, Any] = {}
        if params.temperature is not None:
            options["temperature"] = params.temperature
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        if params.provider_params:
            options.update(params.provider_params)
        if options:
            kwargs["options"] = options

        start = time.time()

        async def _call() -> Any:
            return await self._client.chat(**kwargs)

        def _retryable(e: Exception) -> bool:
            return isinstance(e, ConnectionError)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except ConnectionError:
            raise JigLLMError("Cannot reach Ollama", "ollama", retryable=True)
        except Exception as e:
            if _ollama and isinstance(e, _ollama.ResponseError):
                raise JigLLMError(str(e), "ollama") from e
            raise

        latency_ms = (time.time() - start) * 1000

        tool_calls: list[ToolCall] | None = None
        raw_calls = response.get("message", {}).get("tool_calls")
        if raw_calls:
            try:
                tool_calls = [
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=tc["function"]["name"],
                        arguments=parse_tool_arguments(tc["function"]["arguments"], "ollama"),
                    )
                    for tc in raw_calls
                ]
            except (KeyError, TypeError) as e:
                raise JigLLMError(
                    f"Malformed tool call: {e}. Raw: {raw_calls}",
                    "ollama",
                    retryable=True,
                )

        return LLMResponse(
            content=response.get("message", {}).get("content", ""),
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.get("prompt_eval_count", 0),
                output_tokens=response.get("eval_count", 0),
                cost=0.0,
            ),
            latency_ms=latency_ms,
            model=self._model,
        )

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        messages = self._convert_messages(params)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }

        options: dict[str, Any] = {}
        if params.temperature is not None:
            options["temperature"] = params.temperature
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        if params.provider_params:
            options.update(params.provider_params)
        if options:
            kwargs["options"] = options

        response = await self._client.chat(**kwargs)
        async for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
