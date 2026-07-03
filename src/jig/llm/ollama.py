from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator

import httpx

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
from jig.llm._common import start_timer
from jig.llm._parsing import parse_tool_arguments

logger = logging.getLogger(__name__)

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

        timer = start_timer()

        async def _call() -> Any:
            return await self._client.chat(**kwargs)

        def _retryable(e: Exception) -> bool:
            return isinstance(e, ConnectionError)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except ConnectionError:
            raise JigLLMError("Cannot reach Ollama", "ollama", retryable=True)
        except httpx.ConnectError as e:
            raise JigLLMError("Cannot reach Ollama", "ollama", retryable=True) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            raise JigLLMError(str(e), "ollama", status_code=status) from e
        except httpx.TransportError as e:
            raise JigLLMError(str(e), "ollama", retryable=True) from e
        except Exception as e:
            if _ollama and isinstance(e, _ollama.ResponseError):
                raise JigLLMError(str(e), "ollama") from e
            raise JigLLMError(str(e), "ollama") from e

        latency_ms = timer()

        # ollama-python >= 0.4 returns a typed ``ChatResponse`` pydantic
        # model. We require that floor (see pyproject's
        # ``ollama = ["ollama>=0.4"]`` extra) and access fields via
        # attributes rather than dict ``.get()``.
        message = response.message
        tool_calls: list[ToolCall] | None = None
        raw_calls = message.tool_calls
        if raw_calls:
            try:
                tool_calls = [
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=tc.function.name,
                        arguments=parse_tool_arguments(tc.function.arguments, "ollama"),
                    )
                    for tc in raw_calls
                ]
            except (AttributeError, TypeError) as e:
                raise JigLLMError(
                    f"Malformed tool call: {e}. Raw: {raw_calls}",
                    "ollama",
                    retryable=True,
                ) from e

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
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
            content = chunk.message.content or ""
            if content:
                yield content
