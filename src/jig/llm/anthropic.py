from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from jig.core.errors import JigLLMError
from jig.core.retry import with_retry
from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
)
from jig.llm._common import merge_completion_kwargs, start_timer, wrap_llm_error
from jig.llm.pricing import stamp_cost

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514", **client_kwargs: Any):
        if anthropic is None:
            raise ImportError("Install anthropic: pip install 'jig[anthropic]'")
        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._model = model

    def _convert_messages(self, params: CompletionParams) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.TOOL:
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"].append(
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.tool_call_id,
                                    "content": msg.content,
                                }
                            ],
                        }
                    )
            elif msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": msg.role.value, "content": msg.content})
        return messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    async def complete(self, params: CompletionParams) -> LLMResponse:
        try:
            messages = self._convert_messages(params)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": params.max_tokens or 4096,
            }
            if params.system:
                kwargs["system"] = params.system
            if params.tools:
                kwargs["tools"] = self._convert_tools(params.tools)
            merge_completion_kwargs(kwargs, params)
        except JigLLMError:
            raise
        except Exception as e:
            raise JigLLMError(
                f"Request preparation failed: {type(e).__name__}: {e}",
                "anthropic",
            ) from e

        timer = start_timer()

        async def _call() -> Any:
            return await self._client.messages.create(**kwargs)

        def _retryable(e: Exception) -> bool:
            if anthropic is None:
                return False
            return isinstance(e, anthropic.RateLimitError)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except Exception as e:
            raise wrap_llm_error(e, "anthropic") from e

        latency_ms = timer()

        try:
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input)
                    )

            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            stamp_cost(usage, response.model)
            return LLMResponse(
                content="\n".join(text_parts),
                tool_calls=tool_calls or None,
                usage=usage,
                latency_ms=latency_ms,
                model=response.model,
            )
        except JigLLMError:
            raise
        except Exception as e:
            raise JigLLMError(
                f"Response parsing failed: {type(e).__name__}: {e}",
                "anthropic",
            ) from e

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        messages = self._convert_messages(params)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": params.max_tokens or 4096,
        }
        if params.system:
            kwargs["system"] = params.system
        if params.tools:
            kwargs["tools"] = self._convert_tools(params.tools)
        merge_completion_kwargs(kwargs, params)

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
