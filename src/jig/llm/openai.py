from __future__ import annotations

import json
import logging
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
from jig.llm._common import (
    merge_completion_kwargs,
    openai_tool_payload,
    start_timer,
    wrap_llm_error,
)
from jig.llm._parsing import parse_tool_arguments
from jig.llm.pricing import stamp_cost

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


class OpenAIClient(LLMClient):
    # Provider label stamped onto JigLLMError. Subclasses that point the
    # OpenAI SDK at a different backend (e.g. OpenRouter) override this so
    # error telemetry attributes failures correctly.
    _provider_label: str = "openai"

    def __init__(self, model: str = "gpt-4o", **client_kwargs: Any):
        if openai is None:
            raise ImportError("Install openai: pip install 'jig[openai]'")
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model

    def _apply_extra_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Subclass hook: inject defaults into the chat.completions.create()
        kwargs dict. Mutates ``kwargs`` in place rather than returning a dict
        to update with, so subclasses can deep-merge nested fields like
        ``extra_body`` instead of replacing caller-supplied values wholesale.
        Default is a no-op.
        """
        return None

    def _inline_cost(self, response: Any) -> float | None:
        """Subclass hook: cost reported by the upstream response, if any.

        Returning a float wins over the local pricing table (used by gateways
        like OpenRouter that bill per-call and return the exact charge).
        """
        return None

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
        return openai_tool_payload(tools)

    async def complete(self, params: CompletionParams) -> LLMResponse:
        messages = self._convert_messages(params)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if params.tools:
            kwargs["tools"] = self._convert_tools(params.tools)
        merge_completion_kwargs(kwargs, params)
        self._apply_extra_kwargs(kwargs)

        timer = start_timer()
        logger.debug(
            "%s.complete request model=%s messages=%d tools=%d",
            self._provider_label, self._model, len(messages),
            len(kwargs.get("tools") or ()),
        )

        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        def _retryable(e: Exception) -> bool:
            if openai is None:
                return False
            return isinstance(e, openai.RateLimitError)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except Exception as e:
            err = wrap_llm_error(e, self._provider_label)
            logger.debug(
                "%s.complete failed model=%s status=%s err=%s",
                self._provider_label, self._model, err.status_code, e,
            )
            raise err from e

        latency_ms = timer()
        logger.debug(
            "%s.complete response model=%s latency_ms=%.0f choices=%d",
            self._provider_label, self._model, latency_ms,
            len(response.choices or ()),
        )

        if not response.choices:
            # Gateways (notably OpenRouter) sometimes return 200 OK with a
            # null/empty ``choices`` array when the upstream provider
            # errored after the request was accepted. Surface the response's
            # ``error`` payload if present and mark retryable — these are
            # typically transient and the agent loop should try again rather
            # than crash with TypeError on ``choices[0]``.
            #
            # The OpenAI SDK's pydantic model for ChatCompletion doesn't
            # declare an ``error`` field, so OpenRouter's payload lands in
            # ``response.model_extra`` rather than as a real attribute —
            # mirror the ``usage.cost`` handling in OpenRouterClient and
            # check both locations.
            err = getattr(response, "error", None)
            if err is None:
                extra = getattr(response, "model_extra", None) or {}
                err = extra.get("error")
            detail = ""
            if err is not None:
                err_msg = None
                if isinstance(err, dict):
                    err_msg = err.get("message")
                else:
                    err_msg = getattr(err, "message", None)
                if err_msg:
                    detail = f": {err_msg}"
            raise JigLLMError(
                f"upstream returned no choices{detail}",
                self._provider_label,
                retryable=True,
            )

        choice = response.choices[0].message

        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_arguments(tc.function.arguments, "openai"),
                )
                for tc in choice.tool_calls
            ]

        usage = Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        inline_cost = self._inline_cost(response)
        if inline_cost is not None:
            usage.cost = inline_cost
        stamp_cost(usage, response.model)
        return LLMResponse(
            content=choice.content or "",
            tool_calls=tool_calls,
            usage=usage,
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
        merge_completion_kwargs(kwargs, params)
        self._apply_extra_kwargs(kwargs)

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
