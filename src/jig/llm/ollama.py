from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator

import httpx

from jig.core.errors import JigLLMError, UnsupportedResponseFormatError
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


def _translate_response_format(response_format: Any) -> dict[str, Any]:
    """Validate a portable json_schema response_format and unwrap it for Ollama.

    Mirrors smithers' OllamaExecutor translation so direct and dispatched
    Ollama calls behave identically: the ollama-python client's ``.chat()``
    takes the JSON Schema object itself in a top-level ``format`` kwarg, with
    no use for the OpenAI wrapper's ``name`` / ``description`` / ``strict``
    metadata. Only ``{"type": "json_schema", "json_schema": {"schema": {...}}}``
    is accepted; any other shape raises ``UnsupportedResponseFormatError`` so
    the caller fails before inference instead of silently running
    unconstrained.
    """
    if not isinstance(response_format, dict):
        raise UnsupportedResponseFormatError(
            f"response_format must be a mapping, got {type(response_format).__name__}"
        )
    if response_format.get("type") != "json_schema":
        raise UnsupportedResponseFormatError(
            "Ollama only supports response_format type 'json_schema', "
            f"got {response_format.get('type')!r}"
        )
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        raise UnsupportedResponseFormatError(
            "response_format.json_schema must be a mapping, got "
            f"{type(json_schema).__name__}"
        )
    schema = json_schema.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise UnsupportedResponseFormatError(
            "response_format.json_schema.schema must be a nonempty JSON "
            "object schema"
        )
    return schema


class OllamaClient(LLMClient):
    # Validated then translated (json_schema.schema becomes the top-level
    # ``format`` field) — see _translate_response_format below.
    supports_response_format = True

    def __init__(self, model: str = "llama3.1", host: str | None = None):
        if OllamaAsyncClient is None:
            raise ImportError("Install ollama: pip install 'jig[ollama]'")
        self._client = OllamaAsyncClient(host=host)
        self._model = model
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            # OllamaAsyncClient wraps an httpx.AsyncClient at ._client
            inner = getattr(self._client, "_client", None)
            if inner is not None and hasattr(inner, "aclose"):
                await inner.aclose()
            self._closed = True

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
        try:
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

            if params.response_format is not None:
                kwargs["format"] = _translate_response_format(params.response_format)
        except JigLLMError:
            raise
        except UnsupportedResponseFormatError:
            raise
        except Exception as e:
            raise JigLLMError(
                f"Request preparation failed: {type(e).__name__}: {e}",
                "ollama",
            ) from e

        timer = start_timer()

        try:
            response = await self._client.chat(**kwargs)
        except ConnectionError as e:
            raise JigLLMError("Cannot reach Ollama", "ollama", retryable=True) from e
        except httpx.ConnectError as e:
            raise JigLLMError("Cannot reach Ollama", "ollama", retryable=True) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            raise JigLLMError(str(e), "ollama", status_code=status) from e
        except httpx.TransportError as e:
            raise JigLLMError(str(e), "ollama", retryable=True) from e
        except Exception as e:
            if _ollama and isinstance(e, _ollama.ResponseError):
                status = getattr(e, "status_code", None)
                if status is None:
                    status = getattr(e, "code", None)
                raise JigLLMError(str(e), "ollama", status_code=status) from e
            raise JigLLMError(str(e), "ollama") from e

        latency_ms = timer()

        try:
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
        except JigLLMError:
            raise
        except Exception as e:
            raise JigLLMError(
                f"Response parsing failed: {type(e).__name__}: {e}",
                "ollama",
            ) from e

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
