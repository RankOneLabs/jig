from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any

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
from jig.llm.pricing import stamp_cost

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _sanitize_for_gemini(obj: Any) -> Any:
    """Sanitize a value so it passes google-genai FunctionResponse validation.

    The google-genai SDK uses Pydantic to validate FunctionResponse.response,
    which rejects None values, NaN/Inf floats, and non-primitive types.
    """
    if obj is None:
        return ""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_gemini(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_gemini(v) for v in obj]
    return obj


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-pro", **client_kwargs: Any):
        if genai is None:
            raise ImportError("Install google-genai: pip install 'jig[google]'")
        self._client = genai.Client(**client_kwargs)
        self._model = model

    def _convert_messages(self, params: CompletionParams) -> list[Any]:
        contents: list[genai_types.Content] = []
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
            elif msg.role == Role.USER:
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=msg.content)],
                ))
            elif msg.role == Role.ASSISTANT:
                parts: list[genai_types.Part] = []
                if msg.content:
                    parts.append(genai_types.Part(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        ))
                contents.append(genai_types.Content(role="model", parts=parts))
            elif msg.role == Role.TOOL:
                tool_name = _find_tool_name(params.messages, msg.tool_call_id)
                try:
                    result = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    result = {"result": msg.content}

                result = _sanitize_for_gemini(result)

                try:
                    fr = genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response=result,
                        )
                    )
                except Exception:
                    logger.warning(
                        "FunctionResponse validation failed for tool %s, "
                        "falling back to string representation",
                        tool_name,
                        exc_info=True,
                    )
                    fr = genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response={"result": str(msg.content)},
                        )
                    )
                # Gemini expects function_response parts in a user turn —
                # merge consecutive tool results into one Content block.
                if contents and contents[-1].role == "user" and any(
                    p.function_response for p in contents[-1].parts
                ):
                    contents[-1].parts.append(fr)
                else:
                    contents.append(genai_types.Content(role="user", parts=[fr]))

        return contents

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[Any]:
        declarations = [
            genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in tools
        ]
        return [genai_types.Tool(function_declarations=declarations)]

    async def complete(self, params: CompletionParams) -> LLMResponse:
        contents = self._convert_messages(params)

        config_kwargs: dict[str, Any] = {}
        if params.system:
            config_kwargs["system_instruction"] = params.system
        if params.tools:
            config_kwargs["tools"] = self._convert_tools(params.tools)
        if params.temperature is not None:
            config_kwargs["temperature"] = params.temperature
        if params.max_tokens is not None:
            config_kwargs["max_output_tokens"] = params.max_tokens
        if params.provider_params:
            config_kwargs.update(params.provider_params)

        config = genai_types.GenerateContentConfig(**config_kwargs)

        start = time.time()

        async def _call() -> Any:
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        def _retryable(e: Exception) -> bool:
            # Retry on 429 / resource exhausted
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            return status in (429, 503)

        try:
            response = await with_retry(_call, max_attempts=3, retryable=_retryable)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            raise JigLLMError(str(e), "google", status_code=status) from e

        latency_ms = (time.time() - start) * 1000

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts or []
            for part in parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                    ))

        usage_meta = response.usage_metadata
        input_tokens = usage_meta.prompt_token_count if usage_meta else 0
        output_tokens = usage_meta.candidates_token_count if usage_meta else 0

        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        stamp_cost(usage, self._model)
        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls or None,
            usage=usage,
            latency_ms=latency_ms,
            model=self._model,
        )


def _find_tool_name(messages: list[Message], tool_call_id: str | None) -> str:
    """Find the tool name for a given tool_call_id by scanning prior assistant messages."""
    if not tool_call_id:
        return "unknown"
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id == tool_call_id:
                    return tc.name
    return "unknown"
