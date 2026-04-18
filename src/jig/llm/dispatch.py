"""Smithers dispatch adapter — routes inference to the Springfield homelab fleet.

Thin wrapper around :func:`jig.dispatch.client._submit_and_poll` that
translates jig's :class:`CompletionParams` into the ``inference`` task
payload (and parses results back into :class:`LLMResponse`). Tool use
is supported — adapters serialize tools in the OpenAI-compatible shape
smithers executors expect, and parse ``tool_calls`` out of the response.
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any

import httpx

from jig.core.errors import JigLLMError
from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Role,
    ToolCall,
    ToolDefinition,
    TraceContext,
    Usage,
)
from jig.dispatch.client import (
    DispatchError,
    JobTimeoutError,
    _PollConfig,
    _submit_and_poll,
)

logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> int:
    """Coerce arbitrary JSON values to int, defaulting to 0 on None/garbage.

    Smithers is a moving target and tokens may land as null, strings, or
    missing entirely; we don't want that to crash the caller's agent loop.
    """
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_cost(value: Any) -> float | None:
    """Coerce cost to float, preserving None for 'unknown'.

    ``None`` semantically means the backend didn't report spend — that's
    distinct from a confirmed free call (``0.0``), and ``BudgetTracker``
    deliberately ignores the former. Non-finite floats (``NaN``/``Inf``,
    possibly from ``"nan"``/``"inf"`` string inputs) are treated as
    unknown rather than propagated, because they silently defeat budget
    comparisons downstream.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _tools_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Serialize :class:`ToolDefinition` list to the OpenAI-style shape
    smithers executors (vLLM, Ollama) accept natively."""
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


def _parse_tool_calls(raw: Any) -> list[ToolCall] | None:
    """Parse tool_calls from smithers result into jig's :class:`ToolCall`.

    Workers return the list in OpenAI shape::

        [{"id": "...", "function": {"name": "...", "arguments": "{...}"}}]

    where ``arguments`` is a JSON-encoded string. Malformed entries are
    skipped with a warning; we'd rather lose one tool call than crash
    the whole completion.
    """
    if not raw or not isinstance(raw, list):
        return None
    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args_raw = fn.get("arguments")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed tool-call arguments from dispatch: %r", args_raw,
                )
                continue
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        calls.append(ToolCall(
            id=entry.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=args,
        ))
    return calls or None


class DispatchClient(LLMClient):
    """jig LLMClient that submits inference jobs to a smithers dispatch server.

    Three levels of specificity:
        DispatchClient()                              # router picks model + machine
        DispatchClient(model="llama-70b")             # router picks machine
        DispatchClient(model="llama-70b", machine="mcbain")  # explicit

    Tool use is supported: when :class:`CompletionParams` carries
    ``tools``, they're serialized into the smithers payload and
    ``tool_calls`` are parsed back from the worker's response. Workers
    must be running an executor (vLLM, recent Ollama) whose backend
    model supports structured tool calling.

    Pass ``trace_context`` to propagate the caller's trace identity —
    phase 9 wires workers to reparent their spans under it.
    """

    def __init__(
        self,
        model: str | None = None,
        machine: str | None = None,
        dispatch_url: str = "http://willie:8900",
        requester: str = "jig",
        timeout_seconds: int = 300,
        poll_interval: float = 0.5,
        poll_max_interval: float = 5.0,
        trace_context: TraceContext | None = None,
    ) -> None:
        self._model = model
        self._machine = machine
        self._dispatch_url = dispatch_url.rstrip("/")
        self._requester = requester
        self._trace_context = trace_context
        self._poll_config = _PollConfig(
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            poll_max_interval=poll_max_interval,
        )
        self._http = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient. Safe to call multiple times."""
        await self._http.aclose()

    def _build_payload(self, params: CompletionParams) -> dict[str, Any]:
        """Convert CompletionParams to smithers executor payload format."""
        messages: list[dict[str, Any]] = []
        if params.system:
            messages.append({"role": "system", "content": params.system})
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
            m: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.tool_call_id is not None:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                # Echo the assistant's prior tool calls back in the shape
                # smithers workers parse (OpenAI-compatible JSON string args).
                m["tool_calls"] = [
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
            messages.append(m)

        payload: dict[str, Any] = {"messages": messages}
        if params.tools:
            payload["tools"] = _tools_payload(params.tools)
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.provider_params:
            payload.update(params.provider_params)
        return payload

    async def complete(self, params: CompletionParams) -> LLMResponse:
        payload = self._build_payload(params)
        start = time.time()
        try:
            data = await _submit_and_poll(
                http=self._http,
                dispatch_url=self._dispatch_url,
                task_type="inference",
                payload=payload,
                requester=self._requester,
                model=self._model,
                machine=self._machine,
                trace_context=(
                    self._trace_context.to_dict()
                    if self._trace_context is not None else None
                ),
                poll_config=self._poll_config,
            )
        except JobTimeoutError as e:
            raise JigLLMError(
                str(e), "dispatch", retryable=True,
            ) from e
        except DispatchError as e:
            raise JigLLMError(str(e), "dispatch") from e

        latency_ms = (time.time() - start) * 1000
        result = data.get("result") or {}
        content = result.get("content", "")
        model = data.get("model") or self._model or "dispatch"
        raw_usage = result.get("usage")
        usage_data = raw_usage if isinstance(raw_usage, dict) else {}
        tool_calls = _parse_tool_calls(result.get("tool_calls"))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=_safe_int(usage_data.get("input_tokens")),
                output_tokens=_safe_int(usage_data.get("output_tokens")),
                cost=_safe_cost(usage_data.get("cost")),
            ),
            latency_ms=latency_ms,
            model=model,
        )
