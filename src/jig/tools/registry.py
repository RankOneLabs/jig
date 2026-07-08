from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from typing import Any

from jig.core.types import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    current_tool_context,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        execute_timeout: float | None = None,
        dispatch_url: str | None = None,
    ):
        """Register tools for agent use.

        ``dispatch_url`` overrides the default smithers endpoint for
        any registered tool with ``dispatch=True``. Leave as None to
        use :func:`jig.dispatch.run`'s default (the ``JIG_DISPATCH_URL``
        environment variable, or ``http://localhost:8900``).
        """
        if execute_timeout is not None and execute_timeout <= 0:
            raise ValueError(f"execute_timeout must be > 0 when provided, got {execute_timeout}")
        self._tools: dict[str, Tool] = {}
        self._execute_timeout = execute_timeout
        self._dispatch_url = dispatch_url
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        # getattr-with-default so duck-typed tools (tests, third-party
        # callers that haven't migrated to the Tool base) still register.
        if getattr(tool, "dispatch", False) and not getattr(tool, "dispatch_fn_ref", None):
            raise ValueError(
                f"Tool {tool.definition.name!r} has dispatch=True but "
                f"dispatch_fn_ref is None. Override dispatch_fn_ref to "
                f"point at a smithers-registered function."
            )
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if not tool:
            logger.debug("tool.execute unknown name=%s", call.name)
            return ToolResult(call_id=call.id, output="", error=f"Unknown tool: {call.name}")

        tool_context = current_tool_context.get()
        token = current_tool_context.set(tool_context)
        try:
            if getattr(tool, "dispatch", False):
                logger.debug("tool.execute dispatched name=%s", call.name)
                return await self._execute_dispatched(tool, call, tool_context)

            logger.debug("tool.execute start name=%s timeout=%s", call.name, self._execute_timeout)
            try:
                execute_with_context = getattr(tool, "execute_with_context", None)
                if callable(execute_with_context):
                    execute_fn = execute_with_context
                    args: tuple[object, ...] = (call.arguments, tool_context)
                else:
                    execute_fn = tool.execute
                    args = (call.arguments,)

                if self._execute_timeout is not None:
                    output = await asyncio.wait_for(
                        execute_fn(*args), timeout=self._execute_timeout
                    )
                else:
                    output = await execute_fn(*args)
                logger.debug("tool.execute done name=%s", call.name)
                return ToolResult(call_id=call.id, output=output)
            except asyncio.TimeoutError:
                logger.warning("tool.execute timeout name=%s after=%ss", call.name, self._execute_timeout)
                return ToolResult(
                    call_id=call.id,
                    output="",
                    error=f"TimeoutError: Tool {call.name} timed out after {self._execute_timeout}s",
                )
            except Exception as e:
                msg = str(e)
                error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
                logger.warning("tool.execute error name=%s err=%s", call.name, error, exc_info=True)
                return ToolResult(call_id=call.id, output="", error=error)
        finally:
            current_tool_context.reset(token)

    async def _execute_dispatched(
        self,
        tool: Tool,
        call: ToolCall,
        tool_context: ToolExecutionContext | None,
    ) -> ToolResult:
        """Route a dispatch=True tool through :func:`jig.dispatch.run`.

        Dispatch failures surface as ``ToolResult.error`` so the agent
        loop can react the same way it would to a local tool error —
        the model sees the error, may recover, or the loop exits at
        max_tool_calls. Imported lazily to avoid a runtime dep on httpx
        for callers who don't use dispatch.
        """
        from jig.dispatch import DispatchError, run as dispatch_run

        kwargs: dict[str, object] = {}
        if self._dispatch_url is not None:
            kwargs["dispatch_url"] = self._dispatch_url
        # Plumb the registry timeout through so the dispatch-side poll
        # loop uses the same budget the local wait_for enforces. Without
        # this, ``dispatch_run``'s default 300s cap would clip any
        # registry timeout larger than 5 minutes. ``wait_for`` is still
        # kept as a belt-and-suspenders guard — if the worker hangs on
        # a non-poll branch, we still exit on time.
        if self._execute_timeout is not None:
            kwargs["timeout_seconds"] = max(1, math.ceil(self._execute_timeout))
        if tool_context is not None:
            kwargs["trace_context"] = {
                "trace_id": tool_context.trace_id,
                "parent_span_id": tool_context.span_id,
            }

        try:
            payload = dict(call.arguments)
            payload.update(self._dispatch_payload_extra(tool, call, tool_context))

            if self._execute_timeout is not None:
                result = await asyncio.wait_for(
                    dispatch_run(
                        tool.dispatch_fn_ref,  # type: ignore[arg-type]
                        payload,
                        **kwargs,
                    ),
                    timeout=self._execute_timeout,
                )
            else:
                result = await dispatch_run(
                    tool.dispatch_fn_ref,  # type: ignore[arg-type]
                    payload,
                    **kwargs,
                )
        except asyncio.TimeoutError:
            logger.warning("tool.execute dispatch timeout name=%s after=%ss", call.name, self._execute_timeout)
            return ToolResult(
                call_id=call.id,
                output="",
                error=f"TimeoutError: Dispatched tool {call.name} timed out after {self._execute_timeout}s",
            )
        except DispatchError as e:
            msg = str(e)
            error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
            logger.warning("tool.execute dispatch error name=%s err=%s", call.name, error)
            return ToolResult(
                call_id=call.id,
                output="",
                error=error,
            )
        except Exception as e:
            msg = str(e)
            error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
            logger.warning("tool.execute dispatch error name=%s err=%s", call.name, error, exc_info=True)
            return ToolResult(call_id=call.id, output="", error=error)

        # Workers can return any JSON-serializable shape; coerce to string
        # so the agent loop's Message(role=TOOL) content is always a string.
        if isinstance(result, str):
            output = result
        else:
            try:
                output = json.dumps(result)
            except (TypeError, ValueError):
                output = str(result)
        return ToolResult(call_id=call.id, output=output)

    def _dispatch_payload_extra(
        self,
        tool: Tool,
        call: ToolCall,
        tool_context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        dispatch_payload_extra = getattr(tool, "dispatch_payload_extra", None)
        if not callable(dispatch_payload_extra):
            return {}

        sig = inspect.signature(dispatch_payload_extra)
        params = list(sig.parameters.values())
        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params
        )

        if not params:
            extra = dispatch_payload_extra()
        elif has_var_kwargs:
            kwargs: dict[str, object] = {"arguments": call.arguments}
            first = params[0]
            if first.kind != inspect.Parameter.VAR_KEYWORD:
                kwargs[first.name] = tool_context
            else:
                kwargs["context"] = tool_context
            extra = dispatch_payload_extra(**kwargs)
        elif params[0].kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs = {params[0].name: tool_context}
            if len(params) >= 2:
                kwargs[params[1].name] = call.arguments
            extra = dispatch_payload_extra(**kwargs)
        elif len(params) >= 2:
            second = params[1]
            if second.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                extra = dispatch_payload_extra(
                    tool_context, **{second.name: call.arguments}
                )
            else:
                extra = dispatch_payload_extra(tool_context, call.arguments)
        else:
            extra = dispatch_payload_extra(tool_context)

        if not isinstance(extra, dict):
            return {}
        return {key: value for key, value in extra.items() if value is not None}
