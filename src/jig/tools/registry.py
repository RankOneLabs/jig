from __future__ import annotations

import asyncio
import json

from jig.core.types import Tool, ToolCall, ToolDefinition, ToolResult


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
        use :func:`jig.dispatch.run`'s default (``http://willie:8900``).
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
            return ToolResult(call_id=call.id, output="", error=f"Unknown tool: {call.name}")

        if getattr(tool, "dispatch", False):
            return await self._execute_dispatched(tool, call)

        try:
            if self._execute_timeout is not None:
                output = await asyncio.wait_for(
                    tool.execute(call.arguments), timeout=self._execute_timeout
                )
            else:
                output = await tool.execute(call.arguments)
            return ToolResult(call_id=call.id, output=output)
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.id,
                output="",
                error=f"Tool {call.name} timed out after {self._execute_timeout}s",
            )
        except Exception as e:
            return ToolResult(call_id=call.id, output="", error=str(e))

    async def _execute_dispatched(self, tool: Tool, call: ToolCall) -> ToolResult:
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
            kwargs["timeout_seconds"] = max(1, int(self._execute_timeout))

        try:
            if self._execute_timeout is not None:
                result = await asyncio.wait_for(
                    dispatch_run(
                        tool.dispatch_fn_ref,  # type: ignore[arg-type]
                        call.arguments,
                        **kwargs,
                    ),
                    timeout=self._execute_timeout,
                )
            else:
                result = await dispatch_run(
                    tool.dispatch_fn_ref,  # type: ignore[arg-type]
                    call.arguments,
                    **kwargs,
                )
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.id,
                output="",
                error=f"Dispatched tool {call.name} timed out after {self._execute_timeout}s",
            )
        except DispatchError as e:
            return ToolResult(
                call_id=call.id,
                output="",
                error=f"Dispatch failed for {call.name}: {e}",
            )
        except Exception as e:
            return ToolResult(call_id=call.id, output="", error=str(e))

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
