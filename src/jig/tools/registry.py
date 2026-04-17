from __future__ import annotations

import asyncio

from jig.core.types import Tool, ToolCall, ToolDefinition, ToolResult


class ToolRegistry:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        execute_timeout: float | None = None,
    ):
        self._tools: dict[str, Tool] = {}
        self._execute_timeout = execute_timeout
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(call_id=call.id, output="", error=f"Unknown tool: {call.name}")
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
