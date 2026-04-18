"""Tests for jig.dispatch.run and Tool(dispatch=True) routing."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from jig import DispatchError, JobTimeoutError, dispatch_run
from jig.core.types import Tool, ToolCall, ToolDefinition
from jig.dispatch.client import _PollConfig
from jig.tools import ToolRegistry


def _submit_resp(job_id: str = "j-1"):
    r = MagicMock(spec=httpx.Response)
    r.status_code = 200
    r.json.return_value = {"job_id": job_id}
    r.raise_for_status = MagicMock()
    return r


def _poll_resp(status: str = "complete", result: Any = None, model: str | None = None):
    r = MagicMock(spec=httpx.Response)
    r.status_code = 200
    r.json.return_value = {
        "status": status,
        "result": result,
        "model": model,
    }
    r.raise_for_status = MagicMock()
    return r


@pytest.mark.asyncio
class TestDispatchRun:
    async def test_submit_and_return_value(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(
            status="complete",
            result={"value": [1, 2, 3]},
        )

        out = await dispatch_run(
            "ta.backtester:run_signal_study",
            {"strategy": "mean_reversion", "pair": "BTC"},
            http=http,
            poll_interval=0.01,
            poll_max_interval=0.02,
        )

        assert out == [1, 2, 3]

        # Verify submission carries task_type="function" + fn_ref + args
        submit_body = http.post.call_args.kwargs.get("json") or http.post.call_args[1]["json"]
        assert submit_body["task_type"] == "function"
        assert submit_body["payload"]["fn_ref"] == "ta.backtester:run_signal_study"
        assert submit_body["payload"]["args"] == {
            "strategy": "mean_reversion",
            "pair": "BTC",
        }

    async def test_returns_bare_result_when_no_value_wrapper(self):
        """Worker may return {} or a primitive — don't require .value key."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="complete", result={"other": 42})

        out = await dispatch_run("m:f", http=http, poll_interval=0.01)
        # Without "value", the whole result dict is returned
        assert out == {"other": 42}

    async def test_failed_raises_DispatchError(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()

        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"status": "failed", "error": "worker crashed"}
        r.raise_for_status = MagicMock()
        http.get.return_value = r

        with pytest.raises(DispatchError, match="worker crashed"):
            await dispatch_run("m:f", http=http, poll_interval=0.01)

    async def test_timeout_raises_JobTimeoutError(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="running")

        with pytest.raises(JobTimeoutError):
            await dispatch_run(
                "m:f",
                http=http,
                timeout_seconds=1,  # int, minimum
                poll_interval=0.01,
                poll_max_interval=0.02,
            )

    async def test_trace_context_in_payload(self):
        """Phase 9 will have workers read this; phase 7+8 just propagates."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="complete", result={"value": 1})

        await dispatch_run(
            "m:f",
            http=http,
            poll_interval=0.01,
            trace_context={"trace_id": "t-abc", "parent_span_id": "s-xyz"},
        )

        body = http.post.call_args.kwargs.get("json") or http.post.call_args[1]["json"]
        assert body["trace_context"] == {
            "trace_id": "t-abc",
            "parent_span_id": "s-xyz",
        }


# --- Tool(dispatch=True) routing ---


class _LocalTool(Tool):
    """Normal local-execution tool."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="local",
            description="runs locally",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return "local-output"


class _DispatchedTool(Tool):
    """Tool that should be routed through jig.dispatch.run."""

    dispatch = True

    @property
    def dispatch_fn_ref(self) -> str:
        return "ta.backtester:run_backtest"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="backtest",
            description="offloaded backtest",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        # Should never be called when dispatch=True
        raise AssertionError("execute() called for dispatched tool")


class _BrokenDispatchTool(Tool):
    """dispatch=True without dispatch_fn_ref — must be rejected at register."""

    dispatch = True

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="broken",
            description="missing fn_ref",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return ""


@pytest.mark.asyncio
class TestToolDispatchRouting:
    async def test_local_tool_executes_locally(self, monkeypatch):
        """Baseline: dispatch flag off → execute() runs locally."""
        # Fail loudly if dispatch path is taken
        from jig.dispatch import client as dc
        monkeypatch.setattr(
            dc,
            "_submit_and_poll",
            AsyncMock(side_effect=AssertionError("local tool must not dispatch")),
        )

        reg = ToolRegistry([_LocalTool()])
        result = await reg.execute(ToolCall(id="c1", name="local", arguments={}))
        assert result.output == "local-output"
        assert result.error is None

    async def test_dispatched_tool_routes_through_dispatch_run(self, monkeypatch):
        """dispatch=True → ToolRegistry calls jig.dispatch.run, not execute()."""
        captured: dict[str, Any] = {}

        async def fake_run(fn_ref, payload=None, **kwargs):
            captured["fn_ref"] = fn_ref
            captured["payload"] = payload
            return {"sharpe": 1.42, "trades": 37}

        monkeypatch.setattr("jig.dispatch.run", fake_run)
        # ToolRegistry imports run lazily via ``from jig.dispatch import
        # ... run as dispatch_run``, so patch the module attr too.
        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        reg = ToolRegistry([_DispatchedTool()])
        result = await reg.execute(ToolCall(
            id="c1", name="backtest", arguments={"pair": "BTC"},
        ))

        assert captured["fn_ref"] == "ta.backtester:run_backtest"
        assert captured["payload"] == {"pair": "BTC"}
        # Non-string results get JSON-serialized
        assert '"sharpe": 1.42' in result.output
        assert result.error is None

    async def test_dispatch_error_surfaces_as_tool_error(self, monkeypatch):
        """DispatchError becomes ToolResult.error — agent loop can recover."""
        async def boom(fn_ref, payload=None, **kwargs):
            raise DispatchError("worker not found")

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", boom)

        reg = ToolRegistry([_DispatchedTool()])
        result = await reg.execute(ToolCall(id="c1", name="backtest", arguments={}))

        assert result.output == ""
        assert "worker not found" in result.error

    async def test_register_rejects_dispatch_without_fn_ref(self):
        """register() fails fast if the tool is misconfigured."""
        with pytest.raises(ValueError, match="dispatch_fn_ref is None"):
            ToolRegistry([_BrokenDispatchTool()])


# --- DispatchClient tool-use payload ---


class TestDispatchClientToolPayload:
    """Phase 7+8: DispatchClient no longer rejects tools; they pass through."""

    def test_tool_calls_in_assistant_history_reserialized(self):
        """When the agent has already called a tool, the assistant turn's
        tool_calls must round-trip through the dispatch payload in the
        OpenAI-compatible shape the worker parses."""
        from jig.core.types import CompletionParams, Message, Role
        from jig.llm.dispatch import DispatchClient

        client = DispatchClient(model="llama-70b")
        params = CompletionParams(
            messages=[
                Message(role=Role.USER, content="do it"),
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(
                        id="tc-1",
                        name="echo",
                        arguments={"text": "hi"},
                    )],
                ),
                Message(
                    role=Role.TOOL,
                    content="hi",
                    tool_call_id="tc-1",
                ),
            ],
        )
        payload = client._build_payload(params)

        # Assistant turn carries tool_calls in OpenAI shape
        assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["function"]["name"] == "echo"
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"text": "hi"}'
        # Tool turn carries tool_call_id
        tool_msg = next(m for m in payload["messages"] if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "tc-1"
