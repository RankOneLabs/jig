"""Tests for jig.dispatch.run and Tool(dispatch=True) routing."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jig import DispatchError, JobTimeoutError, dispatch_run
from jig.core.errors import JigLLMError
from jig.core.types import Tool, ToolCall, ToolDefinition
from jig.dispatch import (
    cancel_job,
    cancel_or_fence,
    get_job,
    get_job_by_idempotency_key,
)
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


def _cancel_resp(status_code: int = 200):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    return r


def _job_api_resp(data: Any, status_code: int = 200):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.text = str(data)
    r.json.return_value = data
    if status_code >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            "job API error", request=MagicMock(), response=r,
        )
    else:
        r.raise_for_status = MagicMock()
    return r


@pytest.mark.asyncio
class TestPublicJobAPI:
    async def test_get_job_returns_wire_response(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        expected = {"job_id": "j-1", "status": "running", "machine": "frink"}
        http.get.return_value = _job_api_resp(expected)

        result = await get_job(
            "j-1", dispatch_url="http://localhost:8900/", http=http,
        )

        assert result == expected
        http.get.assert_awaited_once_with("http://localhost:8900/jobs/j-1")

    async def test_get_job_maps_missing_job(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _job_api_resp(
            {"detail": "Job not found"}, status_code=404,
        )

        with pytest.raises(DispatchError) as raised:
            await get_job("missing", http=http)

        assert raised.value.job_id == "missing"
        assert raised.value.status == "not_found"
        assert raised.value.retryable is False

    async def test_cancel_job_returns_acknowledged_state(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        expected = {"job_id": "j-1", "status": "cancelled"}
        http.delete.return_value = _job_api_resp(expected)

        result = await cancel_job(
            "j-1", dispatch_url="http://localhost:8900", http=http,
        )

        assert result == expected
        http.delete.assert_awaited_once_with("http://localhost:8900/jobs/j-1")

    @pytest.mark.parametrize(
        ("code", "status", "retryable"),
        [(404, "not_found", False), (409, "already_terminal", False), (503, None, True)],
    )
    async def test_cancel_job_preserves_smithers_failure_semantics(
        self, code, status, retryable,
    ):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.delete.return_value = _job_api_resp({"detail": "no"}, status_code=code)

        with pytest.raises(DispatchError) as raised:
            await cancel_job("j-1", http=http)

        assert raised.value.status == status
        assert raised.value.retryable is retryable

    async def test_get_job_by_idempotency_key_encodes_one_path_segment(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        expected = {"job_id": "j-1", "status": "running"}
        http.get.return_value = _job_api_resp(expected)

        result = await get_job_by_idempotency_key(
            "attempt/7 retry", dispatch_url="http://localhost:8900/", http=http,
        )

        assert result == expected
        http.get.assert_awaited_once_with(
            "http://localhost:8900/jobs/by-idempotency/attempt%2F7%20retry",
        )

    async def test_get_job_by_idempotency_key_maps_not_found(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _job_api_resp(
            {"detail": "Job not found"}, status_code=404,
        )

        with pytest.raises(DispatchError) as raised:
            await get_job_by_idempotency_key("missing", http=http)

        assert raised.value.status == "not_found"
        assert raised.value.retryable is False

    @pytest.mark.parametrize("code", [200, 409])
    async def test_cancel_or_fence_returns_durable_fence_outcome(self, code):
        http = AsyncMock(spec=httpx.AsyncClient)
        expected = {"status": "cancelled", "job_id": "j-1"}
        http.delete.return_value = _job_api_resp(expected, status_code=code)

        result = await cancel_or_fence(
            "attempt/7", dispatch_url="http://localhost:8900", http=http,
        )

        assert result == expected
        http.delete.assert_awaited_once_with(
            "http://localhost:8900/jobs/by-idempotency/attempt%2F7",
            timeout=10.0,
        )

    async def test_cancel_or_fence_is_bounded_without_a_client_default(self):
        """An explicit timeout, so a client built with ``timeout=None`` cannot
        leave a stalled fence holding the caller's exception indefinitely."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.delete.return_value = _job_api_resp({"status": "cancelled"})

        await cancel_or_fence("attempt-7", http=http)

        assert http.delete.await_args.kwargs["timeout"] is not None

    async def test_cancel_or_fence_503_is_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.delete.return_value = _job_api_resp(
            {"detail": "Worker cancellation was not acknowledged"},
            status_code=503,
        )

        with pytest.raises(DispatchError) as raised:
            await cancel_or_fence("attempt-7", http=http)

        assert raised.value.retryable is True


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
        http.delete.return_value = _cancel_resp()

        with pytest.raises(JobTimeoutError):
            await dispatch_run(
                "m:f",
                http=http,
                timeout_seconds=1,  # int, minimum
                cleanup_grace_seconds=0,
                poll_interval=0.01,
                poll_max_interval=0.02,
            )

        http.delete.assert_awaited_once_with(
            "http://localhost:8900/jobs/j-1", timeout=10.0,
        )

    async def test_durable_timeout_can_leave_remote_job_running(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="running")

        with pytest.raises(JobTimeoutError):
            await dispatch_run(
                "m:f",
                http=http,
                timeout_seconds=0.02,
                cleanup_grace_seconds=0,
                cancel_on_timeout=False,
                poll_interval=0.01,
                poll_max_interval=0.01,
            )

        http.delete.assert_not_awaited()

    async def test_repeated_caller_cancellation_does_not_abandon_remote_cancel(self):
        """Once the caller is cancelled mid-wait the remote cancellation is
        drained, not merely shielded — a second cancel() must not detach it
        and leave the worker slot occupied behind a retry."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="running")

        cancel_started = asyncio.Event()
        cancel_landed = False

        async def slow_cancel(*args, **kwargs):
            nonlocal cancel_landed
            cancel_started.set()
            await asyncio.sleep(0.05)
            cancel_landed = True
            return _job_api_resp({"job_id": "j-1", "status": "cancelled"})

        http.delete.side_effect = slow_cancel

        task = asyncio.create_task(
            dispatch_run("m:f", http=http, poll_interval=0.01),
        )
        await asyncio.sleep(0.03)
        task.cancel()
        await cancel_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert cancel_landed

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

    async def test_idempotency_key_in_submission(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="complete", result={"value": 1})

        await dispatch_run(
            "m:f",
            http=http,
            idempotency_key="validation:workflow-1",
            poll_interval=0.01,
        )

        body = http.post.call_args.kwargs["json"]
        assert body["idempotency_key"] == "validation:workflow-1"

    async def test_idempotency_key_is_optional(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="complete", result={"value": 1})

        await dispatch_run("m:f", http=http, poll_interval=0.01)

        body = http.post.call_args.kwargs["json"]
        assert "idempotency_key" not in body

    async def test_retry_after_response_loss_polls_existing_job(self):
        """A repeated key lets smithers return the first accepted job."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.side_effect = [
            httpx.ReadError("response lost"),
            _submit_resp("j-existing"),
        ]
        http.get.return_value = _poll_resp(
            status="complete", result={"value": "recovered"},
        )

        kwargs = {
            "http": http,
            "idempotency_key": "validation:workflow-1",
            "poll_interval": 0.01,
        }
        with pytest.raises(DispatchError) as exc_info:
            await dispatch_run("m:f", **kwargs)
        assert exc_info.value.retryable is True

        assert await dispatch_run("m:f", **kwargs) == "recovered"
        assert [call.kwargs["json"]["idempotency_key"] for call in http.post.await_args_list] == [
            "validation:workflow-1",
            "validation:workflow-1",
        ]
        http.get.assert_awaited_once_with("http://localhost:8900/jobs/j-existing")


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


# --- PR #19 review fixes ---


@pytest.mark.asyncio
class TestDispatchErrorRetryable:
    """DispatchError carries a ``retryable`` flag so callers can tell
    transient submission failures from terminal worker outcomes."""

    async def test_connect_error_is_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.side_effect = httpx.ConnectError("unreachable")

        with pytest.raises(DispatchError) as exc:
            await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert exc.value.retryable is True

    async def test_5xx_is_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        r = MagicMock(spec=httpx.Response)
        r.status_code = 503
        r.text = "overloaded"
        r.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=r),
        )
        http.post.return_value = r

        with pytest.raises(DispatchError) as exc:
            await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert exc.value.retryable is True

    async def test_4xx_is_not_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        r = MagicMock(spec=httpx.Response)
        r.status_code = 400
        r.text = "bad task_type"
        r.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("400", request=MagicMock(), response=r),
        )
        http.post.return_value = r

        with pytest.raises(DispatchError) as exc:
            await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert exc.value.retryable is False

    async def test_worker_failed_is_not_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"status": "failed", "error": "boom"}
        r.raise_for_status = MagicMock()
        http.get.return_value = r

        with pytest.raises(DispatchError) as exc:
            await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert exc.value.retryable is False

    async def test_timeout_is_retryable(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(status="running")
        http.delete.return_value = _cancel_resp()

        with pytest.raises(JobTimeoutError) as exc:
            await dispatch_run(
                "m:f", http=http, timeout_seconds=1,
                cleanup_grace_seconds=0,
                poll_interval=0.01, poll_max_interval=0.02,
            )
        assert exc.value.retryable is True


@pytest.mark.asyncio
class TestDispatchClientRetryablePropagation:
    """JigLLMError.retryable must mirror the underlying DispatchError."""

    async def test_retryable_connect_surfaces_as_retryable_jig_error(self, monkeypatch):
        from jig.core.types import CompletionParams, Message, Role
        from jig.llm import dispatch as llm_dispatch

        async def boom(**_):
            raise DispatchError("unreachable", retryable=True)

        monkeypatch.setattr(llm_dispatch, "_submit_and_poll", boom)
        client = llm_dispatch.DispatchClient(model="llama-70b")
        try:
            with pytest.raises(JigLLMError) as exc:
                await client.complete(CompletionParams(
                    messages=[Message(role=Role.USER, content="hi")],
                ))
            assert exc.value.retryable is True
        finally:
            await client.aclose()

    async def test_terminal_dispatch_error_stays_terminal(self, monkeypatch):
        from jig.core.types import CompletionParams, Message, Role
        from jig.llm import dispatch as llm_dispatch

        async def boom(**_):
            raise DispatchError("worker failed", retryable=False)

        monkeypatch.setattr(llm_dispatch, "_submit_and_poll", boom)
        client = llm_dispatch.DispatchClient(model="llama-70b")
        try:
            with pytest.raises(JigLLMError) as exc:
                await client.complete(CompletionParams(
                    messages=[Message(role=Role.USER, content="hi")],
                ))
            assert exc.value.retryable is False
        finally:
            await client.aclose()


@pytest.mark.asyncio
class TestFalsyResultPreserved:
    """``dispatch_run`` must not collapse 0 / False / [] / '' into {}."""

    @pytest.mark.parametrize("falsy", [0, False, [], "", 0.0])
    async def test_falsy_value_survives(self, falsy):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.get.return_value = _poll_resp(
            status="complete", result={"value": falsy},
        )
        out = await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert out == falsy
        assert type(out) is type(falsy)

    async def test_missing_result_returns_none(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"status": "complete"}  # no "result" key
        r.raise_for_status = MagicMock()
        http.get.return_value = r

        out = await dispatch_run("m:f", http=http, poll_interval=0.01)
        assert out is None


class TestTraceContextFromDict:
    """TraceContext.from_dict must never raise on malformed input."""

    def test_none_returns_none(self):
        from jig.core.types import TraceContext
        assert TraceContext.from_dict(None) is None

    def test_non_dict_returns_none(self):
        from jig.core.types import TraceContext
        for bad in ["str", 42, [1, 2], True]:
            assert TraceContext.from_dict(bad) is None

    def test_missing_fields_returns_none(self):
        from jig.core.types import TraceContext
        assert TraceContext.from_dict({}) is None
        assert TraceContext.from_dict({"trace_id": "t"}) is None

    def test_non_string_fields_returns_none(self):
        from jig.core.types import TraceContext
        assert TraceContext.from_dict(
            {"trace_id": 1, "parent_span_id": "s"},
        ) is None

    def test_valid_dict_round_trips(self):
        from jig.core.types import TraceContext
        tc = TraceContext.from_dict({"trace_id": "t", "parent_span_id": "s"})
        assert tc is not None
        assert tc.trace_id == "t"


class TestParseToolCallsMalformedArgs:
    """A *named* tool call with malformed or non-object arguments raises a
    retryable JigLLMError (shared-adapter semantics) instead of being
    silently dropped — the model must see its failed attempt. Entries with
    no name are provider garbage and are still skipped."""

    def test_scalar_json_string_raises(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{
            "id": "tc-1",
            "function": {"name": "echo", "arguments": "42"},
        }]
        with pytest.raises(JigLLMError) as excinfo:
            _parse_tool_calls(raw)
        assert excinfo.value.retryable

    def test_list_json_string_raises(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{
            "id": "tc-1",
            "function": {"name": "echo", "arguments": "[1, 2]"},
        }]
        with pytest.raises(JigLLMError) as excinfo:
            _parse_tool_calls(raw)
        assert excinfo.value.retryable

    def test_malformed_json_string_raises(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{
            "id": "tc-1",
            "function": {"name": "echo", "arguments": "{not json"},
        }]
        with pytest.raises(JigLLMError) as excinfo:
            _parse_tool_calls(raw)
        assert excinfo.value.retryable

    def test_valid_dict_string_accepted(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{
            "id": "tc-1",
            "function": {"name": "echo", "arguments": '{"k": "v"}'},
        }]
        calls = _parse_tool_calls(raw)
        assert calls is not None
        assert calls[0].arguments == {"k": "v"}

    def test_dict_arguments_accepted(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{
            "id": "tc-1",
            "function": {"name": "echo", "arguments": {"k": "v"}},
        }]
        calls = _parse_tool_calls(raw)
        assert calls is not None
        assert calls[0].arguments == {"k": "v"}

    def test_unnamed_entry_skipped(self):
        from jig.llm.dispatch import _parse_tool_calls
        raw = [{"id": "tc-1", "function": {"arguments": "{}"}}]
        assert _parse_tool_calls(raw) is None

    def test_non_object_entry_skipped(self):
        from jig.llm.dispatch import _parse_tool_calls
        assert _parse_tool_calls(["garbage"]) is None


@pytest.mark.asyncio
class TestRegistryTimeoutPlumbed:
    """ToolRegistry(execute_timeout=...) must reach dispatch_run."""

    async def test_timeout_passed_to_dispatch_run(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run(fn_ref, payload=None, **kwargs):
            captured.update(kwargs)
            return {"value": 1}

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        reg = ToolRegistry([_DispatchedTool()], execute_timeout=42.5)
        await reg.execute(ToolCall(
            id="c1", name="backtest", arguments={},
        ))
        # Rounded up so fractional registry timeouts are not shortened.
        assert captured["timeout_seconds"] == 43

    async def test_no_timeout_omits_kwarg(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run(fn_ref, payload=None, **kwargs):
            captured.update(kwargs)
            return {"value": 1}

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        reg = ToolRegistry([_DispatchedTool()])  # no execute_timeout
        await reg.execute(ToolCall(
            id="c1", name="backtest", arguments={},
        ))
        assert "timeout_seconds" not in captured


@pytest.mark.asyncio
class TestSharedHttpLifecycle:
    """``jig.dispatch.aclose`` closes the shared client; the client
    gets rebound when the running event loop changes."""

    async def test_aclose_safe_when_never_used(self):
        import jig.dispatch
        # Shouldn't raise, shouldn't hang
        await jig.dispatch.aclose()
        await jig.dispatch.aclose()  # idempotent

    async def test_aclose_closes_shared_client(self):
        import jig.dispatch
        from jig.dispatch import client as dc

        # Force creation
        client = dc._get_shared_http()
        assert not client.is_closed

        await jig.dispatch.aclose()
        assert client.is_closed
        # Module globals cleared so next call creates a fresh client
        assert dc._shared_http is None

    async def test_rebind_on_loop_change(self):
        """A client bound to a closed loop must not be reused."""
        from jig.dispatch import client as dc

        # Simulate a prior loop binding: plant a sentinel loop that
        # isn't the one we're running on.
        dc._shared_http = MagicMock(spec=httpx.AsyncClient)
        dc._shared_http_loop = asyncio.new_event_loop()
        try:
            fresh = dc._get_shared_http()
            # Got a real client bound to the current loop
            assert isinstance(fresh, httpx.AsyncClient)
            assert dc._shared_http_loop is asyncio.get_running_loop()
        finally:
            await dc.aclose()


@pytest.mark.asyncio
class TestOnSubmittedHook:
    """dispatch_run(on_submitted=...) observes the smithers job id at
    acceptance time, before the terminal wait."""

    async def test_hook_receives_job_id_on_success(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp("j-42")
        http.get.return_value = _poll_resp(status="complete", result={"value": 1})
        seen: list[str] = []

        out = await dispatch_run(
            "m:f", http=http, poll_interval=0.01, on_submitted=seen.append,
        )

        assert out == 1
        assert seen == ["j-42"]

    async def test_hook_fires_even_when_job_later_fails(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp("j-42")
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"status": "failed", "error": "boom"}
        r.raise_for_status = MagicMock()
        http.get.return_value = r
        seen: list[str] = []

        with pytest.raises(DispatchError, match="boom"):
            await dispatch_run(
                "m:f", http=http, poll_interval=0.01, on_submitted=seen.append,
            )

        assert seen == ["j-42"]

    async def test_hook_not_called_when_submission_fails(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.side_effect = httpx.ConnectError("nope")
        seen: list[str] = []

        with pytest.raises(DispatchError):
            await dispatch_run(
                "m:f", http=http, poll_interval=0.01, on_submitted=seen.append,
            )

        assert seen == []

    async def test_hook_exception_cancels_job_and_propagates(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp(
            {"job_id": "j-1", "status": "cancelled"},
        )

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded"):
            await dispatch_run(
                "m:f", http=http, poll_interval=0.01, on_submitted=boom,
            )

        http.delete.assert_awaited_once_with(
            "http://localhost:8900/jobs/j-1", timeout=10.0,
        )

    async def test_hook_exception_fences_by_idempotency_key(self):
        """With a key in play, the rejection path uses the durable fence —
        a per-job DELETE only speaks to the attempt in hand."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp(
            {"job_id": "j-1", "status": "cancelled"},
        )

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded"):
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        http.delete.assert_awaited_once_with(
            "http://localhost:8900/jobs/by-idempotency/attempt-7",
            timeout=10.0,
        )

    async def test_unacknowledged_cancellation_is_reported_on_hook_error(self):
        """A 503 fence must not read as a clean record-or-cancel rejection —
        the job may still be running."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({"detail": "not acked"}, 503)

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with (
            patch("jig.dispatch.client._FENCE_RETRY_SECONDS", 0),
            pytest.raises(RuntimeError, match="hook exploded") as caught,
        ):
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        assert any(
            "may still be running" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_unacknowledged_cancellation_is_retried(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.side_effect = [
            _job_api_resp({"detail": "not acked"}, 503),
            _job_api_resp({"job_id": "j-1", "status": "cancelled"}),
        ]

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with (
            patch("jig.dispatch.client._FENCE_RETRY_SECONDS", 0),
            pytest.raises(RuntimeError, match="hook exploded") as caught,
        ):
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        assert http.delete.await_count == 2
        assert not getattr(caught.value, "__notes__", [])

    async def test_non_retryable_cancellation_failure_is_not_retried(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({"detail": "bad key"}, 400)

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded") as caught:
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        assert http.delete.await_count == 1
        assert any(
            "may still be running" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_completed_job_is_not_reported_as_a_clean_cancellation(self):
        """A 409 fence on a job that already finished means it ran unrecorded
        — the opposite of what a record-or-cancel rejection claims. Smithers'
        keyed 409 carries only a ``detail`` string, so the status is read
        back by job id."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp(
            {"detail": "Job already complete"}, status_code=409,
        )
        http.get.return_value = _job_api_resp({"id": "j-1", "status": "complete"})

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded") as caught:
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        http.get.assert_awaited_once_with("http://localhost:8900/jobs/j-1")
        notes = getattr(caught.value, "__notes__", [])
        assert any("'complete'" in note for note in notes)
        assert any("ran unrecorded" in note for note in notes)

    async def test_terminal_fence_without_a_status_still_warns(self):
        """Terminality of an unknown kind must not silently pass as a clean
        stop — the caller is told the job reached a terminal state."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp(
            {"detail": "Job already failed"}, status_code=409,
        )
        http.get.return_value = _job_api_resp({"detail": "unavailable"}, 503)

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded") as caught:
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        assert any(
            "already terminal" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_cancelled_fence_is_reported_as_clean(self):
        """The ordinary case still carries no warning at all — including the
        body smithers sends when the tombstone found no job to cancel."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp(
            {"idempotency_key": "attempt-7", "status": "cancelled"},
        )

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded") as caught:
            await dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            )

        assert not getattr(caught.value, "__notes__", [])

    async def test_keyless_terminal_cancellation_reads_back_the_outcome(self):
        """Without a key the per-job DELETE answers 409 without saying which
        terminal state it found, so the client reads the job to find out."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({}, status_code=409)
        http.get.return_value = _job_api_resp({"id": "j-1", "status": "complete"})

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with pytest.raises(RuntimeError, match="hook exploded") as caught:
            await dispatch_run(
                "m:f", http=http, poll_interval=0.01, on_submitted=boom,
            )

        http.get.assert_awaited_once_with("http://localhost:8900/jobs/j-1")
        assert any(
            "'complete'" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_terminal_status_read_is_bounded(self):
        """The read-back runs while a hook exception waits to propagate, so a
        caller client with no timeout must not be able to stall it."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({}, status_code=409)

        async def never(*args, **kwargs):
            await asyncio.sleep(3600)

        http.get.side_effect = never

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        with (
            patch("jig.dispatch.client._CANCEL_TIMEOUT_SECONDS", 0.01),
            pytest.raises(RuntimeError, match="hook exploded") as caught,
        ):
            await dispatch_run(
                "m:f", http=http, poll_interval=0.01, on_submitted=boom,
            )

        assert any(
            "already terminal" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_cancellation_during_the_fence_does_not_abandon_it(self):
        """shield protects the fence task, not the await on it. A cancel
        arriving mid-fence must not detach the fence, skip the listener
        cleanup, or drop the hook failure on the floor."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()

        fence_landed = False
        fence_started = asyncio.Event()

        async def slow_fence(*args, **kwargs):
            nonlocal fence_landed
            fence_started.set()
            await asyncio.sleep(0.05)
            fence_landed = True
            return _job_api_resp({"job_id": "j-1", "status": "cancelled"})

        http.delete.side_effect = slow_fence

        def boom(job_id: str) -> None:
            raise RuntimeError("hook exploded")

        task = asyncio.create_task(
            dispatch_run(
                "m:f",
                http=http,
                poll_interval=0.01,
                on_submitted=boom,
                idempotency_key="attempt-7",
            ),
        )
        await fence_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError) as caught:
            await task

        assert fence_landed
        # Cancellation wins, but the hook failure it interrupted rides along.
        assert caught.value.__cause__ is not None
        assert "hook exploded" in str(caught.value.__cause__)
        assert any(
            "hook exploded" in note
            for note in getattr(caught.value, "__notes__", [])
        )

    async def test_async_hook_is_awaited_before_polling(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        registered = False

        async def register(job_id: str) -> None:
            nonlocal registered
            await asyncio.sleep(0)
            registered = job_id == "j-1"

        def poll_response(*args, **kwargs):
            assert registered
            return _poll_resp(status="complete", result={"value": "ok"})

        http.get.side_effect = poll_response

        out = await dispatch_run(
            "m:f", http=http, poll_interval=0.01, on_submitted=register,
        )

        assert out == "ok"

    @pytest.mark.parametrize("cancel_caller", [False, True])
    async def test_unexpected_fence_error_preserves_hook_and_cleanup(self, cancel_caller):
        from jig.dispatch.client import _submit_and_poll

        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        fence_started = asyncio.Event()
        release_fence = asyncio.Event()

        async def fail_fence(*args, **kwargs):
            fence_started.set()
            await release_fence.wait()
            raise RuntimeError("transport failed unexpectedly")

        http.delete.side_effect = fail_fence
        listener = MagicMock()
        listener.health_check = AsyncMock()
        listener.register.return_value = (
            "nonce-1", asyncio.get_running_loop().create_future(),
        )
        hook_error = ValueError("registration rejected")

        def reject(job_id):
            raise hook_error

        task = asyncio.create_task(_submit_and_poll(
            http=http,
            dispatch_url="http://localhost:8900",
            task_type="function",
            payload={},
            idempotency_key="attempt-7",
            listener=listener,
            on_submitted=reject,
        ))
        await asyncio.wait_for(fence_started.wait(), timeout=1)
        if cancel_caller:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release_fence.set()

        expected = asyncio.CancelledError if cancel_caller else ValueError
        with pytest.raises(expected) as caught:
            await task

        if cancel_caller:
            assert caught.value.__cause__ is hook_error
            assert task.cancelled()
        else:
            assert caught.value is hook_error
        assert any(
            "may still be running" in note and "transport failed unexpectedly" in note
            for note in caught.value.__notes__
        )
        listener.unregister.assert_called_once_with("nonce-1")
        http.get.assert_not_awaited()

    async def test_success_job_dict_always_carries_job_id(self):
        from jig.dispatch.client import _submit_and_poll

        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp("j-7")
        # Poll body deliberately omits job_id — the client re-attaches it.
        http.get.return_value = _poll_resp(status="complete", result={"value": 1})

        data = await _submit_and_poll(
            http=http,
            dispatch_url="http://test",
            task_type="function",
            payload={},
            poll_config=_PollConfig(
                timeout_seconds=5,
                cleanup_grace_seconds=0,
                poll_interval=0.01,
                poll_max_interval=0.02,
            ),
        )

        assert data["job_id"] == "j-7"


class _DispatchedToolWithHook(_DispatchedTool):
    """Dispatched tool that duck-types on_dispatch_submitted."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def on_dispatch_submitted(self, job_id: str) -> None:
        self.seen.append(job_id)


@pytest.mark.asyncio
class TestRegistryOnSubmittedPlumbed:
    """A dispatched tool's on_dispatch_submitted must reach dispatch_run as
    on_submitted; tools without the method add no kwarg."""

    async def test_duck_typed_hook_forwarded(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run(fn_ref, payload=None, **kwargs):
            captured.update(kwargs)
            return {"value": 1}

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        tool = _DispatchedToolWithHook()
        reg = ToolRegistry([tool])
        await reg.execute(ToolCall(id="c1", name="backtest", arguments={}))

        hook = captured.get("on_submitted")
        assert callable(hook)
        hook("j-9")
        assert tool.seen == ["j-9"]

    async def test_absent_hook_omits_kwarg(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run(fn_ref, payload=None, **kwargs):
            captured.update(kwargs)
            return {"value": 1}

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        reg = ToolRegistry([_DispatchedTool()])
        await reg.execute(ToolCall(id="c1", name="backtest", arguments={}))

        assert "on_submitted" not in captured

    async def test_fence_warning_survives_into_the_tool_result(self, monkeypatch):
        """The failed-fence warning rides on the exception as a note, and
        ToolResult.error is built from str(e) — which drops notes. It is the
        only signal the caller gets that the job may still be running, so it
        has to survive the registry boundary."""
        async def fake_run(fn_ref, payload=None, **kwargs):
            error = RuntimeError("registration rejected")
            error.add_note("Dispatch job j-9 ... may still be running: 503")
            raise error

        import jig.dispatch
        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        reg = ToolRegistry([_DispatchedToolWithHook()])
        result = await reg.execute(
            ToolCall(id="c1", name="backtest", arguments={}),
        )

        assert result.error is not None
        assert "registration rejected" in result.error
        assert "may still be running" in result.error

    async def test_fence_warning_survives_an_execute_timeout(self):
        """When execute_timeout fires while a rejected submission is being
        fenced, wait_for raises TimeoutError *from* the CancelledError that
        carries the fence warning. The warning must still reach the model."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({"detail": "not acked"}, 503)

        class _SlowRejectingTool(_DispatchedTool):
            async def on_dispatch_submitted(self, job_id: str) -> None:
                await asyncio.sleep(3600)

        reg = ToolRegistry(
            [_SlowRejectingTool()],
            dispatch_url="http://localhost:8900",
            execute_timeout=0.05,
        )
        with (
            patch("jig.dispatch.client._get_shared_http", return_value=http),
            patch("jig.dispatch.client._FENCE_RETRY_SECONDS", 0),
        ):
            result = await reg.execute(
                ToolCall(id="c1", name="backtest", arguments={}),
            )

        assert result.error is not None
        assert result.error.startswith("TimeoutError:")
        assert "may still be running" in result.error

    @pytest.mark.parametrize("registry_timeout", [False, True])
    @pytest.mark.parametrize("fence_status", [409, 503])
    async def test_timeout_preserves_fence_warning(
        self, monkeypatch, registry_timeout, fence_status,
    ):
        import jig.dispatch
        from jig.dispatch import client

        http = AsyncMock(spec=httpx.AsyncClient)
        http.post.return_value = _submit_resp()
        http.delete.return_value = _job_api_resp({}, status_code=fence_status)
        http.get.return_value = _job_api_resp({"status": "complete"})
        hook_started = asyncio.Event()

        async def reject(job_id):
            hook_started.set()
            if registry_timeout:
                await asyncio.Event().wait()
            raise TimeoutError("correlation database timed out")

        async def run(*args, **kwargs):
            return await client.run(*args, http=http, **kwargs)

        monkeypatch.setattr(jig.dispatch, "run", run)
        monkeypatch.setattr(client, "_FENCE_RETRY_SECONDS", 0)
        tool = _DispatchedToolWithHook()
        tool.on_dispatch_submitted = reject
        timeout = 0.05 if registry_timeout else None
        reg = ToolRegistry([tool], execute_timeout=timeout)
        result = await reg.execute(ToolCall(id="c1", name="backtest", arguments={}))

        assert hook_started.is_set()
        assert result.error.startswith(
            f"TimeoutError: Dispatched tool backtest timed out after {timeout}s",
        )
        warning = "ran unrecorded" if fence_status == 409 else "may still be running"
        assert warning in result.error
        assert "j-1" in result.error

    async def test_timeout_deduplicates_notes_from_cancellation(self, monkeypatch):
        import jig.dispatch

        warning = "Dispatch job j-1 may still be running"

        async def run(*args, **kwargs):
            cancellation = asyncio.CancelledError()
            cancellation.add_note(warning)
            error = TimeoutError()
            error.add_note(warning)
            raise error from cancellation

        monkeypatch.setattr(jig.dispatch, "run", run)
        reg = ToolRegistry([_DispatchedToolWithHook()])
        result = await reg.execute(ToolCall(id="c1", name="backtest", arguments={}))

        assert result.error.count(warning) == 1


class TestStrictToolPayload:
    """Tools that opt into strict carry "strict": true in both payload
    builders; non-opted tools serialize exactly as before."""

    def test_strict_flag_serialized_only_when_opted_in(self):
        from jig.llm._common import openai_tool_payload
        from jig.llm.dispatch import _tools_payload

        strict_tool = ToolDefinition(
            name="s", description="d",
            parameters={"type": "object", "additionalProperties": False},
            strict=True,
        )
        plain_tool = ToolDefinition(
            name="p", description="d", parameters={"type": "object"},
        )
        for builder in (_tools_payload, openai_tool_payload):
            out = builder([strict_tool, plain_tool])
            assert out[0]["function"]["strict"] is True
            assert "strict" not in out[1]["function"]
