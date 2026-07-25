"""Tests for §1.1-1.4 of the jig lifecycle-hooks spec (PR 1a):

- Tool.on_dispatch_result   (§1.1) — post-dispatch success hook
- Tool.on_dispatch_error    (§1.2) — post-dispatch failure hook
- Tool.pre_dispatch         (§1.3) — explicit gate before dispatch_payload_extra,
  plus JigToolError wiring (extended "gate"/"dispatch" phases)
- __jig_tool_error__        (§1.4) — business-failure result channel + the
  jig.dispatch.tool_error() worker-side helper

§1.5 (validate_arguments / jsonschema) ships separately and is not covered here.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

import jig.dispatch
from jig import DispatchBusinessError, DispatchError, JigToolError, ToolCall, ToolDefinition, ToolExecutionContext, ToolRegistry
from jig.core.types import Tool
from jig.dispatch import tool_error


def _context(**overrides: Any) -> ToolExecutionContext:
    defaults: dict[str, Any] = dict(
        trace_id="t-1", span_id="s-1", parent_span_id=None, tool_call_id="c-1",
    )
    defaults.update(overrides)
    return ToolExecutionContext(**defaults)


class _DispatchedTool(Tool):
    """Minimal dispatch=True tool with none of the new optional hooks — the
    behavior-preservation baseline every hook test is measured against."""

    dispatch = True

    @property
    def dispatch_fn_ref(self) -> str | None:
        return "pkg.module:fn"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dispatched", description="d", parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        raise AssertionError("execute() must not be called for a dispatch=True tool")


def _call(arguments: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id="call-1", name="dispatched", arguments=arguments or {})


# --- §1.1 Tool.on_dispatch_result -------------------------------------------


class _OnResultTool(_DispatchedTool):
    def __init__(self, hook) -> None:
        self._hook = hook
        self.seen: list[tuple[Any, ToolExecutionContext | None]] = []

    def on_dispatch_result(self, result: object, context: ToolExecutionContext | None) -> object | None:
        self.seen.append((result, context))
        return self._hook(result, context)


class _AsyncOnResultTool(_DispatchedTool):
    def __init__(self, hook) -> None:
        self._hook = hook
        self.seen: list[Any] = []

    async def on_dispatch_result(self, result: object, context: ToolExecutionContext | None) -> object | None:
        self.seen.append(result)
        await asyncio.sleep(0)
        return self._hook(result, context)


class TestOnDispatchResultHook:
    async def test_passthrough_when_hook_returns_none(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 1.5}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _OnResultTool(lambda result, ctx: None)
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"sharpe": 1.5}
        # Hook saw the raw dict, not a coerced string.
        assert tool.seen == [({"sharpe": 1.5}, None)]

    async def test_replacement_reflected_in_output(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 1.5}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _OnResultTool(lambda result, ctx: {"sharpe": 1.5, "reconciled": True})
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"sharpe": 1.5, "reconciled": True}

    async def test_veto_raises_becomes_tool_result_error_no_output(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 1.5}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        def boom(result, ctx):
            raise RuntimeError("reconciliation failed")

        tool = _OnResultTool(boom)
        result = await ToolRegistry([tool]).execute(_call())

        assert result.output == ""
        assert result.error == "RuntimeError: reconciliation failed"

    async def test_async_hook_is_awaited(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"a": 1}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _AsyncOnResultTool(lambda result, ctx: {"a": 1, "b": 2})
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"a": 1, "b": 2}
        assert tool.seen == [{"a": 1}]

    async def test_absent_hook_unchanged(self, monkeypatch):
        """A dispatch=True tool with no on_dispatch_result behaves exactly
        as it did before this feature existed."""
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 1.5, "trades": 12}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"sharpe": 1.5, "trades": 12}

    async def test_hook_not_clipped_by_short_execute_timeout(self, monkeypatch):
        """The timeout budget belongs to the remote job; local
        reconciliation in on_dispatch_result must not be cut off by it."""
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"a": 1}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        async def slow_hook(result: object, context: ToolExecutionContext | None) -> None:
            await asyncio.sleep(0.2)
            return None

        class _SlowHookTool(_DispatchedTool):
            on_dispatch_result = staticmethod(slow_hook)

        registry = ToolRegistry([_SlowHookTool()], execute_timeout=0.05)
        start = time.monotonic()
        result = await registry.execute(_call())
        elapsed = time.monotonic() - start

        assert result.error is None
        assert json.loads(result.output) == {"a": 1}
        # If the hook were (incorrectly) wrapped in the 0.05s wait_for,
        # this would come back as a TimeoutError well before 0.2s elapsed.
        assert elapsed >= 0.15


# --- §1.2 Tool.on_dispatch_error ---------------------------------------------


class _OnErrorTool(_DispatchedTool):
    def __init__(self, hook=None) -> None:
        self._hook = hook
        self.seen: list[BaseException] = []

    def on_dispatch_error(self, error: BaseException, context: ToolExecutionContext | None) -> None:
        self.seen.append(error)
        if self._hook is not None:
            self._hook(error, context)


class TestOnDispatchErrorHook:
    async def test_fires_on_dispatch_error(self, monkeypatch):
        async def boom(fn_ref, payload=None, **kwargs):
            raise DispatchError("worker not found")

        monkeypatch.setattr(jig.dispatch, "run", boom)
        tool = _OnErrorTool()
        result = await ToolRegistry([tool]).execute(_call())

        assert "worker not found" in result.error
        assert len(tool.seen) == 1
        assert isinstance(tool.seen[0], DispatchError)
        assert str(tool.seen[0]) == "worker not found"

    async def test_fires_on_timeout(self, monkeypatch):
        async def hang(fn_ref, payload=None, **kwargs):
            await asyncio.Future()  # never resolves

        monkeypatch.setattr(jig.dispatch, "run", hang)
        tool = _OnErrorTool()
        result = await ToolRegistry([tool], execute_timeout=0.01).execute(_call())

        assert "timed out" in result.error
        assert len(tool.seen) == 1
        assert isinstance(tool.seen[0], asyncio.TimeoutError)

    async def test_fires_on_generic_exception(self, monkeypatch):
        async def boom(fn_ref, payload=None, **kwargs):
            raise RuntimeError("worker crashed weirdly")

        monkeypatch.setattr(jig.dispatch, "run", boom)
        tool = _OnErrorTool()
        result = await ToolRegistry([tool]).execute(_call())

        assert "worker crashed weirdly" in result.error
        assert len(tool.seen) == 1
        assert isinstance(tool.seen[0], RuntimeError)

    async def test_hook_exception_swallowed_original_error_preserved(self, monkeypatch):
        async def boom(fn_ref, payload=None, **kwargs):
            raise DispatchError("original dispatch failure")

        monkeypatch.setattr(jig.dispatch, "run", boom)

        def hook_boom(error, context):
            raise RuntimeError("hook exploded")

        tool = _OnErrorTool(hook_boom)
        result = await ToolRegistry([tool]).execute(_call())

        # The hook's own exception must not replace or mask the original.
        assert "original dispatch failure" in result.error
        assert "hook exploded" not in result.error

    async def test_absent_hook_unchanged(self, monkeypatch):
        async def boom(fn_ref, payload=None, **kwargs):
            raise DispatchError("worker not found")

        monkeypatch.setattr(jig.dispatch, "run", boom)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.output == ""
        assert "worker not found" in result.error


# --- §1.3 Tool.pre_dispatch + JigToolError wiring ---------------------------


class _PreDispatchRejectTool(_DispatchedTool):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def pre_dispatch(self, arguments: dict[str, Any] | None, context: ToolExecutionContext | None) -> None:
        raise self._exc


class _AsyncPreDispatchTool(_DispatchedTool):
    async def pre_dispatch(self, arguments=None, context=None) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async gate rejected")


class _ReorderedPreDispatchTool(_DispatchedTool):
    """Signature order flipped to (context, arguments) — the spec's
    canonical order is (arguments, context) — exercising the shared
    signature-sniffing helper reused from dispatch_payload_extra."""

    def pre_dispatch(self, context=None, arguments=None) -> None:
        if (arguments or {}).get("reject"):
            raise RuntimeError("rejected via reordered signature")


class TestPreDispatchGate:
    async def test_rejects_before_dispatch_ships(self, monkeypatch):
        dispatch_calls = 0

        async def fake_run(fn_ref, payload=None, **kwargs):
            nonlocal dispatch_calls
            dispatch_calls += 1
            return {"should": "never happen"}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _PreDispatchRejectTool(RuntimeError("not ready"))
        result = await ToolRegistry([tool]).execute(_call())

        assert dispatch_calls == 0  # the job genuinely never shipped
        assert result.output == ""
        assert result.error == "gate: not ready"

    async def test_error_text_carries_gate_phase(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _PreDispatchRejectTool(RuntimeError("attempt not validated"))
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error.startswith("gate: ")
        assert "attempt not validated" in result.error

    async def test_tool_raised_jig_tool_error_phase_is_preserved(self, monkeypatch):
        """A tool that raises its own JigToolError(phase=...) from
        pre_dispatch is not forced into phase='gate' — its explicit
        phase choice rides through unchanged."""
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        exc = JigToolError("payload shaping failed", tool_name="dispatched", phase="dispatch")
        tool = _PreDispatchRejectTool(exc)
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error == "dispatch: payload shaping failed"

    async def test_on_dispatch_error_does_not_fire_for_gate_rejection(self, monkeypatch):
        """Nothing was dispatched, so the post-dispatch failure hook must
        not fire for a pre_dispatch rejection."""
        dispatch_calls = 0

        async def fake_run(fn_ref, payload=None, **kwargs):
            nonlocal dispatch_calls
            dispatch_calls += 1
            return {}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        class _Tool(_PreDispatchRejectTool):
            def __init__(self, exc):
                super().__init__(exc)
                self.on_dispatch_error_calls = 0

            def on_dispatch_error(self, error, context):
                self.on_dispatch_error_calls += 1

        tool = _Tool(RuntimeError("nope"))
        await ToolRegistry([tool]).execute(_call())

        assert dispatch_calls == 0
        assert tool.on_dispatch_error_calls == 0

    async def test_async_pre_dispatch_is_awaited_and_can_reject(self, monkeypatch):
        dispatch_calls = 0

        async def fake_run(fn_ref, payload=None, **kwargs):
            nonlocal dispatch_calls
            dispatch_calls += 1
            return {}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_AsyncPreDispatchTool()]).execute(_call())

        assert dispatch_calls == 0
        assert result.error == "gate: async gate rejected"

    async def test_reordered_signature_accepted_like_dispatch_payload_extra(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"ok": True}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        tool = _ReorderedPreDispatchTool()

        allowed = await ToolRegistry([tool]).execute(_call({"reject": False}))
        assert allowed.error is None

        rejected = await ToolRegistry([tool]).execute(_call({"reject": True}))
        assert rejected.error == "gate: rejected via reordered signature"

    async def test_absent_hook_unchanged(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 2.0}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"sharpe": 2.0}


class _LocalToolRaisingJigToolError(Tool):
    """Non-dispatch tool whose execute() raises JigToolError directly —
    exercises the phase-carrying catch in ToolRegistry.execute()."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="local", description="d", parameters={"type": "object", "properties": {}})

    async def execute(self, args: dict[str, Any]) -> str:
        raise JigToolError("bad output shape", tool_name="local", phase="serialize")


class TestJigToolErrorPhaseWiring:
    async def test_local_execute_path_carries_phase(self):
        result = await ToolRegistry([_LocalToolRaisingJigToolError()]).execute(
            ToolCall(id="c", name="local", arguments={}),
        )
        assert result.output == ""
        assert result.error == "serialize: bad output shape"

    async def test_dispatch_payload_extra_raising_jig_tool_error_carries_phase(self, monkeypatch):
        """dispatch_payload_extra (not pre_dispatch) raising JigToolError
        is caught by the same handler in _execute_dispatched."""
        dispatch_calls = 0

        async def fake_run(fn_ref, payload=None, **kwargs):
            nonlocal dispatch_calls
            dispatch_calls += 1
            return {}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        class _Tool(_DispatchedTool):
            def dispatch_payload_extra(self, context=None, arguments=None):
                raise JigToolError("shaping blew up", tool_name="dispatched", phase="dispatch")

        result = await ToolRegistry([_Tool()]).execute(_call())

        assert dispatch_calls == 0
        assert result.error == "dispatch: shaping blew up"


# --- §1.4 __jig_tool_error__ business-failure channel -----------------------


class TestBusinessFailureChannel:
    async def test_reserved_key_converted_to_error(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"__jig_tool_error__": "study produced no completed trials"}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.output == ""
        assert result.error == "study produced no completed trials"

    async def test_sibling_payload_fields_preserved_in_output(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"__jig_tool_error__": "boom", "strategy_id": "abc", "attempted_trials": 3}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error == "boom"
        assert json.loads(result.output) == {"strategy_id": "abc", "attempted_trials": 3}

    async def test_dict_shaped_error_payload_extracts_message(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"__jig_tool_error__": {"message": "boom", "code": "E_STUDY"}}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error == "boom"

    async def test_absent_key_unchanged(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return {"sharpe": 1.9, "trades": 40}

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error is None
        assert json.loads(result.output) == {"sharpe": 1.9, "trades": 40}

    async def test_tool_error_helper_shape(self):
        assert tool_error("boom") == {"__jig_tool_error__": "boom"}
        assert tool_error("boom", strategy_id="abc", trials=3) == {
            "__jig_tool_error__": "boom", "strategy_id": "abc", "trials": 3,
        }

    async def test_tool_error_helper_round_trips_through_registry(self, monkeypatch):
        async def fake_run(fn_ref, payload=None, **kwargs):
            return tool_error("study timed out", strategy_id="abc")

        monkeypatch.setattr(jig.dispatch, "run", fake_run)
        result = await ToolRegistry([_DispatchedTool()]).execute(_call())

        assert result.error == "study timed out"
        assert json.loads(result.output) == {"strategy_id": "abc"}

    async def test_business_failure_routes_through_on_dispatch_error_not_result(self, monkeypatch):
        """§1.4: checked before on_dispatch_result fires. A business
        failure is not a success to reconcile — it flows through
        on_dispatch_error, synthesized as a DispatchBusinessError, and
        on_dispatch_result never sees it."""
        async def fake_run(fn_ref, payload=None, **kwargs):
            return tool_error("study failed", strategy_id="abc")

        monkeypatch.setattr(jig.dispatch, "run", fake_run)

        class _Tool(_DispatchedTool):
            def __init__(self):
                self.on_result_calls = 0
                self.on_error_calls: list[BaseException] = []

            def on_dispatch_result(self, result, context):
                self.on_result_calls += 1

            def on_dispatch_error(self, error, context):
                self.on_error_calls.append(error)

        tool = _Tool()
        result = await ToolRegistry([tool]).execute(_call())

        assert result.error == "study failed"
        assert tool.on_result_calls == 0
        assert len(tool.on_error_calls) == 1
        synthesized = tool.on_error_calls[0]
        assert isinstance(synthesized, DispatchBusinessError)
        assert str(synthesized) == "study failed"
        assert synthesized.payload == {"strategy_id": "abc"}

    async def test_dispatch_business_error_not_a_dispatch_error_subclass(self):
        """Distinguishable from infrastructure failures: a consumer must
        be able to tell 'the job never ran' (DispatchError) apart from
        'the job ran and reported a business failure' (this type)."""
        assert not issubclass(DispatchBusinessError, DispatchError)
