from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from collections.abc import Mapping
from typing import Any

from jig.core.errors import JigToolError
from jig.core.types import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    current_tool_context,
)

logger = logging.getLogger(__name__)

# Reserved key a dispatched result Mapping can carry to report a business
# (not infrastructure) failure without raising — see jig.dispatch.tool_error
# and the coercion block in _execute_dispatched.
_TOOL_ERROR_KEY = "__jig_tool_error__"


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
            except JigToolError as e:
                # Caught ahead of the generic handler below so a tool (or
                # a future validate_arguments pass) can tag *where* in the
                # lifecycle it failed; the phase rides along in the
                # model-visible error instead of being lost in the log.
                msg = str(e) or f"{e.phase} raised without message"
                error = f"{e.phase}: {msg}"
                logger.warning("tool.execute error name=%s err=%s", call.name, error, exc_info=True)
                return ToolResult(call_id=call.id, output="", error=error)
            except Exception as e:
                msg = str(e)
                error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
                logger.warning("tool.execute error name=%s err=%s", call.name, error, exc_info=True)
                return ToolResult(call_id=call.id, output="", error=error)
        finally:
            current_tool_context.reset(token)

    async def _pre_dispatch(
        self,
        tool: Tool,
        call: ToolCall,
        tool_context: ToolExecutionContext | None,
    ) -> None:
        """Explicit pre-dispatch gate: duck-typed ``Tool.pre_dispatch``,
        called before ``dispatch_payload_extra`` so a tool no longer has
        to smuggle gating logic into payload shaping. Sync or async;
        absent hook is a no-op.

        Any exception raised here — including a tool-raised
        ``JigToolError`` with its own explicit ``phase`` — is normalized
        to a ``JigToolError``, defaulting to ``phase="gate"`` when the
        tool didn't pick one, so ``_execute_dispatched``'s single
        ``JigToolError`` handler formats the model-visible error the
        same way regardless of which pre-dispatch step rejected the call.
        """
        pre_dispatch = getattr(tool, "pre_dispatch", None)
        if not callable(pre_dispatch):
            return
        try:
            outcome = _call_context_hook(
                pre_dispatch, tool_context=tool_context, arguments=call.arguments,
            )
            if inspect.isawaitable(outcome):
                await outcome
        except JigToolError:
            raise
        except Exception as e:
            msg = str(e) or "pre_dispatch raised without message"
            raise JigToolError(msg, tool_name=call.name, phase="gate") from e

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
        from jig.dispatch import DispatchBusinessError, DispatchError, run as dispatch_run

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
        # Duck-typed like dispatch_payload_extra: a dispatched tool that
        # defines on_dispatch_submitted(job_id) gets the smithers job id at
        # acceptance time, so it can persist the correlation while the job
        # is still running.
        on_submitted = getattr(tool, "on_dispatch_submitted", None)
        if callable(on_submitted):
            kwargs["on_submitted"] = on_submitted

        try:
            async def _gate_and_dispatch() -> object:
                # pre_dispatch runs first and can reject the call before
                # dispatch_payload_extra ever builds a payload or
                # anything ships to smithers.
                await self._pre_dispatch(tool, call, tool_context)

                payload = dict(call.arguments)
                extra = self._dispatch_payload_extra(tool, call, tool_context)
                if inspect.isawaitable(extra):
                    extra = await extra
                if isinstance(extra, Mapping):
                    payload.update(
                        {key: value for key, value in extra.items() if value is not None}
                    )
                return await dispatch_run(
                    tool.dispatch_fn_ref,  # type: ignore[arg-type]
                    payload,
                    **kwargs,
                )

            if self._execute_timeout is not None:
                result = await asyncio.wait_for(
                    _gate_and_dispatch(),
                    timeout=self._execute_timeout,
                )
            else:
                result = await _gate_and_dispatch()
        except JigToolError as e:
            # Raised by pre_dispatch (normalized to phase="gate" there
            # unless the tool picked a different phase itself) or by a
            # tool's own dispatch_payload_extra. Either way nothing was
            # dispatched, so on_dispatch_error — a *post*-dispatch hook —
            # deliberately does not fire here; the ToolResult.error is
            # the only signal, same as a gate rejection has always been.
            msg = str(e) or f"{e.phase} raised without message"
            error = f"{e.phase}: {msg}"
            logger.warning("tool.execute %s error name=%s err=%s", e.phase, call.name, error)
            return ToolResult(call_id=call.id, output="", error=error)
        except asyncio.TimeoutError as e:
            logger.warning("tool.execute dispatch timeout name=%s after=%ss", call.name, self._execute_timeout)
            await _fire_dispatch_error_hook(tool, e, tool_context)
            return ToolResult(
                call_id=call.id,
                output="",
                error=f"TimeoutError: Dispatched tool {call.name} timed out after {self._execute_timeout}s",
            )
        except DispatchError as e:
            msg = str(e)
            error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
            logger.warning("tool.execute dispatch error name=%s err=%s", call.name, error)
            await _fire_dispatch_error_hook(tool, e, tool_context)
            return ToolResult(
                call_id=call.id,
                output="",
                error=error,
            )
        except Exception as e:
            msg = str(e)
            error = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}: tool raised without message"
            logger.warning("tool.execute dispatch error name=%s err=%s", call.name, error, exc_info=True)
            await _fire_dispatch_error_hook(tool, e, tool_context)
            return ToolResult(call_id=call.id, output="", error=error)

        # A dispatched function has, up to this point, exactly two
        # outcomes: return (this branch) or raise (caught above). The
        # reserved-key business-failure check below adds a third: the
        # worker ran to completion but is reporting its own domain-level
        # failure (jig.dispatch.tool_error). It is checked here, on the
        # *raw* result, before on_dispatch_result fires — a business
        # failure is not a success to reconcile, so it must never reach
        # that hook. Instead it is synthesized into a
        # DispatchBusinessError and routed through on_dispatch_error,
        # exactly like a DispatchError or a timeout, so one reconciliation
        # hook implementation sees every dispatch failure the same way,
        # and on_dispatch_result never observes a
        # ``__jig_tool_error__``-bearing result.
        if isinstance(result, Mapping) and _TOOL_ERROR_KEY in result:
            err_payload = result[_TOOL_ERROR_KEY]
            rest = {key: value for key, value in result.items() if key != _TOOL_ERROR_KEY}
            if isinstance(err_payload, str):
                message = err_payload
            elif isinstance(err_payload, Mapping):
                message = str(err_payload.get("message", err_payload))
            else:
                message = str(err_payload)
            logger.warning("tool.execute dispatch business_error name=%s err=%s", call.name, message)
            await _fire_dispatch_error_hook(
                tool, DispatchBusinessError(message, payload=rest), tool_context,
            )
            if rest:
                try:
                    output = json.dumps(rest)
                except (TypeError, ValueError):
                    output = str(rest)
            else:
                output = ""
            return ToolResult(call_id=call.id, output=output, error=message)

        on_dispatch_result = getattr(tool, "on_dispatch_result", None)
        if callable(on_dispatch_result):
            # Runs outside the execute_timeout wait_for above: that
            # budget is for the remote job, and local reconciliation
            # (e.g. persisting a completion row) must not be clipped by
            # it. Unlike on_dispatch_submitted, exceptions here are NOT
            # swallowed — a failed reconciliation means downstream state
            # is unrecorded, and the model must not proceed as if the
            # result were usable.
            try:
                replacement = on_dispatch_result(result, tool_context)
                if inspect.isawaitable(replacement):
                    replacement = await replacement
                if replacement is not None:
                    result = replacement
            except Exception as e:
                msg = str(e) or "on_dispatch_result raised without message"
                logger.warning("tool.on_dispatch_result error name=%s err=%s", call.name, msg)
                return ToolResult(call_id=call.id, output="", error=f"{type(e).__name__}: {msg}")

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
    ) -> Any:
        dispatch_payload_extra = getattr(tool, "dispatch_payload_extra", None)
        if not callable(dispatch_payload_extra):
            return {}
        return _call_context_hook(
            dispatch_payload_extra, tool_context=tool_context, arguments=call.arguments,
        )


def _call_context_hook(
    fn: Any,
    *,
    tool_context: ToolExecutionContext | None,
    arguments: dict[str, Any] | None,
) -> Any:
    """Call a duck-typed hook whose signature may name its parameters
    ``context``/``tool_context``/``ctx`` and ``arguments``/``args``, in
    either order, positional-only or keyword, or take none at all.
    Shared by ``dispatch_payload_extra`` and ``pre_dispatch`` so hook
    authors don't have to match jig's exact parameter names or order.

    Does not await the result — some callers (``dispatch_payload_extra``)
    defer that to their own caller, so this only performs the call and
    lets ``inspect.isawaitable`` be checked at each call site.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )

    positional: list[object] = []
    kwargs: dict[str, object] = {}
    for name, param in params.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue

        if name in ("context", "tool_context", "ctx"):
            value: object = tool_context
        elif name in ("arguments", "args"):
            value = arguments
        else:
            continue

        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[name] = value

    if has_var_kwargs:
        kwargs.setdefault("context", tool_context)
        kwargs.setdefault("arguments", arguments)

    if positional or kwargs:
        return fn(*positional, **kwargs)
    if not params:
        return fn()
    # Unrecognized single-param signature; assume it wants context.
    return fn(tool_context)


async def _fire_dispatch_error_hook(
    tool: Tool, error: BaseException, tool_context: ToolExecutionContext | None,
) -> None:
    """Best-effort invocation of ``Tool.on_dispatch_error``.

    Shared by every dispatch-failure branch in ``_execute_dispatched``:
    the timeout, ``DispatchError``, generic-exception, and business-
    failure (``__jig_tool_error__``) cases all funnel through here. Sync
    or async hook, awaited either way. Hook exceptions are logged and
    swallowed — the error ``ToolResult`` already being returned to the
    model is the signal regardless of whether a hook ran; a buggy hook
    must not mask or replace the original dispatch failure.
    """
    hook = getattr(tool, "on_dispatch_error", None)
    if not callable(hook):
        return
    try:
        outcome = hook(error, tool_context)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        logger.warning(
            "tool.on_dispatch_error hook raised; original error was %s: %s",
            type(error).__name__, error, exc_info=True,
        )
