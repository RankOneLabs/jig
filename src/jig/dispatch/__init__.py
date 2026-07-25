"""Dispatch primitives: route LLM calls and deterministic functions to smithers.

The smithers dispatch server accepts several task types — ``inference`` for
LLM calls (used by :class:`jig.llm.DispatchClient`) and ``function`` for
named Python callables registered on the worker side (used by
:func:`run` below). Both paths share a submit-then-wait primitive in
:mod:`jig.dispatch.client`.

When :func:`listen` has started the callback receiver, in-flight jobs
resolve via an :class:`asyncio.Future` keyed by a callback nonce —
sweeps stop paying for one polling coroutine per run. Listener start
requires ``pip install 'jig[callback]'`` to pull in ``aiohttp``; the
polling path stays available for callers that don't opt in.
"""
from jig.dispatch.client import (
    DispatchBusinessError,
    DispatchError,
    JobTimeoutError,
    aclose,
    run,
)


#: Reserved result key marking a dispatched function's business failure.
#: Single source of truth for the wire contract — ``tool_error`` writes it
#: and ``ToolRegistry._execute_dispatched`` reads it, so it must not be
#: spelled as a literal in either place. Exported for worker-side tooling
#: that needs to recognize the shape without importing the registry.
TOOL_ERROR_KEY = "__jig_tool_error__"


def tool_error(message: str, **payload: object) -> dict:
    """Return value for a dispatched function whose business outcome failed.

    A dispatched worker function has, by default, exactly two outcomes as
    far as ``ToolRegistry`` is concerned: return (treated as success, even
    if the payload happens to look like ``{"error": ...}``) or raise
    (becomes ``ToolResult.error`` directly). ``tool_error`` gives worker
    code a third option — report a business-level failure without raising,
    while still preserving whatever partial payload it already has.

    The returned mapping carries the reserved ``"__jig_tool_error__"`` key;
    ``ToolRegistry._execute_dispatched`` recognizes it on the raw dispatch
    result and converts it into ``ToolResult.error`` (using ``message``),
    while every other key in ``payload`` survives into ``ToolResult.output``
    so the model still sees whatever partial state the worker collected.
    The registry also synthesizes a :class:`DispatchBusinessError` from it
    and routes that through ``Tool.on_dispatch_error``, so a tool's
    reconciliation hook sees this the same way it sees a timeout or a
    submission failure.
    """
    return {TOOL_ERROR_KEY: message, **payload}


async def listen(**kwargs):
    """Start (or return) the process-wide callback listener.

    See :func:`jig.dispatch.listener.listen` for kwargs. Imported lazily
    so callers who don't use the callback path don't pay for aiohttp.
    """
    from jig.dispatch.listener import listen as _listen
    return await _listen(**kwargs)


async def stop():
    """Stop the process-wide callback listener, if any. Idempotent."""
    from jig.dispatch.listener import stop as _stop
    return await _stop()


def __getattr__(name: str):
    """Lazy attribute access for listener types.

    Re-exporting ``CallbackListener`` / ``ListenerError`` via normal
    import would drag ``aiohttp`` into every consumer. With
    ``__getattr__`` the aiohttp import only fires when a caller
    actually references one of these names — matching the lazy
    behavior of the ``listen`` / ``stop`` wrappers.
    """
    if name in {"CallbackListener", "ListenerError"}:
        from jig.dispatch import listener as _listener_mod
        return getattr(_listener_mod, name)
    raise AttributeError(f"module 'jig.dispatch' has no attribute {name!r}")


__all__ = [
    "CallbackListener",
    "DispatchBusinessError",
    "DispatchError",
    "JobTimeoutError",
    "ListenerError",
    "TOOL_ERROR_KEY",
    "aclose",
    "listen",
    "run",
    "stop",
    "tool_error",
]
