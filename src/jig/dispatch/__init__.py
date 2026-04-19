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
    DispatchError,
    JobTimeoutError,
    aclose,
    run,
)


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
    "DispatchError",
    "JobTimeoutError",
    "ListenerError",
    "aclose",
    "listen",
    "run",
    "stop",
]
