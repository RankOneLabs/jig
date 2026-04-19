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


def listen(**kwargs):
    """Start (or return) the process-wide callback listener.

    See :func:`jig.dispatch.listener.listen` for kwargs. Imported lazily
    so callers who don't use the callback path don't pay for aiohttp.
    """
    from jig.dispatch.listener import listen as _listen
    return _listen(**kwargs)


def stop():
    """Stop the process-wide callback listener, if any. Idempotent."""
    from jig.dispatch.listener import stop as _stop
    return _stop()


__all__ = [
    "DispatchError",
    "JobTimeoutError",
    "aclose",
    "listen",
    "run",
    "stop",
]
