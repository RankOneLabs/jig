"""Dispatch primitives: route LLM calls and deterministic functions to smithers.

The smithers dispatch server accepts several task types — ``inference`` for
LLM calls (used by :class:`jig.llm.DispatchClient`) and ``function`` for
named Python callables registered on the worker side (used by
:func:`run` below). Both paths share a submit-then-poll primitive in
:mod:`jig.dispatch.client`.

Future (phase 10): callbacks replace polling so large sweeps don't need
one polling coroutine per in-flight job.
"""
from jig.dispatch.client import (
    DispatchError,
    JobTimeoutError,
    aclose,
    run,
)

__all__ = ["DispatchError", "JobTimeoutError", "aclose", "run"]
