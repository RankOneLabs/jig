"""Replay tool registry — serves canned outputs from a recorded trace.

Substitutes for :class:`jig.tools.ToolRegistry` during replay. Every
``TOOL_CALL`` span in the recorded trace (except ``submit_output``,
which is runner-internal) becomes a canned result keyed by
``(tool_name, canonical_args_json)``.

On ``execute(call)``, the registry looks up a canned result:

- **Strict hit** — deque for this key has entries; pop the first,
  return it.
- **Duplicate keys** are served in recorded (FIFO) order so multiple
  identical calls replay correctly.
- **Strict miss** — raises :class:`ReplayMissError`; the replay caller
  chose to fail fast on divergence.
- **Lenient miss with fallback** — delegates to the fallback registry,
  running the tool live.
- **Lenient miss without fallback** — returns a ``ToolResult`` whose
  ``error`` says "replay miss", matching how an unknown tool would
  surface.

The registry also exposes the recorded ``ToolDefinition`` list (built
either from the fallback's definitions or — absent a fallback — from
minimal stubs derived from the observed call names). That keeps the
LLM's view of "what tools exist" faithful to the recording.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from jig.core.types import Span, SpanKind, Tool, ToolCall, ToolDefinition, ToolResult
from jig.replay.errors import ReplayMissError
from jig.tools.registry import ToolRegistry


# Local re-export to avoid a circular-ish import on runner.py.
_SUBMIT_OUTPUT_TOOL = "submit_output"


@dataclass
class _CannedResult:
    output: str
    error: str | None


def _canonical_args(args: Any) -> str:
    """Deterministic string key for a tool's arguments dict.

    ``json.dumps(..., sort_keys=True)`` gives the same key regardless of
    insertion order. Falls back to ``str(args)`` if the args aren't JSON-
    serializable (datetimes, bytes) so replay doesn't crash — strict
    mode will still miss cleanly since the recording used the same
    fallback.
    """
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def _extract_tool_spans(spans: list[Span]) -> list[Span]:
    """Return TOOL_CALL spans in order, skipping the ``submit_output`` helper."""
    return [
        s for s in spans
        if s.kind == SpanKind.TOOL_CALL and s.name != _SUBMIT_OUTPUT_TOOL
    ]


class ReplayToolRegistry(ToolRegistry):
    """A :class:`ToolRegistry` that serves canned outputs from a trace.

    Constructed from a recorded trace's spans; ``execute(call)`` returns
    the recorded output for matching ``(name, canonical_args)`` keys.
    """

    def __init__(
        self,
        recorded_spans: list[Span],
        *,
        fallback: ToolRegistry | None = None,
        strict: bool = False,
        definitions: list[ToolDefinition] | None = None,
    ) -> None:
        # Deliberately skip ``super().__init__()`` — the base constructor
        # wires up live tools, timeouts, dispatch routing; none of that
        # applies here. We do initialize the attributes the base reads
        # so ``list()`` / ``get()`` stay compatible.
        self._tools: dict[str, Tool] = {}
        self._execute_timeout: float | None = None
        self._dispatch_url: str | None = None

        self._fallback = fallback
        self._strict = strict
        self._canned: dict[tuple[str, str], deque[_CannedResult]] = {}

        for span in _extract_tool_spans(recorded_spans):
            key = (span.name, _canonical_args(span.input))
            output = span.output if isinstance(span.output, str) else (
                "" if span.output is None else str(span.output)
            )
            self._canned.setdefault(key, deque()).append(
                _CannedResult(output=output, error=span.error),
            )

        if definitions is not None:
            self._definitions = list(definitions)
        elif fallback is not None:
            self._definitions = fallback.list()
        else:
            # Last resort: synthesize minimal definitions from observed
            # names. LLMs need *some* tool list to call against; empty-
            # schema params mean the model can still emit calls but
            # won't get rich parameter hints. Usually the caller passes
            # a fallback or explicit definitions, so this is fine.
            seen: set[str] = set()
            self._definitions = []
            for span in _extract_tool_spans(recorded_spans):
                if span.name in seen:
                    continue
                seen.add(span.name)
                self._definitions.append(ToolDefinition(
                    name=span.name,
                    description=f"Replay stub for {span.name!r}",
                    parameters={"type": "object", "properties": {}},
                ))

    def list(self) -> list[ToolDefinition]:
        return list(self._definitions)

    def register(self, tool: Tool) -> None:  # pragma: no cover - unused path
        raise RuntimeError(
            "ReplayToolRegistry is immutable — register tools on the "
            "fallback registry instead."
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        key = (call.name, _canonical_args(call.arguments))
        bucket = self._canned.get(key)
        if bucket:
            canned = bucket.popleft()
            return ToolResult(
                call_id=call.id,
                output=canned.output,
                error=canned.error,
            )

        if self._strict:
            raise ReplayMissError(call.name, key[1])

        if self._fallback is not None:
            return await self._fallback.execute(call)

        return ToolResult(
            call_id=call.id,
            output="",
            error=f"Replay miss for tool {call.name!r} and no fallback registry",
        )
