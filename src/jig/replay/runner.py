"""``jig.replay(trace_id, ...)`` — rerun an agent with canned tool outputs.

The replay reuses the recorded config snapshot to rebuild an
:class:`AgentConfig`, but substitutes a :class:`ReplayToolRegistry` for
``config.tools``. Every tool call the model emits is served from the
recording (matched by tool name + canonical args). The *LLM* runs live
— that's the point: vary the model, hold tool results constant,
compare via :func:`jig.trace_diff`.

Live components — ``llm``, ``tracer``, ``feedback``, optionally
``store``, ``retriever``, ``grader`` — can't be serialized and must be
supplied by the caller. A ``tools_fallback`` is optional: when
``strict=False`` and a tool call has no matching recording, the
fallback runs it live; without a fallback, the lenient path surfaces a
``ToolResult.error`` so the agent loop can continue.
"""
from __future__ import annotations

from typing import Any

from jig.core.runner import AgentConfig, AgentResult, run_agent
from jig.core.types import (
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryStore,
    Retriever,
    SpanKind,
    TracingLogger,
)
from jig.replay.errors import ReplayConfigMissingError
from jig.replay.registry import ReplayToolRegistry
from jig.replay.snapshot import reconstruct_config
from jig.tools.registry import ToolRegistry


async def replay(
    trace_id: str,
    config_override: dict[str, Any] | AgentConfig[Any] | None = None,
    *,
    tracer: TracingLogger,
    llm: LLMClient,
    feedback: FeedbackLoop,
    store: MemoryStore | None = None,
    retriever: Retriever | None = None,
    grader: Grader[Any] | None = None,
    tools_fallback: ToolRegistry | None = None,
    strict: bool = False,
) -> AgentResult[Any]:
    """Rerun a recorded agent trace with canned tool outputs.

    Loads the trace via ``tracer.get_trace(trace_id)``, pulls the
    config snapshot off the root ``AGENT_RUN`` span, and invokes
    :func:`run_agent` against a :class:`ReplayToolRegistry` built from
    the recorded ``TOOL_CALL`` spans.

    ``config_override`` applies during reconstruction of the recorded
    snapshot. Pass a dict of fields to tweak specific knobs
    (``{"max_tool_calls": 5}``) or a full :class:`AgentConfig` to
    replace the reconstructed config broadly. In **either** case, replay
    re-pins its own live components afterwards, so ``llm``, ``tools``,
    ``tracer``, ``feedback``, ``store``, ``retriever``, and ``grader``
    always come from this function's arguments — a full
    :class:`AgentConfig` override can't swap out
    :class:`ReplayToolRegistry` (that would defeat the whole point of
    replay). Full-``AgentConfig`` overrides are not validated against
    the recorded snapshot; tuning knobs dicts are.

    ``strict=True`` raises :class:`ReplayMissError` on any tool call
    without a recorded match — useful to verify a replay actually
    reproduces the recorded path. ``strict=False`` (default) falls
    through to ``tools_fallback`` when provided, or returns a miss
    error on the tool result otherwise.

    Returns a fresh :class:`AgentResult` tied to a new trace — the
    replay itself is recorded via ``tracer``.
    """
    spans = await tracer.get_trace(trace_id)
    if not spans:
        raise ReplayConfigMissingError(
            f"No spans found for trace_id={trace_id!r}"
        )

    root = next(
        (s for s in spans if s.parent_id is None and s.kind == SpanKind.AGENT_RUN),
        None,
    )
    if root is None:
        raise ReplayConfigMissingError(
            f"Trace {trace_id!r} has no AGENT_RUN root span"
        )

    metadata = root.metadata or {}
    snapshot = metadata.get("config")
    if not isinstance(snapshot, dict):
        raise ReplayConfigMissingError(
            f"Trace {trace_id!r} is missing the config snapshot — "
            f"likely predates phase 11 trace capture. Rerun the agent "
            f"to produce a replayable trace."
        )

    recorded_input = metadata.get("input", "")

    replay_tools = ReplayToolRegistry(
        spans, fallback=tools_fallback, strict=strict,
    )

    config = reconstruct_config(
        snapshot,
        config_override,
        llm=llm,
        tools=replay_tools,
        tracer=tracer,
        feedback=feedback,
        store=store,
        retriever=retriever,
        grader=grader,
    )

    # Pin replay-owned live components last. ``reconstruct_config``
    # honors a full-``AgentConfig`` override verbatim and a dict
    # override can still carry ``tools``/``tracer``/``feedback``/etc.
    # — either route would silently let real tools run, defeating the
    # whole point of replay. Re-apply replay's components after
    # reconstruction so they always win.
    config = config.with_(
        llm=llm,
        tools=replay_tools,
        tracer=tracer,
        feedback=feedback,
        store=store,
        retriever=retriever,
        grader=grader,
    )

    return await run_agent(config, recorded_input)
