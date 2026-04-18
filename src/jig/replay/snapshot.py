"""Serialize / reconstruct :class:`AgentConfig` state for replay.

Replay records the config's *state fields* (model hints, prompt text,
tuning knobs, output schema FQN) on the AGENT_RUN span. *Live objects*
— LLM clients, tracers, tool registries, memory stores, feedback stores
— are not serializable and must be re-supplied by the caller of
:func:`jig.replay`.

This split keeps the snapshot small and deterministic while letting
replay run against a *different* model (the whole point — vary the LLM,
pin the tools).
"""
from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel

from jig.core.runner import AgentConfig, _serialize_config_snapshot
from jig.core.types import (
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryStore,
    Retriever,
    TracingLogger,
)
from jig.replay.errors import ReplaySchemaMismatchError
from jig.tools.registry import ToolRegistry


__all__ = ["serialize_config", "reconstruct_config"]


def serialize_config(config: AgentConfig[Any]) -> dict[str, Any]:
    """Public alias for the runner's internal snapshot helper.

    Exposed so tests and tools can produce the same snapshot shape the
    runner writes onto the AGENT_RUN span.
    """
    return _serialize_config_snapshot(config)


def _resolve_output_schema(fqn: str) -> type[BaseModel]:
    """Import ``module:ClassName`` and verify it's a BaseModel subclass.

    **Trust model.** This function calls :func:`importlib.import_module`
    on a string sourced from recorded trace metadata. Jig treats the
    trace store as trusted input — traces are written by the caller's
    own agent runs against the caller's own tracer. Do **not** feed
    :func:`jig.replay` a trace_id sourced from an untrusted user (e.g.
    a shared multi-tenant database or a webhook payload) without an
    import allowlist, because an attacker who can seed the tracer's
    storage can pick any importable module to load.

    In the homelab single-tenant use case this is fine; anyone
    considering a broader deployment should wrap this with an
    allowlist check on the ``fqn``'s module prefix.
    """
    if ":" not in fqn:
        raise ReplaySchemaMismatchError(
            f"Recorded output_schema FQN {fqn!r} is malformed "
            f"(expected 'module:ClassName')"
        )
    module_name, _, qualname = fqn.partition(":")
    # PEP 3155: function-local classes have ``<locals>`` segments in
    # their qualname and cannot be reached via getattr after the
    # defining function returns. Detect up front so the caller sees a
    # clear error instead of an opaque AttributeError on the walk.
    if "<locals>" in qualname:
        raise ReplaySchemaMismatchError(
            f"Recorded output_schema {fqn!r} is a function-local class "
            f"and is not replayable — move the pydantic model to module "
            f"scope and re-record the trace. ``config_override`` cannot "
            f"substitute a different output_schema, so there is no "
            f"in-place fix."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ReplaySchemaMismatchError(
            f"Recorded output_schema {fqn!r} can't be imported: {e}"
        ) from e
    obj: Any = module
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise ReplaySchemaMismatchError(
                f"Recorded output_schema {fqn!r} no longer resolves "
                f"({part!r} missing on {module_name!r})"
            ) from e
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ReplaySchemaMismatchError(
            f"Recorded output_schema {fqn!r} resolved to {obj!r}, "
            f"which is not a pydantic BaseModel subclass"
        )
    return obj


def reconstruct_config(
    snapshot: dict[str, Any],
    override: dict[str, Any] | AgentConfig[Any] | None,
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    tracer: TracingLogger,
    feedback: FeedbackLoop,
    store: MemoryStore | None = None,
    retriever: Retriever | None = None,
    grader: Grader[Any] | None = None,
) -> AgentConfig[Any]:
    """Rebuild an :class:`AgentConfig` from a recorded snapshot.

    ``override`` is either a dict of fields to apply via
    :meth:`AgentConfig.with_`, or a full ``AgentConfig`` replacement.
    When it's a full config, the caller-supplied live components on
    that config take precedence and this function returns it verbatim —
    useful for "throw the recording away and run this config instead,"
    though then most of replay's value is gone.

    Raises :class:`ReplaySchemaMismatchError` when the recorded schema
    can't be resolved or when ``override`` changes ``output_schema`` to
    something different from the recording.
    """
    if isinstance(override, AgentConfig):
        return override

    override_dict: dict[str, Any] = dict(override or {})

    # Resolve recorded output_schema (if any) and reconcile with override.
    recorded_schema_fqn = snapshot.get("output_schema")
    override_schema = override_dict.get("output_schema", _SENTINEL)
    if override_schema is not _SENTINEL:
        # Any presence change (adding a schema to a plain-text recording,
        # or dropping the recorded schema) flips whether ``submit_output``
        # gets injected. That's a semantic shift, not just a shape swap —
        # reject the override so the replay stays comparable to the
        # recording.
        recorded_cls = (
            _resolve_output_schema(recorded_schema_fqn)
            if recorded_schema_fqn is not None
            else None
        )
        if override_schema is not recorded_cls:
            raise ReplaySchemaMismatchError(
                f"override changes output_schema from {recorded_schema_fqn!r} "
                f"to {override_schema!r}; adding, removing, or swapping the "
                f"parsed-output schema is out of replay scope"
            )
        output_schema = override_schema
        override_dict.pop("output_schema")
    elif recorded_schema_fqn is not None:
        output_schema = _resolve_output_schema(recorded_schema_fqn)
    else:
        output_schema = None

    # System prompt: if recorded as a callable, caller must supply one
    # via override. Otherwise reuse the recorded string.
    if snapshot.get("system_prompt_is_callable"):
        if "system_prompt" not in override_dict:
            raise ReplaySchemaMismatchError(
                "Recorded system_prompt was a callable and is not serializable; "
                "pass a new system_prompt in config_override."
            )
        system_prompt = override_dict.pop("system_prompt")
    else:
        system_prompt = override_dict.pop(
            "system_prompt", snapshot.get("system_prompt", ""),
        )

    base = AgentConfig(
        name=snapshot.get("agent_name", "replay"),
        description=snapshot.get("description", ""),
        llm=llm,
        tools=tools,
        tracer=tracer,
        feedback=feedback,
        system_prompt=system_prompt,
        store=store,
        retriever=retriever,
        grader=grader,
        max_tool_calls=snapshot.get("max_tool_calls", 10),
        max_llm_calls=snapshot.get("max_llm_calls", 50),
        max_llm_retries=snapshot.get("max_llm_retries", 3),
        max_parse_retries=snapshot.get("max_parse_retries", 2),
        include_memory_in_prompt=snapshot.get("include_memory_in_prompt", True),
        include_feedback_in_prompt=snapshot.get("include_feedback_in_prompt", True),
        session_id=snapshot.get("session_id"),
        output_schema=output_schema,
    )

    if override_dict:
        return base.with_(**override_dict)
    return base


_SENTINEL: Any = object()
