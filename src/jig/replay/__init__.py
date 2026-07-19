"""Phase 11 — agent replay from a recorded trace.

Two entry points:

- :func:`replay` — rerun an agent run identified by ``trace_id``, with
  tool outputs served from the recording and the LLM (and optionally
  other config knobs) swapped via ``config_override``. The LLM is live;
  only tool results are pinned.
- :func:`trace_diff` — structured diff between two recorded traces.
  Aligns each trace's ``TOOL_CALL`` span stream via
  :mod:`jig.replay.align`: a falsey ``identity_fields`` (``None`` or
  ``{}``) preserves legacy ordinal (zip-by-position) pairing; a
  non-empty ``identity_fields`` map opts the whole diff into the
  three-tier identity-aware aligner. Only the "identity" tier asserts
  entity-level event continuity — "anchor" and "ordinal" pairs are
  structural position matches, not proof that the paired events are
  semantically the same call. Reports the first field that diverges
  per aligned pair, and rolls up final-output, grader-score, cost,
  latency, and error-category deltas.

**Registry-driven identity_fields.** Build ``trace_diff``'s
``identity_fields`` argument straight from a live
:class:`~jig.tools.ToolRegistry` with :func:`identity_map` (exported
from both ``jig.replay`` and ``jig``)::

    await trace_diff(
        trace_a_id, trace_b_id,
        tracer=tracer,
        identity_fields=identity_map(registry.list()),
    )

A registry where no tool declares ``identity_fields`` yields ``{}``
from :func:`identity_map`; since that's falsey, ``trace_diff`` takes
its legacy-ordinal branch and pairs both traces by exact
zip-by-position, unconditionally — the caller never has to branch on
whether any tool opted in. As soon as one tool declares
``identity_fields``, the map is non-empty and the whole diff opts
into the three-tier aligner: tools present in the map pair by
identity where possible, but tools *absent* from a non-empty map are
still identity-less and may be paired by patience anchors or
segment-local ordinal fallback instead — a different result than
exact legacy ordinal pairing for those specific tools.

**Scope — what replay covers.** Tool outputs from the recording.

**Scope — what replay does NOT cover (deferred):**

- LLM response replay. ``LLM_CALL`` spans truncate content to 200
  chars and don't store the full request; replay always calls the
  live LLM. This is intentional — the whole point is to vary the LLM
  while holding tools constant.
- Memory / retriever replay. ``MEMORY_QUERY`` spans run live against
  whatever ``store`` / ``retriever`` the caller supplies. Pass a
  deterministic store (or ``None``) to avoid nondeterminism.
- Grader replay. Graders run live.
- Dispatch trace replay. Dispatch boundaries aren't re-materialized —
  a replayed run can dispatch fresh, or (when the tool is recorded as
  a normal ``TOOL_CALL``) its canned output substitutes as with any
  other tool.
"""
from jig.replay.align import identity_map
from jig.replay.diff import TraceDiff, ToolDiff, ToolEvent, trace_diff
from jig.replay.errors import (
    ReplayConfigMissingError,
    ReplayError,
    ReplayMissError,
    ReplaySchemaMismatchError,
)
from jig.replay.registry import ReplayToolRegistry
from jig.replay.runner import replay
from jig.replay.snapshot import reconstruct_config, serialize_config

__all__ = [
    "ReplayError",
    "ReplayConfigMissingError",
    "ReplayMissError",
    "ReplaySchemaMismatchError",
    "ReplayToolRegistry",
    "TraceDiff",
    "ToolDiff",
    "ToolEvent",
    "identity_map",
    "reconstruct_config",
    "replay",
    "serialize_config",
    "trace_diff",
]
