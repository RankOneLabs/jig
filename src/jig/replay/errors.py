"""Replay-specific error hierarchy.

Replay errors are distinct from :class:`jig.core.errors.AgentError` —
they indicate a problem with *replay itself* (missing trace data, tool
arg drift under strict mode, schema changes), not a terminal agent
outcome. They inherit from :class:`JigError` so callers can catch the
whole jig family in one place.
"""
from __future__ import annotations

from jig.core.errors import JigError


class ReplayError(JigError):
    """Base class for replay-specific failures."""


class ReplayConfigMissingError(ReplayError):
    """Recorded trace is missing the config snapshot.

    Phase 11 requires :func:`jig.core.runner._serialize_config_snapshot`
    to have stamped the root span's metadata. Older traces (from before
    phase 11) lack this and can't be replayed as-is.
    """


class ReplaySchemaMismatchError(ReplayError):
    """The recorded ``output_schema`` can't be reconciled with replay.

    Raised when:
    - the recorded schema FQN no longer resolves to an importable class;
    - the caller's ``config_override`` changes ``output_schema`` to a
      different type than what was recorded.

    Parsed-output shape changes are out of phase 11 scope — rerun the
    original agent rather than replaying across shape changes.
    """


class ReplayMissError(ReplayError):
    """Strict replay saw a tool call it has no recorded answer for.

    Indicates the replay diverged from the recorded execution path.
    The attached ``tool_name`` and ``canonical_args`` identify the
    missing key; the caller can loosen to ``strict=False`` to fall
    through to live execution, or supply a fallback registry.
    """

    def __init__(self, tool_name: str, canonical_args: str):
        super().__init__(
            f"Replay miss for tool {tool_name!r} with args {canonical_args!r} "
            f"— no recorded output and strict=True"
        )
        self.tool_name = tool_name
        self.canonical_args = canonical_args
