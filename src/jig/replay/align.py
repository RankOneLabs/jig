"""Fail-soft identity resolution for replay alignment.

Given a tool call's arguments and a tool's declared ``identity_fields``
(see :class:`jig.core.types.ToolDefinition`), resolve a composite key that
identifies "the same real-world entity" across calls, so that replay
alignment can pair historical and new tool calls beyond simple ordering.

Resolution is deliberately all-or-nothing and never raises: malformed or
missing span data means the entity is unknown, not a crash.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

IdentityKey = tuple[str, tuple[tuple[str, Any], ...]]


@dataclass
class ToolEvent:
    name: str
    args: Any
    output: str | None
    error: str | None

_VALID_LEAF_TYPES = (str, int, float, bool)

_MISSING = object()


def resolve_identity(
    tool_name: str,
    arguments: Any,
    identity_fields: list[str] | None,
) -> IdentityKey | None:
    """Resolve a composite identity key for a tool call, or ``None``.

    Returns ``None`` when ``identity_fields`` is ``None``, ``arguments`` is
    not a dict, or any declared field fails to resolve to a finite JSON
    scalar (missing path, non-dict intermediate, unsupported leaf type).
    """
    if identity_fields is None:
        return None
    if not isinstance(arguments, dict):
        return None

    resolved: list[tuple[str, Any]] = []
    for path in identity_fields:
        value = _resolve_path(tool_name, path, arguments)
        if value is _MISSING:
            return None
        resolved.append((type(value).__name__, value))

    return (tool_name, tuple(resolved))


def _resolve_path(tool_name: str, path: str, arguments: dict) -> Any:
    current: Any = arguments
    for segment in path.split("."):
        if not isinstance(current, dict):
            logger.debug(
                "resolve_identity: tool %r field %r not resolved "
                "(non-dict intermediate at segment %r)",
                tool_name,
                path,
                segment,
            )
            return _MISSING
        if segment not in current:
            logger.debug(
                "resolve_identity: tool %r field %r not resolved "
                "(missing segment %r)",
                tool_name,
                path,
                segment,
            )
            return _MISSING
        current = current[segment]

    if type(current) not in _VALID_LEAF_TYPES:
        logger.debug(
            "resolve_identity: tool %r field %r not resolved "
            "(unsupported leaf type %r)",
            tool_name,
            path,
            type(current).__name__,
        )
        return _MISSING
    if type(current) is float and not math.isfinite(current):
        logger.debug(
            "resolve_identity: tool %r field %r not resolved "
            "(non-finite float %r)",
            tool_name,
            path,
            current,
        )
        return _MISSING
    return current
