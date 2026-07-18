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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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


# --- Alignment model ---
#
# Three tiers, in descending assertion strength:
#
#   "identity" — both events resolved the same composite key via
#       ``resolve_identity``. This is the only tier that asserts
#       entity-level event continuity: the two calls are believed to be
#       about the same real-world thing.
#   "anchor"   — both events share a tool name that is otherwise unique
#       on its side of the identity-less remainder. This is a
#       structural coincidence, not an identity assertion.
#   "ordinal"  — positional fallback within an anchor-delimited segment
#       (or, for :class:`OrdinalAligner`, over the whole sequence). Pure
#       position matching, no assertion about the events being related.
#
# Identity matching is order-insensitive (duplicate calls to the same
# entity pair up ordinally within their group, regardless of what else
# is happening around them); anchor and ordinal tiers are structural
# comparisons over source position. Whether the *sequence* of pairs
# itself is in the expected order is downstream business — see
# ``TrajectoryGrader`` — not something this module asserts.

AlignmentTier = Literal["identity", "anchor", "ordinal"]


@dataclass(frozen=True)
class AlignedPair:
    index_a: int
    index_b: int
    tier: AlignmentTier


@dataclass(frozen=True)
class UnmatchedEvent:
    index: int
    tier: AlignmentTier


@dataclass
class Alignment:
    pairs: list[AlignedPair]
    only_a: list[UnmatchedEvent]
    only_b: list[UnmatchedEvent]

    def validate(
        self,
        a: Sequence[ToolEvent],
        b: Sequence[ToolEvent],
        identity_fields: Mapping[str, list[str]] | None = None,
    ) -> None:
        """Test-facing invariant checker; not called in production.

        Raises an ``AssertionError`` when: a side's indices (across its
        pairs and its unmatched list) are not exactly ``range(len(side))``
        — i.e. any index is missing, duplicated, or out of range; ``pairs``
        is not ascending by ``index_a``; ``only_a`` or ``only_b`` is not
        ascending by ``index``; or a pair involving an event that resolves
        an identity key (per ``identity_fields``) is not itself an
        equal-key ``"identity"``-tier pair.
        """
        a_indices = sorted([p.index_a for p in self.pairs] + [u.index for u in self.only_a])
        assert a_indices == list(range(len(a))), (
            f"pairs/only_a index_a values {a_indices} are not exactly "
            f"range(len(a))={list(range(len(a)))} (missing, duplicated, "
            f"or out-of-range index)"
        )
        b_indices = sorted([p.index_b for p in self.pairs] + [u.index for u in self.only_b])
        assert b_indices == list(range(len(b))), (
            f"pairs/only_b index_b values {b_indices} are not exactly "
            f"range(len(b))={list(range(len(b)))} (missing, duplicated, "
            f"or out-of-range index)"
        )

        pair_a_order = [p.index_a for p in self.pairs]
        assert pair_a_order == sorted(pair_a_order), (
            f"pairs not ascending by index_a: {pair_a_order}"
        )
        only_a_order = [u.index for u in self.only_a]
        assert only_a_order == sorted(only_a_order), (
            f"only_a not ascending by index: {only_a_order}"
        )
        only_b_order = [u.index for u in self.only_b]
        assert only_b_order == sorted(only_b_order), (
            f"only_b not ascending by index: {only_b_order}"
        )

        for p in self.pairs:
            event_a = a[p.index_a]
            event_b = b[p.index_b]
            fields_a = identity_fields.get(event_a.name) if identity_fields else None
            fields_b = identity_fields.get(event_b.name) if identity_fields else None
            key_a = resolve_identity(event_a.name, event_a.args, fields_a)
            key_b = resolve_identity(event_b.name, event_b.args, fields_b)
            if key_a is not None or key_b is not None:
                assert p.tier == "identity" and key_a is not None and key_a == key_b, (
                    f"pair (index_a={p.index_a}, index_b={p.index_b}, "
                    f"tier={p.tier!r}) involves a resolvable keyed event "
                    f"(key_a={key_a!r}, key_b={key_b!r}) but is not an "
                    f"equal-key identity pair"
                )


class Aligner(Protocol):
    def align(
        self,
        a: list[ToolEvent],
        b: list[ToolEvent],
        *,
        identity_fields: Mapping[str, list[str]] | None = None,
    ) -> Alignment: ...
