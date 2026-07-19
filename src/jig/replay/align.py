"""Deterministic identity-aware alignment for replayed tool calls.

Only an ``identity``-tier pair asserts that two calls are the same
entity-level event. ``anchor`` and ``ordinal`` pairs are structural
comparisons that keep the diff useful without claiming semantic event
continuity. Identity matching is order-insensitive, while ordering
assertions remain the responsibility of :class:`TrajectoryGrader`.

Patience anchors are computed from the identity-less remainder of both
traces independently of tier-1 structure. Consequently, no ordering
constraint is imposed between identity pairs and the anchor/ordinal tiers.
Within anchor-delimited ambiguous segments, calls pair only by ordinal
position; this module never infers identity from similarity.

Identity resolution uses a tool's declared ``identity_fields`` (see
:class:`jig.core.types.ToolDefinition`) and is deliberately all-or-nothing.
Malformed or missing span data means the entity is unknown, not a crash.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Literal, Protocol

from jig.core.types import ToolDefinition

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


def identity_map(tools: Iterable[ToolDefinition]) -> dict[str, list[str]]:
    """Build a ``{tool_name: identity_fields}`` map for :func:`trace_diff`.

    Extracts and copies each :class:`ToolDefinition`'s declared
    ``identity_fields``, keyed by tool name; definitions with
    ``identity_fields=None`` are skipped, not represented with an
    empty or ``None`` entry. Each declared list is copied via
    ``list(...)`` so later mutation of a ``ToolDefinition``'s list
    cannot mutate the returned map.

    Returns ``{}`` when ``tools`` is empty or no definition declares
    identity fields — this is falsey, so passing the result straight
    through to ``trace_diff(identity_fields=...)`` naturally preserves
    legacy ordinal pairing for an all-undeclared registry.
    """
    result: dict[str, list[str]] = {}
    for tool in tools:
        if tool.identity_fields is None:
            continue
        result[tool.name] = list(tool.identity_fields)
    return result


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


_OVERHANG = object()


class OrdinalAligner:
    """Legacy pairing: zip by source position, no identity awareness.

    Pairs index ``i`` with ``i`` through ``min(len(a), len(b))``; the
    longer side's overhang becomes sorted ordinal-tier unmatched events.
    ``identity_fields`` is accepted (to satisfy :class:`Aligner`) and
    ignored.
    """

    def align(
        self,
        a: list[ToolEvent],
        b: list[ToolEvent],
        *,
        identity_fields: Mapping[str, list[str]] | None = None,
    ) -> Alignment:
        pairs: list[AlignedPair] = []
        only_a: list[UnmatchedEvent] = []
        only_b: list[UnmatchedEvent] = []
        for i, (event_a, event_b) in enumerate(zip_longest(a, b, fillvalue=_OVERHANG)):
            if event_a is _OVERHANG:
                only_b.append(UnmatchedEvent(index=i, tier="ordinal"))
            elif event_b is _OVERHANG:
                only_a.append(UnmatchedEvent(index=i, tier="ordinal"))
            else:
                pairs.append(AlignedPair(index_a=i, index_b=i, tier="ordinal"))
        return Alignment(pairs=pairs, only_a=only_a, only_b=only_b)


def _resolve_event(
    event: ToolEvent, identity_fields: Mapping[str, list[str]] | None
) -> IdentityKey | None:
    fields = identity_fields.get(event.name) if identity_fields is not None else None
    return resolve_identity(event.name, event.args, fields)


def _tier1_identity(
    a: list[ToolEvent],
    b: list[ToolEvent],
    identity_fields: Mapping[str, list[str]] | None,
) -> tuple[
    list[AlignedPair],
    list[UnmatchedEvent],
    list[UnmatchedEvent],
    list[tuple[int, ToolEvent]],
    list[tuple[int, ToolEvent]],
]:
    """Identity-tier pairing plus the identity-less remainder of each side.

    Every event that resolves a composite key (via ``resolve_identity``)
    is permanently consumed here — paired ordinally within its key's
    group, or left as identity-tier unmatched surplus — and never
    appears in the returned remainders. Groups are processed in
    first-a-encounter order, then b-only keys in first-b-encounter
    order, so the result does not depend on dict iteration order.

    Returns ``(pairs, only_a, only_b, remainder_a, remainder_b)`` where
    the remainders are ``(original_index, event)`` pairs for
    identity-less events, in source order.
    """
    groups: dict[IdentityKey, dict[str, list[int]]] = {}
    key_order: list[IdentityKey] = []

    key_a: list[IdentityKey | None] = []
    for i, event in enumerate(a):
        key = _resolve_event(event, identity_fields)
        key_a.append(key)
        if key is not None:
            if key not in groups:
                groups[key] = {"a": [], "b": []}
                key_order.append(key)
            groups[key]["a"].append(i)

    key_b: list[IdentityKey | None] = []
    for j, event in enumerate(b):
        key = _resolve_event(event, identity_fields)
        key_b.append(key)
        if key is not None:
            if key not in groups:
                groups[key] = {"a": [], "b": []}
                key_order.append(key)
            groups[key]["b"].append(j)

    pairs: list[AlignedPair] = []
    only_a: list[UnmatchedEvent] = []
    only_b: list[UnmatchedEvent] = []
    for key in key_order:
        a_idxs = groups[key]["a"]
        b_idxs = groups[key]["b"]
        n = min(len(a_idxs), len(b_idxs))
        for k in range(n):
            pairs.append(AlignedPair(index_a=a_idxs[k], index_b=b_idxs[k], tier="identity"))
        for idx in a_idxs[n:]:
            only_a.append(UnmatchedEvent(index=idx, tier="identity"))
        for idx in b_idxs[n:]:
            only_b.append(UnmatchedEvent(index=idx, tier="identity"))

    remainder_a = [(i, event) for i, event in enumerate(a) if key_a[i] is None]
    remainder_b = [(j, event) for j, event in enumerate(b) if key_b[j] is None]

    return pairs, only_a, only_b, remainder_a, remainder_b


def _tier2_anchors(
    remainder_a: list[tuple[int, ToolEvent]],
    remainder_b: list[tuple[int, ToolEvent]],
) -> list[AlignedPair]:
    """Deterministic patience-diff anchors over the identity-less remainder.

    A candidate anchor is a tool name that occurs exactly once in each
    remainder. Candidates are naturally ordered by ascending
    ``index_a`` (remainder order preserves source order); the selected
    anchors are the longest strictly-increasing-in-``index_b``
    subsequence of candidates, i.e. the classic patience-diff LIS. When
    multiple maximum-length subsequences exist, the lexicographically
    smallest sequence of ``(index_a, index_b)`` pairs is chosen — since
    candidates are already ascending by ``index_a``, this reduces to
    always preferring the earliest eligible candidate at each step of
    the reconstruction below.

    Unselected (including crossing) candidates are simply not returned;
    they remain in the remainder as ordinary members of whatever
    anchor-delimited segment they fall into.
    """
    count_a: dict[str, int] = {}
    for _, event in remainder_a:
        count_a[event.name] = count_a.get(event.name, 0) + 1
    count_b: dict[str, int] = {}
    for _, event in remainder_b:
        count_b[event.name] = count_b.get(event.name, 0) + 1

    b_index_by_name: dict[str, int] = {}
    for orig_idx, event in remainder_b:
        if count_b[event.name] == 1:
            b_index_by_name[event.name] = orig_idx

    # Ascending by index_a by construction: remainder_a preserves source order.
    candidates: list[tuple[int, int]] = [
        (orig_idx, b_index_by_name[event.name])
        for orig_idx, event in remainder_a
        if count_a[event.name] == 1 and count_b.get(event.name) == 1
    ]

    m = len(candidates)
    if m == 0:
        return []

    # f[k] = length of the longest strictly-increasing-in-index_b run
    # of candidates starting at k (inclusive).
    f = [1] * m
    for k in range(m - 2, -1, -1):
        best = 0
        v_k = candidates[k][1]
        for j in range(k + 1, m):
            if candidates[j][1] > v_k and f[j] > best:
                best = f[j]
        f[k] = 1 + best

    max_len = max(f)
    selected: list[tuple[int, int]] = []
    remaining = max_len
    last_b = -1
    for k in range(m):
        if remaining == 0:
            break
        a_idx, b_idx = candidates[k]
        if f[k] == remaining and b_idx > last_b:
            selected.append((a_idx, b_idx))
            last_b = b_idx
            remaining -= 1

    return [AlignedPair(index_a=a_idx, index_b=b_idx, tier="anchor") for a_idx, b_idx in selected]


def _segment(
    remainder: list[tuple[int, ToolEvent]], anchor_orig_indices: list[int]
) -> list[list[tuple[int, ToolEvent]]]:
    """Split ``remainder`` into ``len(anchor_orig_indices) + 1`` runs.

    Anchor elements themselves are dropped (they're already paired at
    tier 2); everything else lands in the run before, between, or
    after the anchors that bound it.
    """
    anchor_set = set(anchor_orig_indices)
    segments: list[list[tuple[int, ToolEvent]]] = [[]]
    for orig_idx, event in remainder:
        if orig_idx in anchor_set:
            segments.append([])
        else:
            segments[-1].append((orig_idx, event))
    return segments


def _tier3_ordinal_segments(
    remainder_a: list[tuple[int, ToolEvent]],
    remainder_b: list[tuple[int, ToolEvent]],
    anchors: list[AlignedPair],
) -> tuple[list[AlignedPair], list[UnmatchedEvent], list[UnmatchedEvent]]:
    """Ordinal fallback, local to each anchor-delimited segment.

    Partitions both remainders around the selected anchors (before,
    between, after), zips corresponding segments by position, and
    emits each segment's overhang as sorted ordinal-tier unmatched.
    Crossing or otherwise-unselected tier-2 candidates fall into a
    segment here like any other identity-less event — their overhang,
    if any, carries ordinal (not anchor) provenance.
    """
    anchors_sorted = sorted(anchors, key=lambda p: p.index_a)
    segments_a = _segment(remainder_a, [p.index_a for p in anchors_sorted])
    segments_b = _segment(remainder_b, [p.index_b for p in anchors_sorted])

    pairs: list[AlignedPair] = []
    only_a: list[UnmatchedEvent] = []
    only_b: list[UnmatchedEvent] = []

    for seg_a, seg_b in zip(segments_a, segments_b):
        n = min(len(seg_a), len(seg_b))
        for k in range(n):
            pairs.append(AlignedPair(index_a=seg_a[k][0], index_b=seg_b[k][0], tier="ordinal"))
        for orig_idx, _ in seg_a[n:]:
            only_a.append(UnmatchedEvent(index=orig_idx, tier="ordinal"))
        for orig_idx, _ in seg_b[n:]:
            only_b.append(UnmatchedEvent(index=orig_idx, tier="ordinal"))

    return pairs, only_a, only_b


class IdentityAligner:
    """Three-tier deterministic alignment: identity, then anchor, then
    segment-local ordinal fallback.

    Only tier-1 ("identity") pairs assert entity-level event continuity
    — see the module-level note above. Identity matching is
    order-insensitive; anchors and ordinal fallback are purely
    structural. Whether the resulting pair *sequence* is in the
    expected order is downstream business, not asserted here.
    """

    def align(
        self,
        a: list[ToolEvent],
        b: list[ToolEvent],
        *,
        identity_fields: Mapping[str, list[str]] | None = None,
    ) -> Alignment:
        id_pairs, id_only_a, id_only_b, remainder_a, remainder_b = _tier1_identity(
            a, b, identity_fields
        )
        anchors = _tier2_anchors(remainder_a, remainder_b)
        ord_pairs, ord_only_a, ord_only_b = _tier3_ordinal_segments(
            remainder_a, remainder_b, anchors
        )

        pairs = sorted(id_pairs + anchors + ord_pairs, key=lambda p: p.index_a)
        only_a = sorted(id_only_a + ord_only_a, key=lambda u: u.index)
        only_b = sorted(id_only_b + ord_only_b, key=lambda u: u.index)
        return Alignment(pairs=pairs, only_a=only_a, only_b=only_b)
