"""Tests for tool-identity declaration (``ToolDefinition.identity_fields``),
fail-soft resolution (``jig.replay.align.resolve_identity``), and the
three-tier deterministic alignment engine (identity, anchor, ordinal).
"""
from __future__ import annotations

import math
import random
from itertools import zip_longest

import pytest

from jig.core.types import ToolDefinition
from jig.replay.align import (
    AlignedPair,
    Alignment,
    OrdinalAligner,
    UnmatchedEvent,
    ToolEvent,
    _tier1_identity,
    resolve_identity,
)


def _ev(name: str = "t", args: dict | None = None) -> ToolEvent:
    return ToolEvent(name=name, args=args if args is not None else {}, output=None, error=None)


def _make_tool(identity_fields: list[str] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name="lookup_customer",
        description="Look up a customer record",
        parameters={"type": "object", "properties": {}},
        identity_fields=identity_fields,
    )


# --- ToolDefinition.identity_fields construction contract ---


def test_identity_fields_defaults_to_none():
    tool = ToolDefinition(
        name="t", description="d", parameters={"type": "object", "properties": {}}
    )
    assert tool.identity_fields is None


def test_identity_fields_none_is_valid():
    tool = _make_tool(None)
    assert tool.identity_fields is None


def test_identity_fields_flat_valid_declaration():
    tool = _make_tool(["customer_id"])
    assert tool.identity_fields == ["customer_id"]


def test_identity_fields_nested_valid_declaration():
    tool = _make_tool(["customer.id", "customer.region"])
    assert tool.identity_fields == ["customer.id", "customer.region"]


def test_identity_fields_rejects_empty_list():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool([])


def test_identity_fields_rejects_non_string_entry():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool(["customer_id", 123])  # type: ignore[list-item]


def test_identity_fields_rejects_bare_string_instead_of_list():
    # A bare str is iterable character-by-character, which would otherwise
    # silently resolve as a list of single-character identity fields.
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool("customer_id")  # type: ignore[arg-type]


def test_identity_fields_rejects_tuple_instead_of_list():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool(("customer_id",))  # type: ignore[arg-type]


def test_identity_fields_rejects_empty_string_entry():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool([""])


def test_identity_fields_rejects_leading_empty_dot_segment():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool([".customer_id"])


def test_identity_fields_rejects_trailing_empty_dot_segment():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool(["customer_id."])


def test_identity_fields_rejects_interior_empty_dot_segment():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool(["customer..id"])


def test_identity_fields_rejects_duplicate_path():
    with pytest.raises(ValueError, match="identity_fields"):
        _make_tool(["customer_id", "customer_id"])


# --- resolve_identity: top-level short circuits ---


def test_resolve_identity_none_identity_fields_returns_none():
    assert resolve_identity("lookup_customer", {"customer_id": "abc"}, None) is None


def test_resolve_identity_non_dict_arguments_returns_none():
    assert resolve_identity("lookup_customer", "not-a-dict", ["customer_id"]) is None
    assert resolve_identity("lookup_customer", None, ["customer_id"]) is None
    assert resolve_identity("lookup_customer", [1, 2], ["customer_id"]) is None


# --- resolve_identity: flat and nested traversal ---


def test_resolve_identity_flat_field():
    key = resolve_identity("lookup_customer", {"customer_id": "abc"}, ["customer_id"])
    assert key == ("lookup_customer", (("str", "abc"),))


def test_resolve_identity_nested_field():
    args = {"customer": {"id": "abc", "region": "us"}}
    key = resolve_identity(
        "lookup_customer", args, ["customer.id", "customer.region"]
    )
    assert key == ("lookup_customer", (("str", "abc"), ("str", "us")))


def test_resolve_identity_field_order_is_declaration_order():
    args = {"a": 1, "b": 2}
    key_ab = resolve_identity("t", args, ["a", "b"])
    key_ba = resolve_identity("t", args, ["b", "a"])
    assert key_ab == ("t", (("int", 1), ("int", 2)))
    assert key_ba == ("t", (("int", 2), ("int", 1)))


# --- resolve_identity: missing / malformed traversal ---


def test_resolve_identity_missing_leaf_returns_none():
    assert resolve_identity("t", {"other": "x"}, ["customer_id"]) is None


def test_resolve_identity_missing_intermediate_returns_none():
    assert resolve_identity("t", {}, ["customer.id"]) is None


def test_resolve_identity_non_dict_intermediate_returns_none():
    args = {"customer": "not-a-dict"}
    assert resolve_identity("t", args, ["customer.id"]) is None


def test_resolve_identity_non_dict_final_container_but_missing_key():
    args = {"customer": {"id": {"nested": "x"}}}
    # "customer.id.missing" — id resolves to a dict, missing key absent.
    assert resolve_identity("t", args, ["customer.id.missing"]) is None


def test_resolve_identity_partial_composite_returns_none():
    # customer.id resolves, customer.region does not exist -> whole call fails.
    args = {"customer": {"id": "abc"}}
    assert resolve_identity("t", args, ["customer.id", "customer.region"]) is None


# --- resolve_identity: leaf-value validation ---


def test_resolve_identity_none_leaf_returns_none():
    assert resolve_identity("t", {"x": None}, ["x"]) is None


def test_resolve_identity_list_leaf_returns_none():
    assert resolve_identity("t", {"x": [1, 2]}, ["x"]) is None


def test_resolve_identity_dict_leaf_returns_none():
    assert resolve_identity("t", {"x": {"y": 1}}, ["x"]) is None


def test_resolve_identity_nan_leaf_returns_none():
    assert resolve_identity("t", {"x": float("nan")}, ["x"]) is None


def test_resolve_identity_infinity_leaf_returns_none():
    assert resolve_identity("t", {"x": float("inf")}, ["x"]) is None
    assert resolve_identity("t", {"x": float("-inf")}, ["x"]) is None


def test_resolve_identity_custom_hashable_leaf_returns_none():
    class CustomHashable:
        def __hash__(self):
            return 1

    assert resolve_identity("t", {"x": CustomHashable()}, ["x"]) is None


def test_resolve_identity_valid_str_leaf():
    assert resolve_identity("t", {"x": "abc"}, ["x"]) == ("t", (("str", "abc"),))


def test_resolve_identity_valid_int_leaf():
    assert resolve_identity("t", {"x": 5}, ["x"]) == ("t", (("int", 5),))


def test_resolve_identity_valid_float_leaf():
    key = resolve_identity("t", {"x": 5.5}, ["x"])
    assert key is not None
    assert key[1][0][0] == "float"
    assert math.isclose(key[1][0][1], 5.5)


def test_resolve_identity_valid_bool_leaf():
    assert resolve_identity("t", {"x": True}, ["x"]) == ("t", (("bool", True),))


# --- resolve_identity: type-tag collisions (must not alias) ---


def test_resolve_identity_bool_and_int_do_not_collide():
    key_true = resolve_identity("t", {"x": True}, ["x"])
    key_one = resolve_identity("t", {"x": 1}, ["x"])
    assert key_true != key_one
    assert key_true == ("t", (("bool", True),))
    assert key_one == ("t", (("int", 1),))


def test_resolve_identity_int_and_float_do_not_collide():
    key_int = resolve_identity("t", {"x": 1}, ["x"])
    key_float = resolve_identity("t", {"x": 1.0}, ["x"])
    assert key_int != key_float
    assert key_int == ("t", (("int", 1),))
    assert key_float == ("t", (("float", 1.0),))


def test_resolve_identity_string_and_numeric_do_not_collide():
    key_str = resolve_identity("t", {"x": "1"}, ["x"])
    key_int = resolve_identity("t", {"x": 1}, ["x"])
    assert key_str != key_int
    assert key_str == ("t", (("str", "1"),))


def test_resolve_identity_false_and_zero_do_not_collide():
    key_false = resolve_identity("t", {"x": False}, ["x"])
    key_zero = resolve_identity("t", {"x": 0}, ["x"])
    assert key_false != key_zero
    assert key_false == ("t", (("bool", False),))
    assert key_zero == ("t", (("int", 0),))


# --- resolve_identity: tool-name scoping ---


def test_resolve_identity_scopes_by_tool_name():
    args = {"x": "abc"}
    key_a = resolve_identity("tool_a", args, ["x"])
    key_b = resolve_identity("tool_b", args, ["x"])
    assert key_a != key_b
    assert key_a[0] == "tool_a"
    assert key_b[0] == "tool_b"


# --- Alignment.validate(): invariant checks ---


def test_validate_accepts_trivial_ordinal_pair():
    a = [_ev()]
    b = [_ev()]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "ordinal")], only_a=[], only_b=[])
    alignment.validate(a, b)


def test_validate_rejects_missing_a_index():
    a = [_ev(), _ev()]
    b = [_ev()]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "ordinal")], only_a=[], only_b=[])
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_duplicated_a_index():
    a = [_ev()]
    b = [_ev(), _ev()]
    alignment = Alignment(
        pairs=[AlignedPair(0, 0, "ordinal"), AlignedPair(0, 1, "ordinal")],
        only_a=[],
        only_b=[],
    )
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_out_of_range_index():
    a = [_ev()]
    b = [_ev()]
    alignment = Alignment(pairs=[AlignedPair(0, 1, "ordinal")], only_a=[], only_b=[])
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_unsorted_pairs():
    a = [_ev(), _ev()]
    b = [_ev(), _ev()]
    alignment = Alignment(
        pairs=[AlignedPair(1, 1, "ordinal"), AlignedPair(0, 0, "ordinal")],
        only_a=[],
        only_b=[],
    )
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_unsorted_only_a():
    a = [_ev(), _ev()]
    b = []
    alignment = Alignment(
        pairs=[],
        only_a=[UnmatchedEvent(1, "ordinal"), UnmatchedEvent(0, "ordinal")],
        only_b=[],
    )
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_unsorted_only_b():
    a = []
    b = [_ev(), _ev()]
    alignment = Alignment(
        pairs=[],
        only_a=[],
        only_b=[UnmatchedEvent(1, "ordinal"), UnmatchedEvent(0, "ordinal")],
    )
    with pytest.raises(AssertionError):
        alignment.validate(a, b)


def test_validate_rejects_keyed_pair_at_wrong_tier():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "abc"})]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "ordinal")], only_a=[], only_b=[])
    with pytest.raises(AssertionError):
        alignment.validate(a, b, identity_fields={"lookup": ["id"]})


def test_validate_rejects_mismatched_keys_at_identity_tier():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "xyz"})]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "identity")], only_a=[], only_b=[])
    with pytest.raises(AssertionError):
        alignment.validate(a, b, identity_fields={"lookup": ["id"]})


def test_validate_rejects_keyed_versus_unkeyed_pair():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("other", {})]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "anchor")], only_a=[], only_b=[])
    with pytest.raises(AssertionError):
        alignment.validate(a, b, identity_fields={"lookup": ["id"]})


def test_validate_accepts_equal_key_identity_pair():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "abc"})]
    alignment = Alignment(pairs=[AlignedPair(0, 0, "identity")], only_a=[], only_b=[])
    alignment.validate(a, b, identity_fields={"lookup": ["id"]})


# --- OrdinalAligner ---


def test_ordinal_aligner_empty_both_sides():
    alignment = OrdinalAligner().align([], [])
    assert alignment.pairs == []
    assert alignment.only_a == []
    assert alignment.only_b == []
    alignment.validate([], [])


def test_ordinal_aligner_equal_length_all_paired():
    a = [_ev("x"), _ev("y"), _ev("z")]
    b = [_ev("x"), _ev("y2"), _ev("z")]
    alignment = OrdinalAligner().align(a, b)
    assert alignment.pairs == [
        AlignedPair(0, 0, "ordinal"),
        AlignedPair(1, 1, "ordinal"),
        AlignedPair(2, 2, "ordinal"),
    ]
    assert alignment.only_a == []
    assert alignment.only_b == []
    alignment.validate(a, b)


def test_ordinal_aligner_a_longer_overhang_only_a():
    a = [_ev("x"), _ev("y"), _ev("z")]
    b = [_ev("x")]
    alignment = OrdinalAligner().align(a, b)
    assert alignment.pairs == [AlignedPair(0, 0, "ordinal")]
    assert alignment.only_a == [UnmatchedEvent(1, "ordinal"), UnmatchedEvent(2, "ordinal")]
    assert alignment.only_b == []
    alignment.validate(a, b)


def test_ordinal_aligner_b_longer_overhang_only_b():
    a = [_ev("x")]
    b = [_ev("x"), _ev("y"), _ev("z")]
    alignment = OrdinalAligner().align(a, b)
    assert alignment.pairs == [AlignedPair(0, 0, "ordinal")]
    assert alignment.only_a == []
    assert alignment.only_b == [UnmatchedEvent(1, "ordinal"), UnmatchedEvent(2, "ordinal")]
    alignment.validate(a, b)


def test_ordinal_aligner_ignores_identity_fields():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "xyz"})]
    alignment = OrdinalAligner().align(a, b, identity_fields={"lookup": ["id"]})
    assert alignment.pairs == [AlignedPair(0, 0, "ordinal")]


def _legacy_zip_longest_alignment(a, b):
    """Vendored test-only copy of the pre-cohort ``trace_diff`` pairing
    loop (see git history of ``jig/replay/diff.py`` before this cohort):
    ``for idx, (a_span, b_span) in enumerate(zip_longest(a_tools, b_tools))``,
    classifying each position as paired / only_a / only_b by presence.
    Kept independent of ``OrdinalAligner`` so the compatibility check
    below isn't just comparing the implementation to itself.
    """
    sentinel = object()
    pair_indices = []
    only_a_indices = []
    only_b_indices = []
    for idx, (av, bv) in enumerate(zip_longest(a, b, fillvalue=sentinel)):
        if av is sentinel:
            only_b_indices.append(idx)
        elif bv is sentinel:
            only_a_indices.append(idx)
        else:
            pair_indices.append(idx)
    return pair_indices, only_a_indices, only_b_indices


def _random_tool_event(rng: random.Random) -> ToolEvent:
    name = rng.choice(["alpha", "beta", "gamma"])
    args = {"k": rng.choice([1, 2, "x", None])}
    output = rng.choice([None, "ok", "err-ish"])
    error = rng.choice([None, "boom"])
    return ToolEvent(name=name, args=args, output=output, error=error)


def test_ordinal_aligner_matches_legacy_zip_longest_pairing():
    rng = random.Random(0)
    aligner = OrdinalAligner()
    for _ in range(500):
        len_a = rng.randint(0, 8)
        len_b = rng.randint(0, 8)
        a = [_random_tool_event(rng) for _ in range(len_a)]
        b = [_random_tool_event(rng) for _ in range(len_b)]

        expected_pairs, expected_only_a, expected_only_b = _legacy_zip_longest_alignment(a, b)

        alignment = aligner.align(a, b)
        alignment.validate(a, b)

        assert [p.index_a for p in alignment.pairs] == expected_pairs
        assert [p.index_b for p in alignment.pairs] == expected_pairs
        assert all(p.tier == "ordinal" for p in alignment.pairs)
        assert [u.index for u in alignment.only_a] == expected_only_a
        assert all(u.tier == "ordinal" for u in alignment.only_a)
        assert [u.index for u in alignment.only_b] == expected_only_b
        assert all(u.tier == "ordinal" for u in alignment.only_b)


# --- _tier1_identity ---


def test_tier1_equal_key_pairing():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == [AlignedPair(0, 0, "identity")]
    assert only_a == []
    assert only_b == []
    assert rem_a == []
    assert rem_b == []


def test_tier1_a_only_keyed_call_is_unmatched():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "xyz"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == []
    assert only_a == [UnmatchedEvent(0, "identity")]
    assert only_b == [UnmatchedEvent(0, "identity")]
    assert rem_a == []
    assert rem_b == []


def test_tier1_b_only_keyed_call_is_unmatched():
    a: list[ToolEvent] = []
    b = [_ev("lookup", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == []
    assert only_a == []
    assert only_b == [UnmatchedEvent(0, "identity")]
    assert rem_b == []


def test_tier1_duplicate_group_a_surplus():
    a = [_ev("lookup", {"id": "abc"}), _ev("lookup", {"id": "abc"}), _ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == [AlignedPair(0, 0, "identity")]
    assert only_a == [UnmatchedEvent(1, "identity"), UnmatchedEvent(2, "identity")]
    assert only_b == []


def test_tier1_duplicate_group_b_surplus():
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"id": "abc"}), _ev("lookup", {"id": "abc"}), _ev("lookup", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == [AlignedPair(0, 0, "identity")]
    assert only_a == []
    assert only_b == [UnmatchedEvent(1, "identity"), UnmatchedEvent(2, "identity")]


def test_tier1_duplicate_groups_both_surplus_types_in_one_call():
    # key "x": a has 2, b has 1 -> a surplus. key "y": a has 1, b has 2 -> b surplus.
    a = [
        _ev("lookup", {"id": "x"}),
        _ev("lookup", {"id": "x"}),
        _ev("lookup", {"id": "y"}),
    ]
    b = [
        _ev("lookup", {"id": "x"}),
        _ev("lookup", {"id": "y"}),
        _ev("lookup", {"id": "y"}),
    ]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert set((p.index_a, p.index_b) for p in pairs) == {(0, 0), (2, 1)}
    assert all(p.tier == "identity" for p in pairs)
    assert only_a == [UnmatchedEvent(1, "identity")]
    assert only_b == [UnmatchedEvent(2, "identity")]


def test_tier1_keyed_versus_unkeyed_non_pairing():
    # a's keyed event must never pair with b's same-name but unkeyed event.
    a = [_ev("lookup", {"id": "abc"})]
    b = [_ev("lookup", {"other": "field"})]  # missing "id" -> unresolved, falls to remainder
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"lookup": ["id"]})
    assert pairs == []
    assert only_a == [UnmatchedEvent(0, "identity")]
    assert only_b == []
    assert rem_a == []
    assert [i for i, _ in rem_b] == [0]


def test_tier1_no_identity_fields_everything_is_remainder():
    a = [_ev("lookup", {"id": "abc"}), _ev("other")]
    b = [_ev("lookup", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, None)
    assert pairs == []
    assert only_a == []
    assert only_b == []
    assert [i for i, _ in rem_a] == [0, 1]
    assert [i for i, _ in rem_b] == [0]


def test_tier1_missing_tool_entry_in_mapping_is_identity_less():
    a = [_ev("undeclared", {"id": "abc"})]
    b = [_ev("undeclared", {"id": "abc"})]
    pairs, only_a, only_b, rem_a, rem_b = _tier1_identity(a, b, {"other_tool": ["id"]})
    assert pairs == []
    assert [i for i, _ in rem_a] == [0]
    assert [i for i, _ in rem_b] == [0]
