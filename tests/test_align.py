"""Tests for tool-identity declaration (``ToolDefinition.identity_fields``)
and fail-soft resolution (``jig.replay.align.resolve_identity``).
"""
from __future__ import annotations

import math

import pytest

from jig.core.types import ToolDefinition
from jig.replay.align import resolve_identity


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
