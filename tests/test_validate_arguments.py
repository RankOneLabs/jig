"""Tests for §1.5 of the jig lifecycle-hooks spec:

``ToolRegistry(validate_arguments=True)`` — opt-in pre-execute JSON Schema
validation of ``call.arguments`` against ``tool.definition.parameters``,
checked once ahead of the local-execute/dispatch branch so it covers both
paths, and (for ``dispatch=True`` tools) fires before ``pre_dispatch`` so a
malformed call never ships.

See ``comms/jig-lifecycle-hooks-spec.md`` §1.5 in the gecko repo for the
spec and the empirical basis for the ``best_match``/``iter_errors``
combination pinned below: plain ``next(iter_errors(...))`` reports an
``anyOf`` failure (an ``int | None`` param) or a ``$ref`` failure (a nested
model) against the *union* itself — useless to the model — while
``best_match`` alone loses the property path and the ``description`` text
that carries the field's own prose. §1.5 exists specifically to fix that
combination; the anyOf test below is the regression canary.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

# validate_arguments requires the optional `validate` extra. Skip cleanly
# when absent, same convention as test_dispatch_callback.py's aiohttp guard.
pytest.importorskip("jsonschema")

from jig import ToolCall, ToolDefinition, ToolRegistry
from jig.core.types import Tool

# Mirrors what pydantic generates for:
#
#   class Nested(BaseModel):
#       depth: int = Field(ge=1)
#
#   class Params(BaseModel):
#       max_drawdown_pct: float = Field(gt=0, description="Never negative.")
#       seed: int | None = Field(default=None, ge=0, description="Omit for random.")
#       nested: Nested | None = Field(default=None, description="Optional nesting.")
#
# (verified against pydantic 2.x's actual model_json_schema() output —
# notably exclusiveMinimum for gt=0, and the description living on the
# *property* schema for the anyOf/$ref cases, not inside the $ref target).
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "max_drawdown_pct": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Never negative.",
        },
        "seed": {
            "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}],
            "description": "Omit for random.",
        },
        "nested": {
            "anyOf": [{"$ref": "#/$defs/Nested"}, {"type": "null"}],
            "description": "Optional nesting.",
        },
    },
    "required": ["max_drawdown_pct"],
    "$defs": {
        "Nested": {
            "type": "object",
            "properties": {"depth": {"type": "integer", "minimum": 1}},
            "required": ["depth"],
        },
    },
}

VALID_ARGS = {"max_drawdown_pct": 1.0, "seed": 5, "nested": {"depth": 2}}


class _LocalTool(Tool):
    """Non-dispatch tool. Records every args dict it was actually called
    with, so tests can tell "validation rejected the call" apart from
    "validation passed the call through and the tool did something with
    it" — both look like "execute() ran" from a weaker assertion."""

    def __init__(self) -> None:
        self.executed_with: list[dict[str, Any]] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="local", description="d", parameters=_SCHEMA)

    async def execute(self, args: dict[str, Any]) -> str:
        self.executed_with.append(args)
        return "ok"


class _DispatchedTool(Tool):
    """dispatch=True tool with a pre_dispatch that records whether it ran,
    so tests can confirm validation happens strictly before it."""

    dispatch = True

    def __init__(self) -> None:
        self.pre_dispatch_calls = 0

    @property
    def dispatch_fn_ref(self) -> str | None:
        return "pkg.module:fn"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="dispatched", description="d", parameters=_SCHEMA)

    def pre_dispatch(self, arguments: dict[str, Any] | None, context: Any) -> None:
        self.pre_dispatch_calls += 1

    async def execute(self, args: dict[str, Any]) -> str:
        raise AssertionError("execute() must not be called for a dispatch=True tool")


def _call(name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id="call-1", name=name, arguments=arguments or {})


# --- default off: behavior unchanged, no import -----------------------------


async def test_default_off_bad_arguments_pass_through_to_local_execute():
    """validate_arguments defaults to False: a call that would fail schema
    validation still reaches execute() untouched, exactly as before this
    feature existed."""
    tool = _LocalTool()
    result = await ToolRegistry([tool]).execute(_call("local", {"max_drawdown_pct": -30}))

    assert result.error is None
    assert result.output == "ok"
    assert tool.executed_with == [{"max_drawdown_pct": -30}]


async def test_default_off_never_imports_jsonschema(monkeypatch):
    """No import cost for non-adopters: force any jsonschema import to
    blow up, then confirm a default registry never triggers it even when
    invoked with schema-invalid arguments."""
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    tool = _LocalTool()
    result = await ToolRegistry([tool]).execute(_call("local", {"max_drawdown_pct": -30}))

    assert result.error is None
    assert result.output == "ok"


# --- valid arguments pass through unchanged, both paths ---------------------


async def test_valid_arguments_pass_through_local():
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("local", VALID_ARGS))

    assert result.error is None
    assert result.output == "ok"
    assert tool.executed_with == [VALID_ARGS]


async def test_valid_arguments_pass_through_dispatched(monkeypatch):
    import jig.dispatch

    async def fake_run(fn_ref, payload=None, **kwargs):
        return {"ok": True}

    monkeypatch.setattr(jig.dispatch, "run", fake_run)
    tool = _DispatchedTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("dispatched", VALID_ARGS))

    assert result.error is None
    assert tool.pre_dispatch_calls == 1


# --- five error shapes, exact rendered message pinned -----------------------


async def test_plain_constrained_field_message():
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("local", {"max_drawdown_pct": -30}))

    assert result.output == ""
    assert result.error == (
        "schema: -30 is less than or equal to the minimum of 0 "
        "at max_drawdown_pct — Never negative."
    )
    assert tool.executed_with == []


async def test_anyof_optional_field_message():
    """The regression this feature exists to prevent: plain
    next(iter_errors(...)) reports '-1 is not valid under any of the
    given schemas' for an anyOf (int | None) field — the specific bound
    violation and the field's description both get lost. This pinned
    message is the canary: if the combining logic collapses back to a
    single jsonschema call, this test fails first."""
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("local", {"max_drawdown_pct": 1.0, "seed": -1}))

    assert result.error == "schema: -1 is less than the minimum of 0 at seed — Omit for random."
    assert tool.executed_with == []


async def test_ref_nested_field_message():
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(
        _call("local", {"max_drawdown_pct": 1.0, "nested": {"depth": 0}})
    )

    assert result.error == "schema: 0 is less than the minimum of 1 at nested — Optional nesting."


async def test_missing_required_property_message():
    """Root-level error: err.path is an empty deque, so the ' at <path>'
    suffix must be omitted entirely rather than left dangling as
    '... at ' with nothing after it."""
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("local", {}))

    assert result.error == "schema: 'max_drawdown_pct' is a required property"
    assert not result.error.endswith(" at")
    assert not result.error.endswith("at ")


async def test_wrong_type_message():
    tool = _LocalTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("local", {"max_drawdown_pct": "thirty"}))

    assert result.error == (
        "schema: 'thirty' is not of type 'number' at max_drawdown_pct — Never negative."
    )


# --- dispatched: rejected before pre_dispatch fires / before anything ships -


async def test_dispatch_rejected_before_pre_dispatch_and_before_shipping(monkeypatch):
    import jig.dispatch

    dispatch_calls = 0

    async def fake_run(fn_ref, payload=None, **kwargs):
        nonlocal dispatch_calls
        dispatch_calls += 1
        return {"should": "never happen"}

    monkeypatch.setattr(jig.dispatch, "run", fake_run)
    tool = _DispatchedTool()
    registry = ToolRegistry([tool], validate_arguments=True)
    result = await registry.execute(_call("dispatched", {"max_drawdown_pct": -30}))

    assert dispatch_calls == 0
    assert tool.pre_dispatch_calls == 0
    assert result.error == (
        "schema: -30 is less than or equal to the minimum of 0 "
        "at max_drawdown_pct — Never negative."
    )


# --- ImportError when the extra is missing ----------------------------------


async def test_import_error_when_jsonschema_missing_is_actionable(monkeypatch):
    """Simulates the extra not being installed without uninstalling it:
    sys.modules['jsonschema'] = None makes the next `import jsonschema`
    raise ImportError immediately, the same as the package being absent,
    even though it is genuinely installed in this test environment."""
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    tool = _LocalTool()

    with pytest.raises(ImportError, match=r"jig\[validate\]"):
        ToolRegistry([tool], validate_arguments=True)
