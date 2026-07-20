"""Tests for the portable response_format structured-output transport.

CompletionParams.response_format carries the OpenAI-compatible envelope
({"type": "json_schema", "json_schema": {"schema": {...}}}) from callers
through to adapters unmodified. Adapters that speak the OpenAI-compatible
wire shape (dispatch, OpenAI, OpenRouter) forward it unchanged; Ollama
validates and translates it to its native `format` field; adapters
without structured-output support reject a non-null value with
UnsupportedResponseFormatError before making any request.
"""
from __future__ import annotations

import jig
from jig.core.errors import UnsupportedResponseFormatError
from jig.core.types import CompletionParams, Message, Role


class TestCompletionParamsResponseFormat:
    def test_defaults_to_none(self):
        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        assert params.response_format is None

    def test_preserves_caller_object_unmodified(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
        params = CompletionParams(
            messages=[Message(role=Role.USER, content="hi")],
            response_format=rf,
        )
        assert params.response_format is rf
        assert params.response_format == rf


class TestUnsupportedResponseFormatError:
    def test_is_a_value_error(self):
        assert issubclass(UnsupportedResponseFormatError, ValueError)

    def test_importable_from_top_level_and_core(self):
        assert jig.UnsupportedResponseFormatError is UnsupportedResponseFormatError
        from jig.core import UnsupportedResponseFormatError as CoreError
        assert CoreError is UnsupportedResponseFormatError

    def test_message_preserved(self):
        err = UnsupportedResponseFormatError("no can do")
        assert str(err) == "no can do"
