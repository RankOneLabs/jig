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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import jig
from jig.core.errors import UnsupportedResponseFormatError
from jig.core.types import CompletionParams, Message, Role
from jig.llm.openai import OpenAIClient

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
    },
}


def _fake_openai_response() -> SimpleNamespace:
    message = SimpleNamespace(content="hi", tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o")


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


@pytest.mark.asyncio
class TestOpenAIResponseFormat:
    async def test_forwarded_unchanged(self):
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_openai_response()
            )
            client = OpenAIClient(model="gpt-4o")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")],
                response_format=_RESPONSE_FORMAT,
            )
            await client.complete(params)

            create_kwargs = instance.chat.completions.create.call_args.kwargs
            assert create_kwargs["response_format"] == _RESPONSE_FORMAT

    async def test_omitted_when_not_set(self):
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_openai_response()
            )
            client = OpenAIClient(model="gpt-4o")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")],
            )
            await client.complete(params)

            create_kwargs = instance.chat.completions.create.call_args.kwargs
            assert "response_format" not in create_kwargs
