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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import jig
from jig.core.errors import UnsupportedResponseFormatError
from jig.core.types import CompletionParams, Message, Role
from jig.llm.ollama import OllamaClient
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


def _fake_ollama_response() -> SimpleNamespace:
    message = SimpleNamespace(content="{}", tool_calls=None)
    return SimpleNamespace(message=message, prompt_eval_count=1, eval_count=1)


def _make_ollama_client() -> OllamaClient:
    client = OllamaClient.__new__(OllamaClient)
    client._client = MagicMock()
    client._model = "llama3.1"
    return client


@pytest.mark.asyncio
class TestOllamaResponseFormat:
    async def test_valid_schema_translated_to_format_kwarg(self):
        client = _make_ollama_client()
        client._client.chat = AsyncMock(return_value=_fake_ollama_response())
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="hi")],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "the answer",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        await client.complete(params)

        call_kwargs = client._client.chat.call_args.kwargs
        assert call_kwargs["format"] == schema
        assert "response_format" not in call_kwargs

    async def test_omitted_when_not_set(self):
        client = _make_ollama_client()
        client._client.chat = AsyncMock(return_value=_fake_ollama_response())

        params = CompletionParams(messages=[Message(role=Role.USER, content="hi")])
        await client.complete(params)

        call_kwargs = client._client.chat.call_args.kwargs
        assert "format" not in call_kwargs

    @pytest.mark.parametrize(
        "response_format",
        [
            pytest.param("json", id="non_dict"),
            pytest.param({"type": "text"}, id="wrong_type"),
            pytest.param({"type": "json_schema"}, id="missing_json_schema"),
            pytest.param(
                {"type": "json_schema", "json_schema": "nope"},
                id="json_schema_not_mapping",
            ),
            pytest.param(
                {"type": "json_schema", "json_schema": {"name": "x"}},
                id="missing_schema_key",
            ),
            pytest.param(
                {"type": "json_schema", "json_schema": {"schema": {}}},
                id="empty_schema",
            ),
            pytest.param(
                {"type": "json_schema", "json_schema": {"schema": "nope"}},
                id="schema_not_a_mapping",
            ),
        ],
    )
    async def test_malformed_response_format_rejected_before_transport(
        self, response_format,
    ):
        client = _make_ollama_client()
        client._client.chat = AsyncMock(return_value=_fake_ollama_response())

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="hi")],
            response_format=response_format,
        )
        with pytest.raises(UnsupportedResponseFormatError):
            await client.complete(params)

        client._client.chat.assert_not_called()
