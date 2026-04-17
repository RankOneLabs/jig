"""Tests for jig.complete one-shot helper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from jig import complete
from jig.core.types import CompletionParams, LLMResponse, Message, Role, Usage


def _mock_response() -> LLMResponse:
    return LLMResponse(
        content="ok",
        tool_calls=None,
        usage=Usage(input_tokens=10, output_tokens=5, cost=0.001),
        latency_ms=1.0,
        model="claude-sonnet-4-5",
    )


@pytest.mark.asyncio
class TestComplete:
    async def test_message_dict_coercion(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())

            response = await complete(
                "claude-sonnet-4-5",
                [{"role": "user", "content": "hi"}],
            )

            assert response.content == "ok"
            params: CompletionParams = instance.complete.call_args[0][0]
            assert isinstance(params.messages[0], Message)
            assert params.messages[0].role == Role.USER
            assert params.messages[0].content == "hi"

    async def test_passthrough_message_objects(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())

            msg = Message(role=Role.USER, content="hello")
            await complete("claude-sonnet-4-5", [msg])

            params: CompletionParams = instance.complete.call_args[0][0]
            assert params.messages[0] is msg

    async def test_forwards_system_and_params(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())

            await complete(
                "claude-sonnet-4-5",
                [{"role": "user", "content": "x"}],
                system="You are helpful.",
                temperature=0.3,
                max_tokens=256,
            )

            params: CompletionParams = instance.complete.call_args[0][0]
            assert params.system == "You are helpful."
            assert params.temperature == 0.3
            assert params.max_tokens == 256

    async def test_client_kwargs_passed_to_constructor(self):
        with patch("jig.llm.ollama.OllamaClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())

            await complete(
                "ollama/llama3.1",
                [{"role": "user", "content": "x"}],
                client_kwargs={"host": "http://mcbain:11434"},
            )

            mock_cls.assert_called_once_with(
                model="llama3.1", host="http://mcbain:11434"
            )

    async def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="No provider matches"):
            await complete("no-such-model", [{"role": "user", "content": "x"}])
