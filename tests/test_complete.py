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
            instance.aclose = AsyncMock()

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
            instance.aclose = AsyncMock()

            msg = Message(role=Role.USER, content="hello")
            await complete("claude-sonnet-4-5", [msg])

            params: CompletionParams = instance.complete.call_args[0][0]
            assert params.messages[0] is msg

    async def test_forwards_system_and_params(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())
            instance.aclose = AsyncMock()

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
            instance.aclose = AsyncMock()

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

    async def test_message_dict_missing_role_raises(self):
        with pytest.raises(ValueError, match="'role' field"):
            await complete("claude-sonnet-4-5", [{"content": "hi"}])

    async def test_message_dict_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Invalid message role 'bot'"):
            await complete(
                "claude-sonnet-4-5",
                [{"role": "bot", "content": "hi"}],
            )

    async def test_non_dict_message_rejected(self):
        with pytest.raises(ValueError, match="Message or dict"):
            await complete(
                "claude-sonnet-4-5",
                [None],  # type: ignore[list-item]
            )

    async def test_non_string_role_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid message role type int"):
            await complete(
                "claude-sonnet-4-5",
                [{"role": 123, "content": "hi"}],  # type: ignore[list-item]
            )

    async def test_tool_calls_dict_coerced_to_toolcall(self):
        """Assistant messages with dict-form tool_calls get proper ToolCall objects."""
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())
            instance.aclose = AsyncMock()

            await complete(
                "claude-sonnet-4-5",
                [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "tc-1", "name": "echo", "arguments": {"text": "x"}},
                        ],
                    },
                ],
            )

            from jig.core.types import ToolCall

            params: CompletionParams = instance.complete.call_args[0][0]
            tc = params.messages[1].tool_calls[0]
            assert isinstance(tc, ToolCall)
            assert tc.name == "echo"
            assert tc.arguments == {"text": "x"}

    async def test_tool_calls_missing_field_rejected(self):
        with pytest.raises(ValueError, match="missing required field"):
            await complete(
                "claude-sonnet-4-5",
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "echo", "arguments": {}}],
                    },
                ],
            )

    async def test_aclose_called_after_complete(self):
        """complete() releases adapter resources after use."""
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())
            instance.aclose = AsyncMock()

            await complete("claude-sonnet-4-5", [{"role": "user", "content": "hi"}])

            instance.aclose.assert_awaited_once()

    async def test_aclose_called_even_when_complete_raises(self):
        """complete() cleans up even if the underlying call fails."""
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(side_effect=RuntimeError("boom"))
            instance.aclose = AsyncMock()

            with pytest.raises(RuntimeError, match="boom"):
                await complete(
                    "claude-sonnet-4-5",
                    [{"role": "user", "content": "hi"}],
                )

            instance.aclose.assert_awaited_once()
