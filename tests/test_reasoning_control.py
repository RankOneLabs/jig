"""Tests for the portable ``CompletionParams.reasoning`` switch.

``None`` leaves every request untouched. ``True``/``False`` is translated
by adapters with a native reasoning control (Ollama ``think``, OpenRouter
``reasoning.enabled``) and rejected with ``UnsupportedReasoningError``
before any request by adapters without one.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import jig
from jig.core.errors import UnsupportedReasoningError
from jig.core.types import CompletionParams, Message, Role
from jig.llm.anthropic import AnthropicClient
from jig.llm.google import GeminiClient
from jig.llm.ollama import OllamaClient
from jig.llm.openai import OpenAIClient
from jig.llm.openrouter import OpenRouterClient


def _params(reasoning: bool | None) -> CompletionParams:
    return CompletionParams(messages=[Message(role=Role.USER, content="hi")], reasoning=reasoning)


def _fake_openai_response() -> SimpleNamespace:
    message = SimpleNamespace(content="hi", tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage, model="m")


def _fake_ollama_response() -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content="hi", tool_calls=None),
        prompt_eval_count=1,
        eval_count=1,
    )


def test_defaults_to_none() -> None:
    assert _params(None).reasoning is None


def test_error_is_exported_value_error() -> None:
    assert jig.UnsupportedReasoningError is UnsupportedReasoningError
    assert issubclass(UnsupportedReasoningError, ValueError)


class TestOllama:
    @pytest.mark.parametrize("reasoning", [True, False])
    async def test_forwards_think(self, reasoning: bool) -> None:
        with patch("jig.llm.ollama.OllamaAsyncClient") as mock_cls:
            mock_cls.return_value.chat = AsyncMock(return_value=_fake_ollama_response())
            client = OllamaClient(model="gemma4:26b")
            await client.complete(_params(reasoning))
            assert mock_cls.return_value.chat.call_args.kwargs["think"] is reasoning

    async def test_none_sends_no_think(self) -> None:
        with patch("jig.llm.ollama.OllamaAsyncClient") as mock_cls:
            mock_cls.return_value.chat = AsyncMock(return_value=_fake_ollama_response())
            client = OllamaClient(model="gemma4:26b")
            await client.complete(_params(None))
            assert "think" not in mock_cls.return_value.chat.call_args.kwargs

    async def test_think_is_not_a_model_option(self) -> None:
        with patch("jig.llm.ollama.OllamaAsyncClient") as mock_cls:
            mock_cls.return_value.chat = AsyncMock(return_value=_fake_ollama_response())
            client = OllamaClient(model="gemma4:26b")
            await client.complete(_params(False))
            assert "think" not in mock_cls.return_value.chat.call_args.kwargs.get("options", {})


class TestOpenRouter:
    async def test_forwards_reasoning_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=_fake_openai_response())
            client = OpenRouterClient(model="google/gemma-4-26b-a4b-it")
            await client.complete(_params(False))
            extra_body = instance.chat.completions.create.call_args.kwargs["extra_body"]
            assert extra_body["reasoning"] == {"enabled": False}
            assert extra_body["usage"] == {"include": True}

    async def test_none_leaves_extra_body_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=_fake_openai_response())
            client = OpenRouterClient(model="google/gemma-4-26b-a4b-it")
            await client.complete(_params(None))
            extra_body = instance.chat.completions.create.call_args.kwargs["extra_body"]
            assert "reasoning" not in extra_body

    async def test_caller_provider_params_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=_fake_openai_response())
            client = OpenRouterClient(model="google/gemma-4-26b-a4b-it")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")],
                reasoning=False,
                provider_params={"extra_body": {"reasoning": {"enabled": True, "effort": "low"}}},
            )
            await client.complete(params)
            extra_body = instance.chat.completions.create.call_args.kwargs["extra_body"]
            assert extra_body["reasoning"] == {"enabled": True, "effort": "low"}


class TestRejectingAdapters:
    async def test_openai_rejects_before_request(self) -> None:
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=_fake_openai_response())
            client = OpenAIClient(model="gpt-4o", api_key="sk-test")
            with pytest.raises(UnsupportedReasoningError):
                await client.complete(_params(False))
            instance.chat.completions.create.assert_not_called()

    async def test_gemini_rejects_before_request(self) -> None:
        with patch("jig.llm.google.genai") as mock_genai:
            client = GeminiClient(model="gemini-2.5-flash", api_key="k")
            with pytest.raises(UnsupportedReasoningError):
                await client.complete(_params(True))
            mock_genai.Client.return_value.aio.models.generate_content.assert_not_called()

    async def test_anthropic_rejects_before_request(self) -> None:
        with patch("jig.llm.anthropic.anthropic") as mock_anthropic:
            instance = mock_anthropic.AsyncAnthropic.return_value
            instance.messages.create = AsyncMock()
            client = AnthropicClient(model="claude-sonnet-5", api_key="k")
            with pytest.raises(UnsupportedReasoningError):
                await client.complete(_params(False))
            instance.messages.create.assert_not_called()
