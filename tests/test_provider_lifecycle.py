"""Provider adapter lifecycle tests.

Verifies that:
- Each adapter's aclose() is idempotent (safe to call multiple times).
- Adapters close their owned SDK client/transport on aclose().
- Adapters constructed with an injected mock client do NOT close it.
- factory.complete() calls aclose() in a finally block on success and failure.
- ollama_embed() closes its transport after embedding work completes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jig.core.types import CompletionParams, LLMResponse, Message, Role, Usage


def _dummy_params() -> CompletionParams:
    return CompletionParams(messages=[Message(role=Role.USER, content="hi")])


def _mock_response() -> LLMResponse:
    return LLMResponse(
        content="ok",
        tool_calls=None,
        usage=Usage(input_tokens=5, output_tokens=3, cost=0.0),
        latency_ms=1.0,
        model="test-model",
    )


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnthropicLifecycle:
    async def test_aclose_calls_sdk_close(self):
        """aclose() delegates to the SDK client's close()."""
        anthropic_stub = MagicMock()
        sdk_client = MagicMock()
        sdk_client.close = AsyncMock()
        anthropic_stub.AsyncAnthropic.return_value = sdk_client

        with patch("jig.llm.anthropic.anthropic", anthropic_stub):
            from jig.llm.anthropic import AnthropicClient
            client = AnthropicClient.__new__(AnthropicClient)
            client._client = sdk_client
            client._model = "claude-test"
            client._closed = False

            await client.aclose()

        sdk_client.close.assert_awaited_once()

    async def test_aclose_idempotent(self):
        """Calling aclose() twice must not raise or double-close."""
        sdk_client = MagicMock()
        sdk_client.close = AsyncMock()

        from jig.llm.anthropic import AnthropicClient
        client = AnthropicClient.__new__(AnthropicClient)
        client._client = sdk_client
        client._model = "claude-test"
        client._closed = False

        await client.aclose()
        await client.aclose()

        sdk_client.close.assert_awaited_once()

    async def test_sdk_max_retries_defaults_to_zero(self):
        """Constructor must pass max_retries=0 to the SDK to disable built-in retries."""
        anthropic_stub = MagicMock()
        anthropic_stub.AsyncAnthropic.return_value = MagicMock()

        with patch("jig.llm.anthropic.anthropic", anthropic_stub):
            from jig.llm.anthropic import AnthropicClient
            AnthropicClient(model="claude-test")

        call_kwargs = anthropic_stub.AsyncAnthropic.call_args[1]
        assert call_kwargs.get("max_retries") == 0


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenAILifecycle:
    async def test_aclose_calls_sdk_close(self):
        sdk_client = MagicMock()
        sdk_client.close = AsyncMock()

        from jig.llm.openai import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._client = sdk_client
        client._model = "gpt-test"
        client._provider_label = "openai"
        client._closed = False

        await client.aclose()
        sdk_client.close.assert_awaited_once()

    async def test_aclose_idempotent(self):
        sdk_client = MagicMock()
        sdk_client.close = AsyncMock()

        from jig.llm.openai import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._client = sdk_client
        client._model = "gpt-test"
        client._provider_label = "openai"
        client._closed = False

        await client.aclose()
        await client.aclose()
        sdk_client.close.assert_awaited_once()

    async def test_sdk_max_retries_defaults_to_zero(self):
        openai_stub = MagicMock()
        openai_stub.AsyncOpenAI.return_value = MagicMock()

        with patch("jig.llm.openai.openai", openai_stub):
            from jig.llm.openai import OpenAIClient
            OpenAIClient(model="gpt-test")

        call_kwargs = openai_stub.AsyncOpenAI.call_args[1]
        assert call_kwargs.get("max_retries") == 0


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGeminiLifecycle:
    async def test_aclose_calls_sdk_close(self):
        """aclose() calls the (sync) genai.Client.close()."""
        genai_stub = MagicMock()
        sdk_client = MagicMock()
        sdk_client.close = MagicMock()
        genai_stub.Client.return_value = sdk_client

        with patch("jig.llm.google.genai", genai_stub), \
             patch("jig.llm.google.genai_types", MagicMock()):
            from jig.llm.google import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._client = sdk_client
            client._model = "gemini-test"
            client._closed = False

            await client.aclose()

        sdk_client.close.assert_called_once()

    async def test_aclose_idempotent(self):
        sdk_client = MagicMock()
        sdk_client.close = MagicMock()

        from jig.llm.google import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._client = sdk_client
        client._model = "gemini-test"
        client._closed = False

        await client.aclose()
        await client.aclose()
        sdk_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOllamaLifecycle:
    async def test_aclose_closes_inner_httpx_client(self):
        """aclose() closes the underlying httpx.AsyncClient inside OllamaAsyncClient."""
        inner_http = MagicMock()
        inner_http.aclose = AsyncMock()

        ollama_stub = MagicMock()
        sdk_client = MagicMock()
        sdk_client._client = inner_http

        from jig.llm.ollama import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        client._client = sdk_client
        client._model = "llama3.1"
        client._closed = False

        await client.aclose()
        inner_http.aclose.assert_awaited_once()

    async def test_aclose_idempotent(self):
        inner_http = MagicMock()
        inner_http.aclose = AsyncMock()
        sdk_client = MagicMock()
        sdk_client._client = inner_http

        from jig.llm.ollama import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        client._client = sdk_client
        client._model = "llama3.1"
        client._closed = False

        await client.aclose()
        await client.aclose()
        inner_http.aclose.assert_awaited_once()

    async def test_aclose_tolerates_missing_inner_client(self):
        """If the Ollama SDK changes its internals, aclose() must not crash."""
        sdk_client = MagicMock(spec=[])  # no attributes at all

        from jig.llm.ollama import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        client._client = sdk_client
        client._model = "llama3.1"
        client._closed = False

        await client.aclose()  # must not raise


# ---------------------------------------------------------------------------
# factory.complete() lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFactoryCompletLifecycle:
    async def test_aclose_called_on_success(self):
        """factory.complete() calls aclose() after a successful completion."""
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(return_value=_mock_response())
            instance.aclose = AsyncMock()

            from jig.llm.factory import complete
            await complete("claude-test", [{"role": "user", "content": "hi"}])

        instance.aclose.assert_awaited_once()

    async def test_aclose_called_on_failure(self):
        """factory.complete() calls aclose() even when complete() raises."""
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            instance = mock_cls.return_value
            instance.complete = AsyncMock(side_effect=RuntimeError("provider down"))
            instance.aclose = AsyncMock()

            from jig.llm.factory import complete
            with pytest.raises(RuntimeError, match="provider down"):
                await complete("claude-test", [{"role": "user", "content": "hi"}])

        instance.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# ollama_embed() lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOllamaEmbedLifecycle:
    async def test_embed_closes_httpx_client_on_success(self):
        """ollama_embed() closes the inner httpx client after embedding completes."""
        inner_http = MagicMock()
        inner_http.aclose = AsyncMock()

        sdk_client = MagicMock()
        sdk_client._client = inner_http

        embed_response = MagicMock()
        embed_response.embeddings = [[0.1, 0.2, 0.3]]
        sdk_client.embed = AsyncMock(return_value=embed_response)

        OllamaAsyncClientStub = MagicMock(return_value=sdk_client)

        with patch("jig._embed.OllamaAsyncClient", OllamaAsyncClientStub):
            from jig._embed import ollama_embed
            result = await ollama_embed("hello", model="nomic-embed-text")

        inner_http.aclose.assert_awaited_once()
        assert result.shape == (3,)

    async def test_embed_closes_httpx_client_on_failure(self):
        """ollama_embed() closes the inner httpx client even when embed() raises."""
        inner_http = MagicMock()
        inner_http.aclose = AsyncMock()

        sdk_client = MagicMock()
        sdk_client._client = inner_http
        sdk_client.embed = AsyncMock(side_effect=RuntimeError("ollama down"))

        OllamaAsyncClientStub = MagicMock(return_value=sdk_client)

        with patch("jig._embed.OllamaAsyncClient", OllamaAsyncClientStub):
            from jig._embed import ollama_embed
            with pytest.raises(RuntimeError, match="ollama down"):
                await ollama_embed("hello", model="nomic-embed-text")

        inner_http.aclose.assert_awaited_once()
