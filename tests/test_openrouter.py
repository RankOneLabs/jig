"""Tests for the OpenRouter adapter."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from jig.core.errors import JigLLMError
from jig.core.types import CompletionParams, Message, Role, ToolCall
from jig.llm.openrouter import OpenRouterClient, _OPENROUTER_BASE_URL


def _fake_response(cost: float | None = None) -> SimpleNamespace:
    """Build a minimal response shaped like openai's chat.completions object."""
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
    )
    if cost is not None:
        usage.cost = cost
    message = SimpleNamespace(content="hi", tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model="anthropic/claude-3.5-sonnet",
    )


class TestOpenRouterInit:
    def test_uses_env_var_when_api_key_omitted(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
        with patch("jig.llm.openai.openai") as mock_openai:
            OpenRouterClient(model="openai/gpt-4o-mini")
            mock_openai.AsyncOpenAI.assert_called_once()
            kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
            assert kwargs["api_key"] == "sk-from-env"
            assert kwargs["base_url"] == _OPENROUTER_BASE_URL

    def test_explicit_api_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
        with patch("jig.llm.openai.openai") as mock_openai:
            OpenRouterClient(model="openai/gpt-4o-mini", api_key="sk-explicit")
            kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
            assert kwargs["api_key"] == "sk-explicit"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("jig.llm.openai.openai"):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterClient(model="openai/gpt-4o-mini")

    def test_attribution_headers_attached(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            OpenRouterClient(
                model="openai/gpt-4o-mini",
                http_referer="https://my.app",
                x_title="my-app",
            )
            kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
            assert kwargs["default_headers"] == {
                "HTTP-Referer": "https://my.app",
                "X-Title": "my-app",
            }

    def test_no_headers_when_attribution_omitted(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            OpenRouterClient(model="openai/gpt-4o-mini")
            kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
            assert kwargs["default_headers"] is None

    def test_custom_base_url_override(self, monkeypatch):
        # Useful for self-hosted OpenRouter-compatible proxies.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            OpenRouterClient(
                model="openai/gpt-4o-mini",
                base_url="https://gateway.internal/v1",
            )
            kwargs = mock_openai.AsyncOpenAI.call_args.kwargs
            assert kwargs["base_url"] == "https://gateway.internal/v1"


class TestOpenRouterComplete:
    def test_none_extra_body_is_normalized_before_usage_injection(self):
        client = OpenRouterClient.__new__(OpenRouterClient)
        kwargs = {"extra_body": None}

        client._apply_extra_kwargs(kwargs)

        assert kwargs["extra_body"] == {"usage": {"include": True}}

    async def test_complete_requests_inline_cost(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_response(cost=0.00042)
            )
            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            response = await client.complete(params)

            create_kwargs = instance.chat.completions.create.call_args.kwargs
            assert create_kwargs["extra_body"] == {
                "usage": {"include": True}
            }
            assert response.usage.cost == pytest.approx(0.00042)

    async def test_empty_choices_raises_jig_llm_error(self, monkeypatch):
        # OpenRouter has been observed returning 200 OK with ``choices=None``
        # when the upstream provider errored after request acceptance.
        # Previously this crashed with ``TypeError: 'NoneType' object is not
        # subscriptable`` in ``response.choices[0]``, killing long sweeps.
        # The adapter should now raise a retryable ``JigLLMError`` so the
        # agent loop can recover.
        from jig.core.errors import JigLLMError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            response = SimpleNamespace(
                choices=None,
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                model="anthropic/claude-3.5-sonnet",
                error={"message": "provider downstream timeout"},
            )
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=response)
            mock_openai.RateLimitError = type(
                "RateLimitError", (Exception,), {}
            )

            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            with pytest.raises(JigLLMError) as exc:
                await client.complete(params)
            assert exc.value.provider == "openrouter"
            assert exc.value.retryable is True
            assert "provider downstream timeout" in str(exc.value)

    async def test_empty_choices_list_raises_jig_llm_error(self, monkeypatch):
        # Same as above but with ``choices=[]`` (IndexError shape rather than
        # TypeError shape). Both should funnel into the same JigLLMError.
        from jig.core.errors import JigLLMError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            response = SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                model="anthropic/claude-3.5-sonnet",
            )
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=response)
            mock_openai.RateLimitError = type(
                "RateLimitError", (Exception,), {}
            )

            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            with pytest.raises(JigLLMError) as exc:
                await client.complete(params)
            assert exc.value.provider == "openrouter"
            assert exc.value.retryable is True

    async def test_empty_choices_error_in_model_extra(self, monkeypatch):
        # The OpenAI SDK's pydantic ChatCompletion model doesn't declare an
        # ``error`` field, so OpenRouter's payload arrives in
        # ``response.model_extra`` rather than as a top-level attribute
        # (same shape as ``usage.cost``). We must look there too, or real
        # responses will pass the guard but drop the diagnostic message.
        from jig.core.errors import JigLLMError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            response = SimpleNamespace(
                choices=None,
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                model="anthropic/claude-3.5-sonnet",
                model_extra={"error": {"message": "rate limited upstream"}},
            )
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=response)
            mock_openai.RateLimitError = type(
                "RateLimitError", (Exception,), {}
            )

            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            with pytest.raises(JigLLMError) as exc:
                await client.complete(params)
            assert exc.value.provider == "openrouter"
            assert exc.value.retryable is True
            assert "rate limited upstream" in str(exc.value)

    async def test_provider_label_on_error(self, monkeypatch):
        from jig.core.errors import JigLLMError

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("boom")
            )
            mock_openai.RateLimitError = type(
                "RateLimitError", (Exception,), {}
            )
            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            with pytest.raises(JigLLMError) as exc:
                await client.complete(params)
            assert exc.value.provider == "openrouter"

    async def test_caller_extra_body_survives_merge(self, monkeypatch):
        # provider_params with its own extra_body must not be clobbered by
        # the adapter's usage.include injection. Both should end up in the
        # outgoing request — caller's fallback model list and our cost
        # tracking flag coexist.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_response(cost=0.001)
            )
            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")],
                provider_params={
                    "extra_body": {
                        "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
                        "provider": {"order": ["Anthropic", "OpenAI"]},
                    }
                },
            )
            await client.complete(params)

            create_kwargs = instance.chat.completions.create.call_args.kwargs
            assert create_kwargs["extra_body"] == {
                "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
                "provider": {"order": ["Anthropic", "OpenAI"]},
                "usage": {"include": True},
            }

    async def test_caller_can_disable_inline_cost(self, monkeypatch):
        # Explicit usage.include=False from the caller wins over our default.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_response(cost=None)
            )
            client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")],
                provider_params={"extra_body": {"usage": {"include": False}}},
            )
            await client.complete(params)
            create_kwargs = instance.chat.completions.create.call_args.kwargs
            assert create_kwargs["extra_body"]["usage"] == {"include": False}

    async def test_inline_cost_from_model_extra(self, monkeypatch):
        # Older openai SDK schemas don't yet declare ``cost`` on
        # CompletionUsage, so OpenRouter's value lands in
        # ``usage.model_extra``. Make sure that fallback path is exercised.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                model_extra={"cost": 0.00099},
            )
            # No ``.cost`` attribute — forces the model_extra branch.
            message = SimpleNamespace(content="hi", tool_calls=None)
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=usage,
                model="some/model",
            )
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=response)

            client = OpenRouterClient(model="some/model")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            result = await client.complete(params)
            assert result.usage.cost == pytest.approx(0.00099)

    async def test_inline_cost_falls_back_to_pricing_table(self, monkeypatch):
        # When OpenRouter doesn't return inline cost (older accounts, or
        # usage.include unsupported on a route), the pricing table still
        # gets a chance — and leaves cost None for unknown slugs.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_response(cost=None)
            )
            client = OpenRouterClient(model="some/unknown-slug")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            response = await client.complete(params)
            assert response.usage.cost is None

    async def test_usage_none_normalizes_to_zero_tokens(self, monkeypatch):
        # Some upstream providers return usage=None on certain error paths.
        # The adapter must not AttributeError — normalize to 0 tokens.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            message = SimpleNamespace(content="hi", tool_calls=None)
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=None,
                model="some/model",
            )
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=response)
            mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})

            client = OpenRouterClient(model="some/model")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            result = await client.complete(params)
            assert result.usage.input_tokens == 0
            assert result.usage.output_tokens == 0

    async def test_request_preparation_error_is_wrapped(self, monkeypatch):
        # json.dumps on assistant tool-call arguments can fail before the SDK
        # call. That is still an adapter boundary and must be classified.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock()

            client = OpenRouterClient(model="some/model")
            params = CompletionParams(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="tc",
                                name="bad_args",
                                arguments={"not_json": object()},
                            )
                        ],
                    )
                ]
            )

            with pytest.raises(JigLLMError) as exc:
                await client.complete(params)

            assert exc.value.provider == "openrouter"
            assert "Request preparation failed" in str(exc.value)
            instance.chat.completions.create.assert_not_called()

    async def test_cost_stamping_error_is_wrapped(self, monkeypatch):
        # Cost stamping is post-SDK adapter bookkeeping. It should not escape as
        # a raw RuntimeError if pricing code regresses or receives bad data.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with patch("jig.llm.openai.openai") as mock_openai:
            instance = mock_openai.AsyncOpenAI.return_value
            instance.chat.completions.create = AsyncMock(
                return_value=_fake_response(cost=None)
            )
            mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})

            client = OpenRouterClient(model="some/model")
            params = CompletionParams(
                messages=[Message(role=Role.USER, content="hi")]
            )
            with patch("jig.llm.openai.stamp_cost", side_effect=RuntimeError("pricing exploded")):
                with pytest.raises(JigLLMError) as exc:
                    await client.complete(params)

            assert exc.value.provider == "openrouter"
            assert "Response parsing failed" in str(exc.value)
            assert "pricing exploded" in str(exc.value)
