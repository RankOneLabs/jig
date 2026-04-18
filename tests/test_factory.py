"""Tests for jig.llm.from_model factory routing."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jig.llm import from_model


class TestFromModel:
    def test_claude_routes_to_anthropic(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            from_model("claude-sonnet-4-5")
            mock_cls.assert_called_once_with(model="claude-sonnet-4-5")

    def test_claude_dated_variant(self):
        with patch("jig.llm.anthropic.AnthropicClient") as mock_cls:
            from_model("claude-sonnet-4-5-20251022")
            mock_cls.assert_called_once_with(model="claude-sonnet-4-5-20251022")

    def test_gpt_routes_to_openai(self):
        with patch("jig.llm.openai.OpenAIClient") as mock_cls:
            from_model("gpt-5-mini")
            mock_cls.assert_called_once_with(model="gpt-5-mini")

    def test_o_series_routes_to_openai(self):
        with patch("jig.llm.openai.OpenAIClient") as mock_cls:
            from_model("o3-mini")
            mock_cls.assert_called_once_with(model="o3-mini")

    def test_o4_routes_to_openai(self):
        with patch("jig.llm.openai.OpenAIClient") as mock_cls:
            from_model("o4-mini")
            mock_cls.assert_called_once_with(model="o4-mini")

    def test_bare_o_series_routes_to_openai(self):
        with patch("jig.llm.openai.OpenAIClient") as mock_cls:
            from_model("o1")
            mock_cls.assert_called_once_with(model="o1")

    def test_o_series_does_not_match_unrelated_names(self):
        """'o11-mini' (hypothetical non-OpenAI model) should not route to OpenAI."""
        with pytest.raises(ValueError, match="No provider matches"):
            from_model("o11-mini")
        with pytest.raises(ValueError, match="No provider matches"):
            from_model("o2-whatever")

    def test_gemini_routes_to_google(self):
        with patch("jig.llm.google.GeminiClient") as mock_cls:
            from_model("gemini-2.5-pro")
            mock_cls.assert_called_once_with(model="gemini-2.5-pro")

    def test_ollama_prefix_stripped(self):
        with patch("jig.llm.ollama.OllamaClient") as mock_cls:
            from_model("ollama/llama3.1")
            mock_cls.assert_called_once_with(model="llama3.1")

    def test_dispatch_prefix_stripped(self):
        with patch("jig.llm.dispatch.DispatchClient") as mock_cls:
            from_model("dispatch/llama-70b")
            mock_cls.assert_called_once_with(model="llama-70b")

    def test_overrides_pass_through(self):
        with patch("jig.llm.ollama.OllamaClient") as mock_cls:
            from_model("ollama/llama3.1", host="http://mcbain:11434")
            mock_cls.assert_called_once_with(
                model="llama3.1", host="http://mcbain:11434"
            )

    def test_dispatch_overrides_pass_through(self):
        with patch("jig.llm.dispatch.DispatchClient") as mock_cls:
            from_model(
                "dispatch/llama-70b",
                dispatch_url="http://willie:8900",
                machine="mcbain",
            )
            mock_cls.assert_called_once_with(
                model="llama-70b",
                dispatch_url="http://willie:8900",
                machine="mcbain",
            )

    def test_chatgpt_routes_to_openai(self):
        with patch("jig.llm.openai.OpenAIClient") as mock_cls:
            from_model("chatgpt-4o-latest")
            mock_cls.assert_called_once_with(model="chatgpt-4o-latest")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="No provider matches"):
            from_model("llama3.1")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="No provider matches"):
            from_model("")

    def test_empty_ollama_suffix_raises(self):
        with pytest.raises(ValueError, match="Model name required after 'ollama/'"):
            from_model("ollama/")

    def test_empty_dispatch_suffix_raises(self):
        with pytest.raises(ValueError, match="Model name required after 'dispatch/'"):
            from_model("dispatch/")
