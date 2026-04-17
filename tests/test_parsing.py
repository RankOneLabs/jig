from __future__ import annotations

import pytest

from jig.core.errors import JigLLMError
from jig.llm._parsing import parse_tool_arguments


def test_dict_passthrough():
    raw = {"city": "Paris", "units": "metric"}
    assert parse_tool_arguments(raw, "anthropic") is raw


def test_json_string_parsed():
    assert parse_tool_arguments('{"city": "Paris"}', "openai") == {"city": "Paris"}


def test_empty_string_becomes_empty_dict():
    assert parse_tool_arguments("", "openai") == {}


def test_none_becomes_empty_dict():
    assert parse_tool_arguments(None, "openai") == {}


def test_malformed_json_raises_retryable():
    with pytest.raises(JigLLMError) as excinfo:
        parse_tool_arguments('{"city": "Paris"', "openai")
    err = excinfo.value
    assert err.provider == "openai"
    assert err.retryable is True
    assert "Malformed tool call arguments" in str(err)


def test_json_array_rejected():
    with pytest.raises(JigLLMError) as excinfo:
        parse_tool_arguments("[1, 2, 3]", "openai")
    assert excinfo.value.retryable is True
    assert "must be a JSON object" in str(excinfo.value)


def test_json_scalar_rejected():
    with pytest.raises(JigLLMError):
        parse_tool_arguments('"just a string"', "openai")


def test_unexpected_type_raises():
    with pytest.raises(JigLLMError) as excinfo:
        parse_tool_arguments(42, "ollama")
    assert "Unexpected tool call arguments type" in str(excinfo.value)
    assert excinfo.value.provider == "ollama"
    assert excinfo.value.retryable is True


def test_long_raw_is_truncated_in_error():
    huge = "x" * 5000  # invalid JSON, huge payload
    with pytest.raises(JigLLMError) as excinfo:
        parse_tool_arguments(huge, "openai")
    msg = str(excinfo.value)
    assert "len=5000" in msg
    assert len(msg) < 500  # full payload would blow this up


def test_provider_name_propagated():
    with pytest.raises(JigLLMError) as excinfo:
        parse_tool_arguments("{bad", "openai")
    assert excinfo.value.provider == "openai"
