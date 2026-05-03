"""Tests for jig.eval.dataset — jsonl + promptfoo-yaml I/O."""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

from jig.core.types import EvalCase
from jig.eval import load_jsonl, load_promptfoo_yaml, write_jsonl


# --- jsonl ---


def test_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "cases.jsonl"
    cases = [
        EvalCase(input="what is 2+2?", expected="4"),
        EvalCase(
            input="hello",
            expected="hi",
            context={"locale": "en"},
            metadata={"tags": ["smoke"]},
        ),
        EvalCase(input="bare-input-only"),
    ]
    write_jsonl(cases, path)
    loaded = load_jsonl(path)
    assert loaded == cases


def test_load_jsonl_skips_blank_and_comments(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        "# this is a comment about the dataset\n"
        '{"input": "first"}\n'
        "\n"
        "  # indented comment, also skipped\n"
        '{"input": "second"}\n'
    )
    cases = load_jsonl(path)
    assert [c.input for c in cases] == ["first", "second"]


def test_load_jsonl_rejects_missing_input(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"expected": "x"}\n')
    with pytest.raises(ValueError, match="missing required 'input' field"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_string_input(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": 42}\n')
    with pytest.raises(ValueError, match="'input' must be a string, got int"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_string_expected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "ok", "expected": ["list", "not", "string"]}\n')
    with pytest.raises(ValueError, match="'expected' must be a string or null"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_dict_context(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "ok", "context": [1, 2, 3]}\n')
    with pytest.raises(ValueError, match=r"'context' must be a dict or null"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_dict_metadata(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "ok", "metadata": "tag-string"}\n')
    with pytest.raises(ValueError, match=r"'metadata' must be a dict or null"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_object_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('"just a string"\n')
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_jsonl(path)


def test_load_jsonl_helpful_error_on_bad_json(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "ok"}\n{not valid json}\n')
    with pytest.raises(ValueError) as exc:
        load_jsonl(path)
    msg = str(exc.value)
    assert ":2:" in msg  # line number
    assert "invalid JSON" in msg


def test_write_jsonl_omits_unset_optional_fields(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl([EvalCase(input="bare")], path)
    payload = path.read_text().strip()
    obj = json.loads(payload)
    assert obj == {"input": "bare"}


# --- promptfoo yaml ---


def test_load_promptfoo_yaml_basic(tmp_path):
    path = tmp_path / "tests.yaml"
    path.write_text(
        """
tests:
  - description: factual question
    vars:
      input: what is 2+2?
      expected: "4"
  - vars:
      prompt: tell me a joke
      style: dry
"""
    )
    cases = load_promptfoo_yaml(path)
    assert len(cases) == 2
    assert cases[0].input == "what is 2+2?"
    assert cases[0].expected == "4"
    assert cases[0].metadata == {"description": "factual question"}
    assert cases[1].input == "tell me a joke"
    assert cases[1].expected is None
    # Non-input/expected vars become context
    assert cases[1].context == {"style": "dry"}


def test_load_promptfoo_yaml_imports_assertions_into_metadata(tmp_path):
    path = tmp_path / "tests.yaml"
    path.write_text(
        """
tests:
  - description: cite a source
    vars:
      input: who wrote The Iliad?
    assert:
      - type: contains
        value: Homer
"""
    )
    cases = load_promptfoo_yaml(path)
    assert cases[0].metadata is not None
    assert cases[0].metadata["assertions"] == [
        {"type": "contains", "value": "Homer"}
    ]
    assert cases[0].metadata["description"] == "cite a source"


def test_load_promptfoo_yaml_preserves_explicit_empty_input(tmp_path):
    """An explicit empty-string ``vars.input`` should be preserved,
    not silently fall through to ``vars.prompt``.
    """
    path = tmp_path / "tests.yaml"
    path.write_text(
        """
tests:
  - vars:
      input: ""
      prompt: fallback should not be used
"""
    )
    cases = load_promptfoo_yaml(path)
    assert cases[0].input == ""


def test_load_promptfoo_yaml_rejects_missing_input(tmp_path):
    path = tmp_path / "tests.yaml"
    path.write_text(
        """
tests:
  - vars:
      style: dry
"""
    )
    with pytest.raises(ValueError, match=r"missing vars\.input or vars\.prompt"):
        load_promptfoo_yaml(path)


def test_load_promptfoo_yaml_rejects_missing_tests_key(tmp_path):
    path = tmp_path / "tests.yaml"
    path.write_text("not_tests: []\n")
    with pytest.raises(ValueError, match="expected a top-level mapping"):
        load_promptfoo_yaml(path)


def test_load_promptfoo_yaml_rejects_tests_not_list(tmp_path):
    path = tmp_path / "tests.yaml"
    path.write_text("tests: oops\n")
    with pytest.raises(ValueError, match="'tests' must be a list"):
        load_promptfoo_yaml(path)


def test_load_promptfoo_yaml_helpful_error_without_pyyaml(tmp_path):
    """When pyyaml is missing, the import error becomes a helpful
    install-extra hint instead of a bare ModuleNotFoundError.
    """
    path = tmp_path / "tests.yaml"
    path.write_text("tests: []\n")

    # Force the dataset module's local `import yaml` to fail by
    # masking the module in sys.modules.
    with patch.dict(sys.modules, {"yaml": None}):
        with pytest.raises(ImportError, match="pip install 'jig\\[eval\\]'"):
            load_promptfoo_yaml(path)
