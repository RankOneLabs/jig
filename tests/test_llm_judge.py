"""Tests for LLMJudge.

Tightly scoped — most LLMJudge behavior is exercised indirectly via
the higher-level judge variants, but the underlying class itself
needs direct coverage of the system-prompt format path. Without it,
a regression in the prompt template (e.g., un-escaped JSON braces
that ``str.format`` interprets as placeholders) goes uncaught until
a downstream consumer actually calls ``grade()``.
"""
from __future__ import annotations

import json

import pytest

from jig.core.errors import GradeParseError
from jig.core.types import CompletionParams, LLMClient, LLMResponse, Usage
from jig.feedback.llm_judge import LLMJudge


class _CannedLLM(LLMClient):
    """Returns a fixed JSON envelope and records the system prompt
    it received, so we can assert on prompt assembly."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_system: str | None = None

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.last_system = params.system
        return LLMResponse(
            content=self._response,
            tool_calls=None,
            usage=Usage(1, 1, cost=0.0),
            latency_ms=1.0,
            model="canned",
        )


async def test_grade_does_not_raise_on_default_dimensions() -> None:
    """Regression: the system-prompt template's literal JSON braces
    must be escaped (``{{ }}``) so ``str.format(...)`` doesn't
    interpret them as placeholders. An un-escaped template raises
    KeyError at format() time before the LLM is even called."""
    llm = _CannedLLM(
        json.dumps(
            {
                "scores": [
                    {"dimension": "relevance", "value": 0.8},
                    {"dimension": "completeness", "value": 0.6},
                    {"dimension": "accuracy", "value": 0.9},
                ]
            }
        )
    )
    judge = LLMJudge(llm)
    scores = await judge.grade(input="x", output="y")
    # System prompt assembled successfully (no KeyError on format).
    assert llm.last_system is not None
    # Scores parsed into the right dimensions.
    assert {s.dimension for s in scores} == {
        "relevance",
        "completeness",
        "accuracy",
    }


async def test_grade_with_custom_dimensions_and_rubric() -> None:
    """Custom dimensions and a custom rubric both flow into the
    system prompt and out the other side as scored Score objects."""
    llm = _CannedLLM(
        json.dumps(
            {
                "scores": [
                    {"dimension": "tone", "value": 0.7},
                    {"dimension": "clarity", "value": 0.5},
                ]
            }
        )
    )
    judge = LLMJudge(
        llm,
        dimensions=["tone", "clarity"],
        rubric="Score harshly on marketing speak.",
    )
    scores = await judge.grade(input="brief", output="some draft")
    assert llm.last_system is not None
    assert "tone" in llm.last_system
    assert "clarity" in llm.last_system
    assert "Score harshly on marketing speak." in llm.last_system
    assert [s.dimension for s in scores] == ["tone", "clarity"]
    assert scores[0].value == 0.7
    assert scores[1].value == 0.5


async def test_grade_raises_on_malformed_response() -> None:
    """Malformed judge response raises GradeParseError — not a 0.0
    fallback. Parse failure is an infrastructure failure, not a score."""
    llm = _CannedLLM("not json at all")
    judge = LLMJudge(llm, dimensions=["a", "b"])
    with pytest.raises(GradeParseError):
        await judge.grade(input="x", output="y")


async def test_grade_tolerates_fenced_json_response() -> None:
    """Models — Claude in particular — wrap structured output in a
    ```json ... ``` block even when the system prompt forbids it.
    Without absorbing the fence, every judge call falls into the
    malformed-response path and silently scores 0.0 across every
    dimension. The regression where this surfaced: every cell of an
    eval-graded study run reported a uniform 0.0 grade because the
    judge model wrapped its (correct) JSON in fences."""
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "scores": [
                    {"dimension": "relevance", "value": 0.9},
                    {"dimension": "completeness", "value": 0.8},
                    {"dimension": "accuracy", "value": 0.7},
                ]
            }
        )
        + "\n```"
    )
    llm = _CannedLLM(fenced)
    judge = LLMJudge(llm)
    scores = await judge.grade(input="x", output="y")
    by_dim = {s.dimension: s.value for s in scores}
    assert by_dim["relevance"] == 0.9
    assert by_dim["completeness"] == 0.8
    assert by_dim["accuracy"] == 0.7


async def test_grade_tolerates_fenced_json_no_lang_tag() -> None:
    """Bare ``` fence (no `json` lang tag) is also stripped — some
    models emit the fence without the language hint."""
    fenced = (
        "```\n"
        + json.dumps({"scores": [{"dimension": "a", "value": 0.5}]})
        + "\n```"
    )
    llm = _CannedLLM(fenced)
    judge = LLMJudge(llm, dimensions=["a"])
    scores = await judge.grade(input="x", output="y")
    assert scores[0].value == 0.5


async def test_grade_raises_on_non_numeric_value() -> None:
    """Non-numeric value raises GradeParseError — not a 0.0 fallback.
    A judge writing 'high' instead of 0.9 is a protocol failure."""
    llm = _CannedLLM(
        json.dumps(
            {
                "scores": [
                    {"dimension": "a", "value": "high"},
                    {"dimension": "b", "value": "medium"},
                ]
            }
        )
    )
    judge = LLMJudge(llm, dimensions=["a", "b"])
    with pytest.raises(GradeParseError):
        await judge.grade(input="x", output="y")


async def test_grade_raises_on_missing_required_dimension() -> None:
    """LLM returning fewer dimensions than requested raises GradeParseError.
    Missing data cannot be safely treated as an implicit zero."""
    llm = _CannedLLM(
        json.dumps({"scores": [{"dimension": "a", "value": 0.5}]})
    )
    judge = LLMJudge(llm, dimensions=["a", "b"])
    with pytest.raises(GradeParseError, match="missing required dimensions"):
        await judge.grade(input="x", output="y")


async def test_grade_raises_on_out_of_range_value() -> None:
    """Out-of-range numeric values raise GradeParseError — not clamped.
    A judge producing 1.7 or -0.2 indicates a judge/protocol failure."""
    for bad_val in (1.7, -0.2):
        llm = _CannedLLM(
            json.dumps({"scores": [{"dimension": "a", "value": bad_val}]})
        )
        judge = LLMJudge(llm, dimensions=["a"])
        with pytest.raises(GradeParseError):
            await judge.grade(input="x", output="y")


async def test_grade_raises_on_nan_value() -> None:
    """NaN in a judge response raises GradeParseError."""
    # JSON doesn't support NaN literals, but the model could embed
    # Python's float('nan') via a non-standard serializer or the score
    # field could be the string "NaN" that float() accepts.
    llm = _CannedLLM(
        json.dumps({"scores": [{"dimension": "a", "value": "NaN"}]})
    )
    judge = LLMJudge(llm, dimensions=["a"])
    with pytest.raises(GradeParseError):
        await judge.grade(input="x", output="y")


async def test_grade_raises_on_infinite_value() -> None:
    """Infinity in a judge response raises GradeParseError."""
    llm = _CannedLLM(
        json.dumps({"scores": [{"dimension": "a", "value": "Infinity"}]})
    )
    judge = LLMJudge(llm, dimensions=["a"])
    with pytest.raises(GradeParseError):
        await judge.grade(input="x", output="y")
