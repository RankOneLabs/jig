import dataclasses

import pytest

from jig.jev import (
    ChoiceQuestion, JevError, JevUsage, NoulAnswer, NoulQuestion, ScoreQuestion,
)
from jig.jev.errors import JEV_ERROR_DETAIL_MAX_LENGTH


@pytest.mark.parametrize("instructions", ["plain", {"question": ["a", 1]}, ["a", {"b": True}]])
def test_noul_preserves_instructions(instructions):
    assert NoulQuestion("q", instructions).instructions is instructions


def test_criteria_contracts():
    with pytest.raises(ValueError, match="True"):
        NoulQuestion("q", "?", {True: "yes", False: "no"})
    with pytest.raises(ValueError, match="typo"):
        NoulQuestion("q", "?", {"typo": "yes"})
    with pytest.raises(ValueError, match="at least one"):
        ChoiceQuestion("q", "?", {})
    keys = ["a.b", "two words", "東京"]
    question = ChoiceQuestion("q", "?", dict.fromkeys(keys))
    assert list(question.criteria) == keys
    assert ChoiceQuestion("q", "?", {"only": None}).criteria == {"only": None}


def test_score_level_bounds_and_order():
    for count in (1, 11):
        with pytest.raises(ValueError):
            ScoreQuestion("q", "?", list(range(count)))
    for count in (2, 10):
        levels = [f"level {i}" for i in range(count)]
        assert ScoreQuestion("q", "?", levels).criteria is levels


def test_answer_and_usage_field_sets():
    assert not hasattr(NoulAnswer("q", 0.5), "confidence")
    with pytest.raises(TypeError):
        NoulAnswer("q", 0.5, confidence=0.5)
    assert [field.name for field in dataclasses.fields(JevUsage)] == [
        "input_tokens", "output_tokens"
    ]
    assert not hasattr(JevUsage(1, 2), "cost")


def test_error_detail_cap():
    error = JevError("transport", 503, "call-1", None, "x" * 10_000, 2, 123.0)
    assert len(error.detail) == JEV_ERROR_DETAIL_MAX_LENGTH
    assert len(str(error)) < 600
    assert "transport" in str(error) and "503" in str(error) and "call-1" in str(error)
