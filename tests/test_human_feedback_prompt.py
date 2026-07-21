"""Tests for build_human_feedback_section — prompt rendering of qualified
human-reviewed positive/negative exemplars."""
from __future__ import annotations

from jig.core.prompt import build_human_feedback_section
from jig.core.types import HumanExample, HumanExampleDimension, HumanExampleSet


def _example(result_id: str, classification: str, dim: str, value: float, note: str | None,
             input_text: str = "input text", output: str = "output text") -> HumanExample:
    return HumanExample(
        result_id=result_id,
        input_text=input_text,
        output=output,
        classification=classification,
        dimensions=[HumanExampleDimension(dimension=dim, value=value, note=note)],
    )


def test_no_examples_omits_both_sections_and_returns_empty_string():
    out = build_human_feedback_section(HumanExampleSet(positive=[], negative=[]), 6000)
    assert out == ""


def test_positive_only_omits_negative_heading():
    ex = _example("r1", "positive", "plausibility", 0.9, "great")
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 6000)
    assert "Human-reviewed positive examples" in out
    assert "Human-reviewed negative examples" not in out


def test_negative_only_omits_positive_heading():
    ex = _example("r1", "negative", "plausibility", 0.1, "bad")
    out = build_human_feedback_section(HumanExampleSet(positive=[], negative=[ex]), 6000)
    assert "Human-reviewed negative examples" in out
    assert "Human-reviewed positive examples" not in out


def test_both_sections_present_in_order():
    pos = _example("r1", "positive", "plausibility", 0.9, "great")
    neg = _example("r2", "negative", "plausibility", 0.1, "bad")
    out = build_human_feedback_section(HumanExampleSet(positive=[pos], negative=[neg]), 6000)
    pos_idx = out.index("Human-reviewed positive examples")
    neg_idx = out.index("Human-reviewed negative examples")
    assert pos_idx < neg_idx


def test_examples_wrapped_in_untrusted_delimiters():
    ex = _example("r1", "positive", "plausibility", 0.9, "great", input_text="the task", output="the result")
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 6000)
    assert "<UNTRUSTED_EXAMPLE_INPUT>" in out
    assert "the task" in out
    assert "</UNTRUSTED_EXAMPLE_INPUT>" in out
    assert "<UNTRUSTED_EXAMPLE_OUTPUT>" in out
    assert "the result" in out
    assert "</UNTRUSTED_EXAMPLE_OUTPUT>" in out


def test_dimension_value_and_note_rendered():
    ex = _example("r1", "positive", "plausibility", 0.9, "very believable")
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 6000)
    assert "plausibility=0.90" in out
    assert "very believable" in out


def test_note_none_renders_without_parenthetical():
    ex = _example("r1", "positive", "plausibility", 0.9, None)
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 6000)
    assert "plausibility=0.90" in out
    assert "note:" not in out


def test_small_budget_truncates_with_explicit_marker():
    ex = _example("r1", "positive", "plausibility", 0.9, "note", input_text="x" * 5000, output="y" * 5000)
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 100)
    assert "[... truncated" in out


def test_zero_budget_still_renders_marker_not_full_content():
    ex = _example("r1", "positive", "plausibility", 0.9, "note", input_text="x" * 5000, output="y" * 5000)
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 0)
    assert "[... truncated" in out
    assert "x" * 5000 not in out


def test_ample_budget_does_not_truncate():
    ex = _example("r1", "positive", "plausibility", 0.9, "note", input_text="short input", output="short output")
    out = build_human_feedback_section(HumanExampleSet(positive=[ex], negative=[]), 6000)
    assert "[... truncated" not in out
    assert "short input" in out
    assert "short output" in out


def test_first_example_cannot_consume_entire_budget():
    """Round-robin allocation: with a tight shared budget, a later example
    must still get some of its content rendered, not zero."""
    big = "z" * 10_000
    ex1 = _example("r1", "positive", "plausibility", 0.9, "n1", input_text=big, output=big)
    ex2 = _example("r2", "positive", "plausibility", 0.9, "n2", input_text="second example body", output="second output")
    out = build_human_feedback_section(HumanExampleSet(positive=[ex1, ex2], negative=[]), 1000)
    assert "second example body" in out


def test_budget_shared_across_positive_and_negative_sections():
    big = "z" * 10_000
    pos = _example("r1", "positive", "plausibility", 0.9, "n1", input_text=big, output=big)
    neg = _example("r2", "negative", "plausibility", 0.1, "n2", input_text="short neg input", output="short neg output")
    out = build_human_feedback_section(HumanExampleSet(positive=[pos], negative=[neg]), 1000)
    assert "short neg input" in out
