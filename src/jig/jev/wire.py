"""Pure Jev request serialization and all-or-nothing response parsing."""

from __future__ import annotations

import math
from typing import Any, Sequence

from jig.jev.errors import JevError
from jig.jev.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevAnswer,
    JevJson,
    JevQuestion,
    JevResult,
    JevUsage,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

# All 711 recorded Choice/Score distributions use two-decimal probabilities;
# 10 sum to 0.99 after rounding. Two grid units allow that observed rounding.
PROBABILITY_SUM_TOLERANCE = 0.02


def build_request_body(
    state: JevJson, questions: Sequence[JevQuestion], model: str
) -> dict[str, Any]:
    """Build the provider body while retaining caller key order and JSON values."""
    wire_questions: dict[str, dict[str, Any]] = {}
    for question in questions:
        if question.id in wire_questions:
            raise ValueError(f"duplicate question id {question.id!r}")
        if isinstance(question, NoulQuestion):
            item: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.criteria is not None:
                item["criteria"] = question.criteria
        elif isinstance(question, ChoiceQuestion):
            item = {"type": "choice", "instructions": question.instructions,
                    "criteria": question.criteria}
        elif isinstance(question, ScoreQuestion):
            item = {"type": "score", "instructions": question.instructions,
                    "criteria": question.criteria}
        else:
            raise ValueError(f"unknown question type {type(question).__name__}")
        wire_questions[question.id] = item
    return {"state": state, "model": model, "questions": wire_questions}


def parse_response(
    payload: Any,
    call_id: str,
    latency_ms: float,
    attempts: int,
    questions: Sequence[JevQuestion],
) -> JevResult:
    """Validate the entire payload before constructing any result or answer."""
    violations: list[str] = []
    requested: dict[str, JevQuestion] = {}
    for question in questions:
        if question.id in requested:
            violations.append(f"duplicate requested question id {question.id!r}")
        requested[question.id] = question

    if not isinstance(payload, dict):
        payload = {}
        violations.append("response must be an object")
    model = payload.get("model")
    if not isinstance(model, str):
        violations.append("model must be a string")
    provider_request_id = payload.get("request_id")
    if provider_request_id is not None and not isinstance(provider_request_id, str):
        violations.append("request_id must be a string when present")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        violations.append("answers must be an object")
        raw_answers = {}
    for question_id in requested.keys() - raw_answers.keys():
        violations.append(f"missing answer for {question_id!r}")
    for question_id in raw_answers.keys() - requested.keys():
        violations.append(f"unexpected answer id {question_id!r}")

    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        violations.append("usage must be an object")
        raw_usage = {}
    for name in ("input_tokens", "output_tokens"):
        value = raw_usage.get(name)
        if type(value) is not int or value < 0:
            violations.append(f"usage.{name} must be a nonnegative integer")

    for question_id, raw_answer in raw_answers.items():
        question = requested.get(question_id)
        if question is None:
            continue
        if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
            raise ValueError(f"unknown question type {type(question).__name__}")
        prefix = f"answer {question_id!r}"
        if not isinstance(raw_answer, dict):
            violations.append(f"{prefix} must be an object")
            continue
        expected_type = (
            "noul" if isinstance(question, NoulQuestion) else
            "choice" if isinstance(question, ChoiceQuestion) else "score"
        )
        if raw_answer.get("type") != expected_type:
            violations.append(f"{prefix}.type must be {expected_type!r}")
        if isinstance(question, NoulQuestion):
            _check_unit(raw_answer.get("noul"), f"{prefix}.noul", violations)
        elif isinstance(question, ChoiceQuestion):
            choice = raw_answer.get("choice")
            if not isinstance(choice, str) or choice not in question.criteria:
                violations.append(f"{prefix}.choice must name a requested option")
            _check_distribution(raw_answer.get("probabilities"), set(question.criteria),
                                f"{prefix}.probabilities", violations)
            _check_unit(raw_answer.get("confidence"), f"{prefix}.confidence", violations)
        elif isinstance(question, ScoreQuestion):
            score = raw_answer.get("score")
            if not _finite_number(score) or not 0 <= score <= len(question.criteria) - 1:
                violations.append(f"{prefix}.score must be within level range")
            level_keys = {str(index) for index in range(len(question.criteria))}
            _check_distribution(raw_answer.get("probabilities"), level_keys,
                                f"{prefix}.probabilities", violations)
            legend = raw_answer.get("legend")
            if legend is not None:
                if not isinstance(legend, dict):
                    violations.append(f"{prefix}.legend must be an object or null")
                elif any(not isinstance(key, str) for key in legend):
                    violations.append(f"{prefix}.legend keys must be strings")
            _check_unit(raw_answer.get("confidence"), f"{prefix}.confidence", violations)

    if violations:
        raise JevError(
            kind="invalid_response", status_code=None, call_id=call_id,
            provider_request_id=provider_request_id if isinstance(provider_request_id, str) else None,
            detail="; ".join(violations), attempts=attempts, elapsed_ms=latency_ms,
        )

    # Construction starts only after the complete validation pass succeeds.
    answers: dict[str, JevAnswer] = {}
    for question_id, raw_answer in raw_answers.items():
        question = requested[question_id]
        if isinstance(question, NoulQuestion):
            answers[question_id] = NoulAnswer(question_id, raw_answer["noul"])
        elif isinstance(question, ChoiceQuestion):
            answers[question_id] = ChoiceAnswer(
                question_id, raw_answer["choice"], raw_answer["probabilities"],
                raw_answer["confidence"],
            )
        elif isinstance(question, ScoreQuestion):
            answers[question_id] = ScoreAnswer(
                question_id, raw_answer["score"], raw_answer["probabilities"],
                raw_answer.get("legend"), raw_answer["confidence"],
            )
        else:
            raise ValueError(f"unknown question type {type(question).__name__}")
    return JevResult(
        model=model, answers=answers,
        usage=JevUsage(raw_usage["input_tokens"], raw_usage["output_tokens"]),
        latency_ms=latency_ms, call_id=call_id,
        provider_request_id=provider_request_id, attempts=attempts,
    )


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_unit(value: Any, path: str, violations: list[str]) -> None:
    if not _finite_number(value) or not 0.0 <= value <= 1.0:
        violations.append(f"{path} must be finite and between 0 and 1")


def _check_distribution(
    value: Any, expected_keys: set[str], path: str, violations: list[str]
) -> None:
    if not isinstance(value, dict):
        violations.append(f"{path} must be an object")
        return
    if set(value) != expected_keys:
        violations.append(f"{path} keys must match requested options/levels")
    for key, probability in value.items():
        _check_unit(probability, f"{path}[{key!r}]", violations)
    if all(_finite_number(v) for v in value.values()):
        if abs(math.fsum(value.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            violations.append(f"{path} must sum to 1 within {PROBABILITY_SUM_TOLERANCE}")
