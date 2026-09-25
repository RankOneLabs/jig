import copy
import json
import math

import pytest

from jig.jev import (
    ChoiceQuestion, JevError, NoulQuestion, ScoreQuestion,
    build_request_body, parse_response,
)


QUESTIONS = [
    NoulQuestion("n", {"ask": ["yes?", True]}, {"true": {"signal": 1}}),
    ChoiceQuestion("c", ["pick"], {"a.b": None, "two words": [1], "東京": "x"}),
    ScoreQuestion("s", "rate", [{"summary": "low"}, {"summary": "high"}]),
]


def response():
    return {
        "model": "jev-1", "request_id": "req_123", "future_field": {"ok": True},
        "answers": {
            "n": {"type": "noul", "noul": 0.8},
            "c": {"type": "choice", "choice": "東京", "probabilities": {
                "a.b": 0.1, "two words": 0.1, "東京": 0.8,
            }, "confidence": 0.7},
            "s": {"type": "score", "score": 0.99,
                  "probabilities": {"0": 0.0, "1": 0.99},
                  "legend": {"0": {"summary": "low"}, "1": {"summary": "high"}},
                  "confidence": 0.99},
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def parse(payload):
    return parse_response(payload, "call-1", 25.0, 2, QUESTIONS)


def test_request_preserves_json_structure_and_order():
    body = build_request_body({"items": [1]}, QUESTIONS, "jev-latest")
    assert list(body) == ["state", "model", "questions"]
    restored = json.loads(json.dumps(body, ensure_ascii=False))
    assert restored == body
    assert list(restored["questions"]["c"]["criteria"]) == ["a.b", "two words", "東京"]
    assert restored["questions"]["n"]["instructions"] == {"ask": ["yes?", True]}


def test_parse_realistic_response_and_optional_request_id():
    payload = response()
    result = parse(payload)
    assert result.provider_request_id == "req_123"
    assert result.answers["s"].legend is payload["answers"]["s"]["legend"]
    assert result.answers["s"].probabilities["1"] == 0.99
    assert result.attempts == 2 and result.latency_ms == 25.0
    del payload["request_id"]
    assert parse(payload).provider_request_id is None


def test_invalid_distributions():
    for total in (0.5, 1.5):
        payload = response()
        payload["answers"]["s"]["probabilities"] = {"0": 0.0, "1": total}
        with pytest.raises(JevError, match="sum to 1") as caught:
            parse(payload)
        assert caught.value.kind == "invalid_response"
    for invalid in (math.nan, math.inf, -0.1, 1.1):
        payload = response()
        payload["answers"]["c"]["probabilities"]["a.b"] = invalid
        with pytest.raises(JevError) as caught:
            parse(payload)
        assert caught.value.kind == "invalid_response"


def test_all_answers_validate_before_result():
    payload = response()
    payload["answers"]["extra_good"] = {"type": "noul", "noul": 0.4}
    payload["answers"]["c"]["choice"] = "unknown"
    with pytest.raises(JevError) as caught:
        parse(payload)
    assert "extra_good" in caught.value.detail and "'c'" in caught.value.detail

    four_questions = [*QUESTIONS, NoulQuestion("fourth", "yes?")]
    payload = response()
    payload["answers"]["fourth"] = {"type": "noul", "noul": 0.2}
    payload["answers"]["c"]["confidence"] = None
    with pytest.raises(JevError, match="confidence"):
        parse_response(payload, "call-1", 25.0, 2, four_questions)


def test_missing_extra_and_wrong_type():
    payload = response()
    del payload["answers"]["n"]
    with pytest.raises(JevError, match="'n'"):
        parse(payload)
    payload = response()
    payload["answers"]["n"] = copy.deepcopy(payload["answers"]["c"])
    with pytest.raises(JevError, match="'n'.type"):
        parse(payload)


def test_score_range_and_usage():
    payload = response()
    payload["answers"]["s"]["score"] = 2.0
    with pytest.raises(JevError, match="score"):
        parse(payload)
    for bad in (1.5, -1, None):
        payload = response()
        if bad is None:
            del payload["usage"]["input_tokens"]
        else:
            payload["usage"]["input_tokens"] = bad
        with pytest.raises(JevError, match="input_tokens"):
            parse(payload)
