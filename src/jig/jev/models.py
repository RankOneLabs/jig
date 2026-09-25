"""Named question, answer, and result values for Jev."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

JevJson: TypeAlias = "str | int | float | bool | None | list[JevJson] | dict[str, JevJson]"


def _check_keys(criteria: dict[str, JevJson]) -> None:
    if not isinstance(criteria, dict):
        raise ValueError(f"criteria must be a mapping, got {criteria!r}")
    for key in criteria:
        if not isinstance(key, str):
            raise ValueError(f"criteria key {key!r} must be str")


@dataclass(frozen=True)
class NoulQuestion:
    id: str
    instructions: JevJson
    criteria: dict[str, JevJson] | None = None

    def __post_init__(self) -> None:
        if self.criteria is not None:
            _check_keys(self.criteria)
            for key in self.criteria:
                if key not in ("true", "false"):
                    raise ValueError(f"Noul criteria key {key!r} must be 'true' or 'false'")


@dataclass(frozen=True)
class ChoiceQuestion:
    id: str
    instructions: JevJson
    criteria: dict[str, JevJson]

    def __post_init__(self) -> None:
        _check_keys(self.criteria)
        if not self.criteria:
            raise ValueError("Choice criteria must contain at least one option")


@dataclass(frozen=True)
class ScoreQuestion:
    id: str
    instructions: JevJson
    criteria: list[JevJson]

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
            raise ValueError("Score criteria must contain 2 to 10 ordered levels")


JevQuestion: TypeAlias = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class NoulAnswer:
    question_id: str
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    question_id: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    question_id: str
    score: float
    probabilities: dict[str, float]
    legend: dict[str, Any] | None
    confidence: float


JevAnswer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class JevUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: dict[str, JevAnswer]
    usage: JevUsage
    latency_ms: float
    call_id: str
    provider_request_id: str | None
    attempts: int
