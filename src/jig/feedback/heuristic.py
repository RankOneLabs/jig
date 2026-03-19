from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from jig.core.types import Grader, Score, ScoreSource


@dataclass
class Check:
    name: str
    pattern: str | Callable[[str, str], float]


class HeuristicGrader(Grader):
    def __init__(self, checks: list[Check]):
        self._checks = checks

    async def grade(
        self,
        input: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        scores: list[Score] = []
        for check in self._checks:
            if callable(check.pattern):
                value = check.pattern(input, output)
            else:
                value = 1.0 if re.search(check.pattern, output) else 0.0
            scores.append(
                Score(
                    dimension=check.name,
                    value=max(0.0, min(1.0, value)),
                    source=ScoreSource.HEURISTIC,
                )
            )
        return scores
