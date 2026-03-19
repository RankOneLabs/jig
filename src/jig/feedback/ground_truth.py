from __future__ import annotations

from typing import Any, Callable

from jig.core.types import Grader, Score, ScoreSource


class GroundTruthGrader(Grader):
    def __init__(self, comparator: Callable[[str, str], float]):
        self._comparator = comparator

    async def grade(
        self,
        input: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        if not context or "expected" not in context:
            return [Score(dimension="correctness", value=0.0, source=ScoreSource.GROUND_TRUTH)]
        value = self._comparator(output, context["expected"])
        return [
            Score(
                dimension="correctness",
                value=max(0.0, min(1.0, value)),
                source=ScoreSource.GROUND_TRUTH,
            )
        ]
