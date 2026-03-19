from __future__ import annotations

import asyncio
from typing import Any

from jig.core.types import Grader, Score


class CompositeGrader(Grader):
    def __init__(self, graders: list[Grader]):
        self._graders = graders

    async def grade(
        self,
        input: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        results = await asyncio.gather(
            *[g.grade(input, output, context) for g in self._graders]
        )
        return [score for scores in results for score in scores]
