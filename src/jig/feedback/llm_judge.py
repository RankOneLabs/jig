from __future__ import annotations

import json
from typing import Any

from jig.core.types import (
    CompletionParams,
    Grader,
    LLMClient,
    Message,
    Role,
    Score,
    ScoreSource,
)

_SYSTEM_PROMPT = """You are an evaluation judge. Grade the assistant's output on the specified dimensions.

Return ONLY valid JSON in this exact format:
{"scores": [{"dimension": "<name>", "value": <0.0-1.0>, "reasoning": "<brief explanation>"}]}

Dimensions to grade: {dimensions}

{rubric}"""


class LLMJudge(Grader):
    def __init__(
        self,
        llm: LLMClient,
        dimensions: list[str] | None = None,
        rubric: str = "",
    ):
        self._llm = llm
        self._dimensions = dimensions or ["relevance", "completeness", "accuracy"]
        self._rubric = rubric

    async def grade(
        self,
        input: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        system = _SYSTEM_PROMPT.format(
            dimensions=", ".join(self._dimensions),
            rubric=self._rubric,
        )
        user_msg = f"**Input:** {input}\n\n**Output:** {output}"
        if context:
            user_msg += f"\n\n**Context:** {json.dumps(context)}"

        params = CompletionParams(
            messages=[Message(role=Role.USER, content=user_msg)],
            system=system,
            temperature=0.0,
        )
        response = await self._llm.complete(params)

        try:
            data = json.loads(response.content)
            return [
                Score(
                    dimension=s["dimension"],
                    value=max(0.0, min(1.0, float(s["value"]))),
                    source=ScoreSource.LLM_JUDGE,
                )
                for s in data["scores"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [
                Score(dimension=d, value=0.0, source=ScoreSource.LLM_JUDGE)
                for d in self._dimensions
            ]
