from __future__ import annotations

import json
from typing import Any

from jig.core.errors import GradeParseError
from jig.core.types import (
    CompletionParams,
    Grader,
    LLMClient,
    Message,
    Role,
    Score,
    ScoreSource,
)
from jig.feedback.parsing import strip_markdown_fence
from jig.feedback.validation import validate_scores

_SYSTEM_PROMPT = """You are an evaluation judge. Grade the assistant's output on the specified dimensions.

Return ONLY valid JSON in this exact format:
{{"scores": [{{"dimension": "<name>", "value": <0.0-1.0>, "reasoning": "<brief explanation>"}}]}}

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
            data = json.loads(strip_markdown_fence(response.content))
            scores = [
                Score(
                    dimension=s["dimension"],
                    value=float(s["value"]),
                    source=ScoreSource.LLM_JUDGE,
                )
                for s in data["scores"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise GradeParseError(
                f"LLMJudge could not parse judge response: {exc}"
            ) from exc

        try:
            returned_dims = {s.dimension for s in scores}
        except TypeError as exc:
            raise GradeParseError(
                f"LLMJudge response contains unhashable dimension value: {exc}"
            ) from exc
        missing = [d for d in self._dimensions if d not in returned_dims]
        if missing:
            raise GradeParseError(
                f"LLMJudge response missing required dimensions: {missing}"
            )

        try:
            validate_scores(scores)
        except ValueError as exc:
            raise GradeParseError(
                f"LLMJudge returned invalid score values: {exc}"
            ) from exc

        return scores
