from __future__ import annotations

import json
import re
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
{{"scores": [{{"dimension": "<name>", "value": <0.0-1.0>, "reasoning": "<brief explanation>"}}]}}

Dimensions to grade: {dimensions}

{rubric}"""


_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n```\s*\Z",
    re.DOTALL,
)


def _strip_markdown_fence(text: str) -> str:
    """Strip a surrounding ``` fence (with optional ``json`` lang tag).

    Models — especially Claude — wrap structured output in a fenced
    code block even when the system prompt forbids it. That habit
    isn't reliably overridden by instructions, so we absorb the
    common "whole response is one fenced block" case here before
    json.loads runs.

    Conservative: only strips when the *entire* response is one
    fenced block. A response with leading prose and a fenced block
    in the middle is still treated as malformed (handled by the
    grade() fallback path) — that's a real prompt-tuning issue, not
    a formatting quirk to absorb silently.
    """
    match = _FENCE_RE.match(text)
    if match is None:
        return text
    return match.group("body")


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
            data = json.loads(_strip_markdown_fence(response.content.strip()))
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
