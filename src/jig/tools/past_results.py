"""``PastResults`` — a canned tool that lets an agent query its own history.

Wraps :meth:`FeedbackLoop.query` so agents don't have to reinvent the
"what have we tried before?" lookup. Ta's original ``PastResultsTool``
is the pattern this generalizes.
"""
from __future__ import annotations

from typing import Any

from jig.core.types import FeedbackLoop, FeedbackQuery, Tool, ToolDefinition


class PastResults(Tool):
    """Look up prior scored results similar to a hypothesis.

    When constructed with ``agent_name`` set, queries are automatically
    scoped to that agent's prior runs (so an explorer doesn't pull up
    the refiner's successes, which solve different problems).

    Callable by the LLM with ``{hypothesis, min_score?, k?}``; returns
    a text summary of up to ``k`` matching prior results with scores.
    """

    def __init__(
        self,
        feedback: FeedbackLoop,
        *,
        default_k: int = 5,
        agent_name: str | None = None,
    ) -> None:
        self._feedback = feedback
        self._default_k = default_k
        self._agent_name = agent_name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="past_results",
            description=(
                "Look up prior scored runs similar to a hypothesis. Use this "
                "before proposing a new idea so you can build on patterns "
                "that worked and avoid ones that didn't."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "The idea or approach to find prior attempts for.",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Optional: only return results with avg score >= this.",
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Max results to return (default {self._default_k}).",
                    },
                },
                "required": ["hypothesis"],
            },
        )

    async def execute(self, args: dict[str, Any]) -> str:
        hypothesis = args["hypothesis"]

        k_raw = args.get("k", self._default_k)
        try:
            k = int(k_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"k must be an integer, got {k_raw!r}") from e
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")

        min_score_raw = args.get("min_score")
        min_score: float | None
        if min_score_raw is None:
            min_score = None
        else:
            try:
                min_score = float(min_score_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"min_score must be a number, got {min_score_raw!r}"
                ) from e

        results = await self._feedback.query(FeedbackQuery(
            similar_to=hypothesis,
            agent_name=self._agent_name,
            min_score=min_score,
            limit=k,
        ))

        if not results:
            return "No prior results match this hypothesis."

        lines = [f"Found {len(results)} prior result(s):"]
        for r in results:
            excerpt = r.content[:180].replace("\n", " ")
            if len(r.content) > 180:
                excerpt += "…"
            lines.append(f"- [avg={r.avg_score:.2f}] {excerpt}")
        return "\n".join(lines)
