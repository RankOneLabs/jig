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
        k = int(args.get("k", self._default_k))
        min_score = args.get("min_score")

        results = await self._feedback.query(FeedbackQuery(
            similar_to=hypothesis,
            agent_name=self._agent_name,
            min_score=float(min_score) if min_score is not None else None,
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
