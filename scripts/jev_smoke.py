"""Optional credentialed Jev smoke run. Prints metadata only; not packaged."""

import asyncio
import os

from jig.jev import ChoiceQuestion, JevClient, NoulQuestion, ScoreQuestion


async def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("Set TYPESAFE_API_KEY to run this smoke check")
    questions = [
        NoulQuestion("relevant", "Is the state about operating agents?"),
        ChoiceQuestion("kind", "Choose a category", {"practice": "Practice", "incident": "Incident"}),
        ScoreQuestion("quality", "Rate quality", ["low", "high"]),
    ]
    async with JevClient() as client:
        result = await client.evaluate({"topic": "agent operations"}, questions)
    print("model:", result.model)
    print("question_ids:", sorted(question.id for question in questions))
    print("usage:", result.usage.input_tokens, result.usage.output_tokens)
    print("answer_keys:", sorted(result.answers))


if __name__ == "__main__":
    asyncio.run(main())
