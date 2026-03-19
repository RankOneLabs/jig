from __future__ import annotations

from jig.core.types import MemoryEntry, ScoredResult


def build_system_message(
    system_prompt: str,
    memory: list[MemoryEntry],
    signals: list[ScoredResult],
) -> str:
    prompt = system_prompt

    if memory:
        prompt += "\n\n## Relevant context from memory\n"
        for entry in memory:
            prompt += f"- {entry.content}\n"

    if signals:
        prompt += "\n\n## Quality signals from past similar queries\n"
        for signal in signals:
            score_str = ", ".join(f"{s.dimension}: {s.value:.2f}" for s in signal.scores)
            prompt += f"- [{score_str}] {signal.content[:200]}\n"

    return prompt
