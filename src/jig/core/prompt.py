from __future__ import annotations

from jig.core.types import HumanExample, HumanExampleSet, MemoryEntry, ScoredResult


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


_POSITIVE_HEADING = "## Human-reviewed positive examples: imitate the cited strengths"
_NEGATIVE_HEADING = "## Human-reviewed negative examples: avoid the cited failures"

# Per-round allocation in the round-robin character budget below. Bounds how
# much of the shared budget any single example can claim before every other
# selected example gets a turn — the first example in render order can never
# consume the whole budget in one step.
_ROUND_CHUNK_CHARS = 400

_TRUNCATION_MARKER = "\n[... truncated: character budget exhausted ...]"


def _render_example_body(example: HumanExample) -> str:
    dims = "; ".join(
        f"{d.dimension}={d.value:.2f}" + (f" (note: {d.note})" if d.note else "")
        for d in example.dimensions
    )
    return (
        "<UNTRUSTED_EXAMPLE_INPUT>\n"
        f"{example.input_text}\n"
        "</UNTRUSTED_EXAMPLE_INPUT>\n"
        "<UNTRUSTED_EXAMPLE_OUTPUT>\n"
        f"{example.output}\n"
        "</UNTRUSTED_EXAMPLE_OUTPUT>\n"
        f"Human dimensions: {dims}\n"
    )


def _round_robin_allocate(bodies: list[str], budget: int) -> list[str]:
    """Split ``budget`` characters across ``bodies`` in fixed-size rounds.

    Each active body gets up to ``_ROUND_CHUNK_CHARS`` per round, cycling
    until every body is fully written or the budget runs out — so the
    first body can never claim the entire budget in one pass. A body left
    short of its full length gets an explicit truncation marker appended
    (not counted against the budget) so a cut example is never silently
    presented as complete.
    """
    written = [0] * len(bodies)
    remaining_budget = max(budget, 0)
    active = {i for i, b in enumerate(bodies) if b}
    while remaining_budget > 0 and active:
        for i in list(active):
            if remaining_budget <= 0:
                break
            take = min(_ROUND_CHUNK_CHARS, len(bodies[i]) - written[i], remaining_budget)
            if take <= 0:
                active.discard(i)
                continue
            written[i] += take
            remaining_budget -= take
            if written[i] >= len(bodies[i]):
                active.discard(i)

    result: list[str] = []
    for body, count in zip(bodies, written):
        if count >= len(body):
            result.append(body)
        else:
            result.append(body[:count] + _TRUNCATION_MARKER)
    return result


def build_human_feedback_section(example_set: HumanExampleSet, character_budget: int) -> str:
    """Render qualified human-reviewed exemplars for prompt injection.

    Returns an empty string — no headings, no placeholders — when neither
    section has any examples, so absence of evidence never becomes noisy
    prompt text. Otherwise renders the positive section (if any) followed
    by the negative section (if any), each example wrapped in untrusted-data
    delimiters so historical content can't be read as a new instruction.
    The combined character budget is allocated round-robin across every
    selected example (both sections together) — see
    :func:`_round_robin_allocate`.
    """
    if not example_set.positive and not example_set.negative:
        return ""

    bodies = [_render_example_body(e) for e in (*example_set.positive, *example_set.negative)]
    allocated = _round_robin_allocate(bodies, character_budget)
    split = len(example_set.positive)
    positive_bodies, negative_bodies = allocated[:split], allocated[split:]

    parts: list[str] = []
    if positive_bodies:
        parts.append(f"\n\n{_POSITIVE_HEADING}\n")
        parts.extend(positive_bodies)
    if negative_bodies:
        parts.append(f"\n\n{_NEGATIVE_HEADING}\n")
        parts.extend(negative_bodies)
    return "".join(parts)
