"""Shared response-parsing utilities for graders that consume JSON
from LLM responses.

LLM-backed graders (:class:`LLMJudge`, :class:`PairwiseLLMJudge`,
etc.) all share a "respond ONLY in JSON" contract that models
routinely violate by wrapping the structured output in a markdown
code fence. The behavior isn't reliably suppressible via prompting
— it's a default formatting habit, especially in Claude — so the
graders absorb the common case at parse time.

Lives in its own module so the strip-fence helper is shared across
graders rather than duplicated. New JSON-shaped graders should
route their `response.content` through :func:`strip_markdown_fence`
before `json.loads`.
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n```\s*\Z",
    re.DOTALL,
)


def strip_markdown_fence(text: str) -> str:
    """Strip a surrounding ``` fence (with optional ``json`` lang tag).

    Conservative: only strips when the *entire* input is one fenced
    block. A response with leading prose and a fenced block in the
    middle is treated as un-fenceable and returned as-is — that's a
    real prompt-tuning issue, not a formatting quirk to absorb
    silently. Trims surrounding whitespace before matching so a
    trailing newline after the closing fence doesn't defeat the
    regex.
    """
    match = _FENCE_RE.match(text.strip())
    if match is None:
        return text
    return match.group("body")
