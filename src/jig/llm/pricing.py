"""Per-model pricing table for cost calculation.

Rates are USD per 1M tokens (input, output). Approximate — update when
providers change pricing. Unknown models return None; cost stamping is
best-effort and never blocks a response.
"""
from __future__ import annotations

from jig.core.types import Usage

# (input_per_mtok, output_per_mtok) in USD
_PRICING: dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.0, 75.0),
    # --- OpenAI ---
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (2.0, 8.0),
    "o1-mini": (3.0, 12.0),
    "o1": (15.0, 60.0),
    # --- Google ---
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
}


def get_pricing(model: str) -> tuple[float, float] | None:
    """Return (input_per_mtok, output_per_mtok) for a model, or None if unknown.

    Tries exact match first; falls back to longest-prefix match so dated
    variants (e.g. ``claude-sonnet-4-5-20251022``) resolve to their family.
    """
    if model in _PRICING:
        return _PRICING[model]
    best: str | None = None
    for key in _PRICING:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return _PRICING[best] if best else None


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Compute USD cost for a completion, or None if the model isn't priced."""
    pricing = get_pricing(model)
    if pricing is None:
        return None
    input_rate, output_rate = pricing
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def stamp_cost(usage: Usage, model: str) -> Usage:
    """Fill in ``usage.cost`` from the pricing table. No-op if unknown."""
    if usage.cost is not None:
        return usage
    usage.cost = compute_cost(model, usage.input_tokens, usage.output_tokens)
    return usage
