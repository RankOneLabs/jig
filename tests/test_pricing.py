"""Tests for the pricing table and cost stamping."""
from __future__ import annotations

import math

import pytest

from jig.core.types import Usage
from jig.llm.pricing import compute_cost, get_pricing, stamp_cost


class TestGetPricing:
    def test_exact_match(self):
        assert get_pricing("claude-sonnet-4-5") == (3.0, 15.0)

    def test_longest_prefix_match_for_dated_variant(self):
        # Dated model IDs should resolve to their family
        assert get_pricing("claude-sonnet-4-5-20251022") == (3.0, 15.0)
        assert get_pricing("claude-haiku-4-5-20251001") == (1.0, 5.0)

    def test_prefix_precedence(self):
        # More specific prefix wins over general
        assert get_pricing("gpt-5-mini") == (0.25, 2.0)
        assert get_pricing("gpt-5") == (1.25, 10.0)

    def test_unknown_model_returns_none(self):
        assert get_pricing("llama3.1") is None
        assert get_pricing("unknown-model-x") is None


class TestComputeCost:
    def test_basic_cost(self):
        # 1M input @ $3 + 1M output @ $15 = $18
        cost = compute_cost("claude-sonnet-4-5", 1_000_000, 1_000_000)
        assert cost == 18.0

    def test_small_token_counts(self):
        # 1k input + 500 output with claude-haiku-4-5 (1.0, 5.0)
        cost = compute_cost("claude-haiku-4-5", 1_000, 500)
        # 1000 * 1.0 / 1M + 500 * 5.0 / 1M = 0.001 + 0.0025 = 0.0035
        assert math.isclose(cost, 0.0035, abs_tol=1e-9)

    def test_unknown_model_returns_none(self):
        assert compute_cost("unknown-model", 1000, 1000) is None

    def test_negative_tokens_clamped_to_zero(self):
        """Negative tokens shouldn't produce negative cost (budget bypass)."""
        # Both negative → cost is zero, not negative
        assert compute_cost("claude-sonnet-4-5", -1000, -1000) == 0.0
        # Only input negative → input contributes 0, output normal
        # 1000 output @ $15/Mtok = 0.015
        cost = compute_cost("claude-sonnet-4-5", -500, 1000)
        assert cost == pytest.approx(0.015, abs=1e-9)


class TestStampCost:
    def test_stamps_cost_on_known_model(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        stamp_cost(usage, "claude-sonnet-4-5")
        assert usage.cost == 18.0

    def test_preserves_existing_cost(self):
        usage = Usage(input_tokens=1000, output_tokens=1000, cost=0.0)
        stamp_cost(usage, "claude-sonnet-4-5")
        # Ollama-style zero cost isn't overwritten
        assert usage.cost == 0.0

    def test_leaves_none_on_unknown_model(self):
        usage = Usage(input_tokens=1000, output_tokens=1000)
        stamp_cost(usage, "unknown-model")
        assert usage.cost is None

    def test_returns_same_usage_instance(self):
        usage = Usage(input_tokens=100, output_tokens=100)
        result = stamp_cost(usage, "claude-sonnet-4-5")
        assert result is usage
