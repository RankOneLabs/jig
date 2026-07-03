"""Shared score validation used by persistence paths, judges, and composites."""
from __future__ import annotations

import math

from jig.core.types import Score


def validate_scores(scores: list[Score]) -> None:
    """Raise ValueError if scores cannot be safely persisted.

    Rules enforced:
    - scores must be non-empty
    - every dimension must be a non-empty string
    - every value must be a finite float in the inclusive range [0.0, 1.0]
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    for s in scores:
        if not isinstance(s.dimension, str) or not s.dimension:
            raise ValueError(
                f"score dimension must be a non-empty string, got {s.dimension!r}"
            )
        try:
            v = float(s.value)
        except (TypeError, ValueError):
            raise ValueError(
                f"score value must be numeric, got {s.value!r} for dimension {s.dimension!r}"
            )
        if math.isnan(v):
            raise ValueError(
                f"score value must not be NaN for dimension {s.dimension!r}"
            )
        if math.isinf(v):
            raise ValueError(
                f"score value must be finite, got {v!r} for dimension {s.dimension!r}"
            )
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"score value must be in [0.0, 1.0], got {v!r} for dimension {s.dimension!r}"
            )
