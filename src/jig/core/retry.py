from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: Callable[[Exception], bool] = lambda _: False,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args)
        except Exception as e:
            last_exc = e
            if not retryable(e) or attempt == max_attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc  # unreachable, but satisfies type checker
