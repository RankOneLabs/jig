from __future__ import annotations

import uuid
from datetime import datetime

from jig.core.types import EvalCase, FeedbackLoop, FeedbackQuery, Score, ScoreSource, ScoredResult


class NullFeedbackLoop(FeedbackLoop):
    """No-I/O FeedbackLoop implementation.

    Satisfies AgentConfig's required ``feedback`` field without creating a
    SQLite file or forcing every consumer that doesn't need persistence
    (hermetic tests, scripted eval sweeps) to maintain its own protocol
    shim. Every method is a no-op: writes are discarded, reads return
    empty.
    """

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        return str(uuid.uuid4())

    async def score(self, result_id: str, scores: list[Score]) -> None:
        return None

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        return []

    async def query(self, q: FeedbackQuery) -> list[ScoredResult]:
        return []

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        return []
