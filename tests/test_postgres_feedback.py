"""PostgresFeedbackLoop guarantees the shared suites do not pin down:
semantic (not chronological) retrieval, namespaces, the externally-applied
schema, and how values jsonb/text reject are handled."""
from __future__ import annotations

import math
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import psycopg
import pytest

from jig import FeedbackQuery
from jig.core.types import Score, ScoreSource
from jig.feedback.postgres import PostgresFeedbackLoop, feedback_schema_sql

type LoopFactory = Callable[..., PostgresFeedbackLoop]

_AXES = {"near": [1.0, 0.0, 0.0], "mid": [0.7, 0.7, 0.0], "far": [0.0, 0.0, 1.0]}


async def _axis_embed(text: str) -> np.ndarray:
    """``near`` / ``mid`` / ``far`` sit at fixed angles to the query ``near``."""
    return np.array(_AXES[text.split(":")[0]], dtype=np.float32)


@pytest.fixture
async def make_loop(pg_dsn: str, pg_schema: str) -> AsyncIterator[LoopFactory]:
    opened: list[PostgresFeedbackLoop] = []

    def factory(**kwargs: Any) -> PostgresFeedbackLoop:
        kwargs.setdefault("schema", pg_schema)
        loop = PostgresFeedbackLoop(pg_dsn, **kwargs)
        loop._embed = _axis_embed  # type: ignore[method-assign]
        opened.append(loop)
        return loop

    try:
        yield factory
    finally:
        for loop in opened:
            await loop.close()


async def _store_scored(loop: PostgresFeedbackLoop, content: str, input_text: str) -> str:
    rid = await loop.store_result(content, input_text, {"agent_name": "explorer"})
    await loop.score(rid, [Score("quality", 0.8, ScoreSource.HEURISTIC)])
    return rid


async def test_get_signals_ranks_by_similarity_not_recency(make_loop: LoopFactory) -> None:
    """A backend ordering by created_at passes every shape check; this is the
    assertion that catches it. The newest result is the least similar."""
    loop = make_loop()
    for name in ("mid", "near", "far"):
        await _store_scored(loop, name, f"{name}:input")

    signals = await loop.get_signals("near:query", limit=3)
    assert [s.content for s in signals] == ["near", "mid", "far"]


async def test_embeddings_are_stored_as_float32_bytes(
    make_loop: LoopFactory, pg_dsn: str, pg_schema: str,
) -> None:
    loop = make_loop()
    rid = await _store_scored(loop, "near", "near:input")
    with psycopg.connect(pg_dsn) as conn:
        [(blob,)] = conn.execute(
            f'SELECT embedding FROM "{pg_schema}".feedback_results WHERE id = %s', (rid,),
        ).fetchall()
    assert np.frombuffer(blob, dtype=np.float32).tolist() == _AXES["near"]


async def test_namespaces_isolate_queries_and_exports(make_loop: LoopFactory) -> None:
    trial_a = make_loop(namespace="trial-a")
    trial_b = make_loop(namespace="trial-b")
    await _store_scored(trial_a, "near", "near:input")

    assert (
        await trial_b.get_signals("near:query"),
        await trial_b.export_eval_set(),
    ) == ([], [])


async def test_metadata_filters_run_against_jsonb(make_loop: LoopFactory) -> None:
    """A tags value that is not an array never matches — a bare string must
    not match by substring or character."""
    loop = make_loop()
    for content, tags in (("x", ["fund", "mom"]), ("y", "fund"), ("z", ["mom"])):
        rid = await loop.store_result(content, f"near:{content}", {"agent_name": "explorer", "tags": tags})
        await loop.score(rid, [Score("quality", 0.5, ScoreSource.HEURISTIC)])

    results = await loop.query(FeedbackQuery(agent_name="explorer", tags=["fund"]))
    assert [r.content for r in results] == ["x"]


async def test_externally_applied_schema_needs_no_runtime_ddl(
    make_loop: LoopFactory, pg_dsn: str, pg_schema: str,
) -> None:
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(feedback_schema_sql(pg_schema))
    loop = make_loop(create_schema=False)
    await _store_scored(loop, "near", "near:input")
    assert [s.content for s in await loop.get_signals("near:query")] == ["near"]


async def test_nul_in_content_is_replaced_not_rejected(make_loop: LoopFactory) -> None:
    loop = make_loop()
    await _store_scored(loop, "out\x00put", "near:in\x00put")
    [signal] = await loop.get_signals("near:query")
    assert signal.content == "out�put"


async def test_non_finite_score_metadata_is_rejected(make_loop: LoopFactory) -> None:
    loop = make_loop()
    rid = await loop.store_result("x", "near:a", {})
    with pytest.raises(ValueError, match="not JSON-serializable"):
        await loop.score(rid, [Score("q", 0.5, ScoreSource.HUMAN, metadata={"v": math.nan})])


async def test_rejected_score_batch_inserts_nothing(make_loop: LoopFactory) -> None:
    loop = make_loop()
    rid = await loop.store_result("x", "near:a", {})
    with pytest.raises(ValueError):
        await loop.score(rid, [
            Score("ok", 0.5, ScoreSource.HUMAN),
            Score("bad", 0.5, ScoreSource.HUMAN, metadata={"v": object()}),
        ])
    assert await loop.get_signals("near:query") == []
