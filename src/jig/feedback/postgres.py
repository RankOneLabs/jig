"""PostgreSQL feedback backend — requires the ``postgres`` extra.

Import from this module directly (``from jig.feedback.postgres import
PostgresFeedbackLoop``); ``jig.feedback`` does not re-export it, so
SQLite-only consumers never import psycopg.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

from jig._embed import ollama_embed
from jig._postgres import LazyPgConnection, pg_text, quoted_schema
from jig.core.types import (
    EvalCase,
    FeedbackLoop,
    FeedbackQuery,
    HumanExampleSet,
    HumanFeedbackPromptConfig,
    Score,
    ScoredResult,
    ScoreSource,
)
from jig.feedback.loop import (
    _assemble_eval_cases,
    _assemble_query_results,
    _cosine_similarity,
    _EvalRow,
    _ExampleCandidate,
    _human_examples_window,
    _query_window,
    _QueryCandidate,
    _rank_human_examples,
    _ScoreRow,
    _validate_export_limit,
    _validate_export_range,
)
from jig.feedback.validation import validate_scores


def feedback_schema_sql(schema: str = "jig") -> str:
    """DDL for :class:`PostgresFeedbackLoop`'s tables, idempotent, in ``schema``.

    Exposed so a consumer with its own versioned migrator can apply it there
    and construct the loop with ``create_schema=False``. Mirrors the SQLite
    ``results`` / ``scores`` tables with ``namespace`` added, real
    timestamps, ``jsonb`` metadata, and an identity column standing in for
    SQLite's rowid as the score tie-breaker. Embeddings stay float32 bytes
    ranked in-process, exactly as SQLite stores them — no server extension.
    """
    s = quoted_schema(schema)
    return f"""
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.feedback_results (
    id text PRIMARY KEY,
    namespace text NOT NULL,
    content text NOT NULL,
    input text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding bytea,
    created_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS feedback_results_namespace_created_idx
    ON {s}.feedback_results (namespace, created_at DESC);

CREATE TABLE IF NOT EXISTS {s}.feedback_scores (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    result_id text NOT NULL REFERENCES {s}.feedback_results (id),
    dimension text NOT NULL,
    value double precision NOT NULL,
    source text NOT NULL,
    created_at timestamptz NOT NULL,
    metadata jsonb
);

CREATE INDEX IF NOT EXISTS feedback_scores_result_idx
    ON {s}.feedback_scores (result_id, id);
"""


def _metadata_text(metadata: Any, what: str) -> str:
    """Strict JSON for feedback metadata: reject rather than rewrite.

    Spans are observability and degrade to a best-effort encoding; feedback
    metadata is graded data, so a value jsonb cannot hold (non-JSON type,
    NaN/Infinity) rejects the call, matching SQLite's rejection of
    non-serializable metadata.
    """
    try:
        text = json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not JSON-serializable: {exc}") from exc
    return pg_text(text)


class PostgresFeedbackLoop(FeedbackLoop):
    """:class:`~jig.core.types.FeedbackLoop` over PostgreSQL.

    Behaviourally identical to :class:`~jig.feedback.loop.SQLiteFeedbackLoop`
    — both hand their rows to the same ranking, filtering and assembly
    transforms in :mod:`jig.feedback.loop`. Retrieval stays semantic:
    embeddings are stored as float32 ``bytea`` and ranked by cosine
    similarity in NumPy. ``namespace`` scopes every result write and read,
    standing in for the separate database file a SQLite consumer would open
    per trial.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "jig",
        namespace: str = "default",
        create_schema: bool = True,
        embed_model: str = "nomic-embed-text",
        ollama_host: str | None = None,
    ) -> None:
        s = quoted_schema(schema)
        self._results = f"{s}.feedback_results"
        self._scores = f"{s}.feedback_scores"
        self._namespace = namespace
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._conn = LazyPgConnection(dsn, feedback_schema_sql(schema) if create_schema else None)
        self._db_lock: asyncio.Lock | None = None

    def _get_db_lock(self) -> asyncio.Lock:
        if self._db_lock is None:
            self._db_lock = asyncio.Lock()
        return self._db_lock

    async def _embed(self, text: str) -> np.ndarray:
        return await ollama_embed(text, self._embed_model, self._ollama_host)

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta_text = _metadata_text(metadata or {}, "store_result metadata")
        result_id = str(uuid.uuid4())
        embedding = np.asarray(await self._embed(input_text), dtype=np.float32)
        async with self._get_db_lock():
            conn = await self._conn.get()
            await conn.execute(
                f"INSERT INTO {self._results} "
                "(id, namespace, content, input, metadata, embedding, created_at) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)",
                (
                    result_id,
                    self._namespace,
                    pg_text(content),
                    pg_text(input_text),
                    meta_text,
                    embedding.tobytes(),
                    datetime.now(UTC),
                ),
            )
        return result_id

    async def score(self, result_id: str, scores: list[Score]) -> None:
        validate_scores(scores)
        # Serialize before the transaction: bad metadata rejects the whole
        # call without inserting any score.
        serialized_metadata = [
            None if s.metadata is None
            else _metadata_text(s.metadata, f"Score.metadata for dimension {s.dimension!r}")
            for s in scores
        ]
        now = datetime.now(UTC)
        async with self._get_db_lock():
            conn = await self._conn.get()
            async with conn.transaction(), conn.cursor() as cur:
                await cur.executemany(
                    f"INSERT INTO {self._scores} "
                    "(result_id, dimension, value, source, created_at, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                    [
                        (result_id, s.dimension, s.value, s.source.value, now, meta)
                        for s, meta in zip(scores, serialized_metadata)
                    ],
                )

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        q = FeedbackQuery(similar_to=query, limit=limit, min_score=min_score, source=source)
        return await self.query(q)

    async def _score_rows(
        self, result_ids: list[str], *, with_order: bool,
    ) -> list[_ScoreRow]:
        """Score rows for ``result_ids`` in insertion order. Caller holds the lock."""
        if not result_ids:
            return []
        conn = await self._conn.get()
        cur = await conn.execute(
            f"SELECT result_id, dimension, value, source, metadata, created_at, id "
            f"FROM {self._scores} WHERE result_id = ANY(%s) ORDER BY id ASC",
            (result_ids,),
        )
        return [
            _ScoreRow(
                result_id=rid,
                dimension=dim,
                value=val,
                source=ScoreSource(src),
                metadata=meta,
                created_at=created.astimezone(UTC) if with_order else None,
                seq=seq if with_order else None,
            )
            for rid, dim, val, src, meta, created, seq in await cur.fetchall()
        ]

    async def query(self, q: FeedbackQuery) -> list[ScoredResult]:
        """Similarity + metadata filter search.

        Same contract as :meth:`SQLiteFeedbackLoop.query`. The metadata
        filters run in SQL against ``jsonb``; ranking stays in-process.
        """
        # Embed before taking the lock — it can call an external service.
        query_emb: np.ndarray | None = None
        query_norm = 0.0
        if q.similar_to is not None:
            query_emb = await self._embed(q.similar_to)
            query_norm = float(np.linalg.norm(query_emb))
            if query_norm == 0:
                return []

        sql = (
            f"SELECT id, content, metadata, embedding, created_at FROM {self._results} "
            "WHERE namespace = %s"
        )
        params: list[Any] = [self._namespace]
        if q.max_age is not None:
            sql += " AND created_at >= %s"
            params.append(datetime.now(UTC) - q.max_age)
        if q.agent_name is not None:
            sql += " AND metadata->>'agent_name' = %s"
            params.append(q.agent_name)
        if q.model is not None:
            sql += " AND metadata->>'model' = %s"
            params.append(q.model)
        if q.tags:
            sql += " AND jsonb_typeof(metadata->'tags') = 'array' AND metadata->'tags' ?| %s"
            params.append(list(q.tags))
        sql += " ORDER BY created_at DESC"

        want_effective = q.resolve_effective or q.effective_filters is not None
        async with self._get_db_lock():
            conn = await self._conn.get()
            # One read transaction: a concurrent score() batch cannot
            # produce a mixed before/after view across the two SELECTs.
            async with conn.transaction():
                rows = await (await conn.execute(sql, params)).fetchall()

                candidates: list[_QueryCandidate] = []
                for rid, content, meta, emb_bytes, created in rows:
                    similarity = 0.0
                    if query_emb is not None:
                        ranked = _cosine_similarity(query_emb, query_norm, emb_bytes)
                        if ranked is None:
                            continue
                        similarity = ranked
                    candidates.append((similarity, rid, content, meta, created.astimezone(UTC)))

                if query_emb is not None:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:_query_window(len(candidates), q.limit)]
                score_rows = await self._score_rows(
                    [c[1] for c in candidates], with_order=want_effective,
                )

        if not candidates:
            return []
        return _assemble_query_results(candidates, score_rows, q)

    async def get_human_examples(
        self,
        task_input: str,
        config: HumanFeedbackPromptConfig,
    ) -> HumanExampleSet:
        """Task-similar, human-graded positive/negative exemplars.

        Same contract as :meth:`SQLiteFeedbackLoop.get_human_examples`.
        """
        if not config.dimensions:
            return HumanExampleSet(positive=[], negative=[])
        if config.positive_limit == 0 and config.negative_limit == 0:
            return HumanExampleSet(positive=[], negative=[])

        query_emb = await self._embed(task_input)
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return HumanExampleSet(positive=[], negative=[])

        async with self._get_db_lock():
            conn = await self._conn.get()
            async with conn.transaction():
                rows = await (await conn.execute(
                    f"SELECT id, content, input, embedding FROM {self._results} "
                    "WHERE namespace = %s AND embedding IS NOT NULL ORDER BY created_at DESC",
                    (self._namespace,),
                )).fetchall()

                candidates: list[_ExampleCandidate] = []
                for rid, content, inp, emb_bytes in rows:
                    similarity = _cosine_similarity(query_emb, query_norm, emb_bytes)
                    if similarity is None:
                        continue
                    candidates.append((similarity, rid, content, inp))

                candidates.sort(key=lambda c: c[0], reverse=True)
                candidates = candidates[:_human_examples_window(len(candidates), config)]
                score_rows = await self._score_rows([c[1] for c in candidates], with_order=True)

        return _rank_human_examples(candidates, score_rows, config)

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        _validate_export_limit(limit)
        if limit == 0:
            return []
        _validate_export_range(since, min_score, max_score)

        sql = (
            f"SELECT r.id, r.content, r.input, r.metadata, "
            f"s.dimension, s.value, s.source, s.metadata "
            f"FROM {self._results} r LEFT JOIN {self._scores} s ON s.result_id = r.id "
            "WHERE r.namespace = %s"
        )
        params: list[Any] = [self._namespace]
        if since:
            sql += " AND r.created_at >= %s"
            params.append(since)
        sql += " ORDER BY r.created_at DESC, r.id, s.id ASC"

        async with self._get_db_lock():
            conn = await self._conn.get()
            rows = await (await conn.execute(sql, params)).fetchall()

        eval_rows = [
            _EvalRow(
                result_id=rid,
                content=content,
                input=inp,
                metadata=meta or {},
                dimension=dim,
                value=val,
                source=src,
                score_metadata=score_meta,
            )
            for rid, content, inp, meta, dim, val, src, score_meta in rows
        ]
        return _assemble_eval_cases(eval_rows, min_score, max_score, limit)

    async def close(self) -> None:
        await self._conn.close()
