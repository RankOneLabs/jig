from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import aiosqlite
import numpy as np

from jig._embed import ollama_embed
from jig._sqlite import LazyConnection, json_dumps, json_loads
from jig.feedback.validation import validate_scores
from jig.core.types import (
    EvalCase,
    FeedbackLoop,
    FeedbackQuery,
    Score,
    ScoreSource,
    ScoredResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    input TEXT NOT NULL,
    metadata JSON,
    embedding BLOB,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    result_id TEXT NOT NULL REFERENCES results(id),
    dimension TEXT NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_result ON scores(result_id);
"""


async def _rollback_safely(db: aiosqlite.Connection) -> None:
    rollback = asyncio.create_task(db.rollback())
    try:
        await asyncio.shield(rollback)
    except BaseException:
        await rollback
        raise


class SQLiteFeedbackLoop(FeedbackLoop):
    def __init__(
        self,
        db_path: str = "jig_feedback.db",
        embed_model: str = "nomic-embed-text",
        ollama_host: str | None = None,
    ):
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._conn = LazyConnection(db_path, _SCHEMA)
        self._write_lock: asyncio.Lock | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        return await self._conn.get()

    def _get_write_lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def _embed(self, text: str) -> np.ndarray:
        return await ollama_embed(text, self._embed_model, self._ollama_host)

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        result_id = str(uuid.uuid4())
        embedding = await self._embed(input_text)
        async with self._get_write_lock():
            db = await self._get_db()
            try:
                await db.execute(
                    "INSERT INTO results (id, content, input, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        result_id,
                        content,
                        input_text,
                        json_dumps(metadata or {}),
                        embedding.tobytes(),
                        datetime.now().isoformat(),
                    ),
                )
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise
        return result_id

    async def score(self, result_id: str, scores: list[Score]) -> None:
        validate_scores(scores)
        now = datetime.now().isoformat()
        async with self._get_write_lock():
            db = await self._get_db()
            # Explicit BEGIN pins the batch boundary before the first insert.
            # The write lock keeps other feedback writes from committing or
            # rolling back this shared connection until the batch is complete.
            try:
                await db.execute("BEGIN")
                for s in scores:
                    await db.execute(
                        "INSERT INTO scores (result_id, dimension, value, source, created_at) VALUES (?, ?, ?, ?, ?)",
                        (result_id, s.dimension, s.value, s.source.value, now),
                    )
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        q = FeedbackQuery(similar_to=query, limit=limit, min_score=min_score, source=source)
        return await self.query(q)

    async def query(self, q: FeedbackQuery) -> list[ScoredResult]:
        """Similarity + metadata filter search.

        Embedding similarity ranks candidates when ``q.similar_to`` is
        set; otherwise rows are ordered by ``created_at`` descending.
        ``agent_name``, ``model``, and ``tags`` filter on fields the
        caller set via :meth:`store_result`'s metadata.
        """
        db = await self._get_db()

        # Push max_age into SQL; JSON metadata filters stay in Python
        # since SQLite JSON support is uneven across builds and the
        # filter shapes are simple enough to be fast in-memory.
        base_sql = (
            "SELECT id, content, input, metadata, embedding, created_at "
            "FROM results"
        )
        params: list[Any] = []
        if q.max_age is not None:
            cutoff_iso = (datetime.now() - q.max_age).isoformat()
            base_sql += " WHERE created_at >= ?"
            params.append(cutoff_iso)
        base_sql += " ORDER BY created_at DESC"
        rows = await (await db.execute(base_sql, params)).fetchall()

        # Optional similarity ranking
        query_emb: np.ndarray | None = None
        query_norm = 0.0
        if q.similar_to is not None:
            query_emb = await self._embed(q.similar_to)
            query_norm = float(np.linalg.norm(query_emb))
            if query_norm == 0:
                return []

        # (similarity, rid, content, meta, created_at) — similarity=0 when not ranking
        candidates: list[tuple[float, str, str, dict[str, Any], datetime]] = []
        for rid, content, _inp, meta_json, emb_bytes, created_str in rows:
            created = datetime.fromisoformat(created_str)
            meta = json_loads(meta_json) if meta_json else {}
            if q.agent_name is not None and meta.get("agent_name") != q.agent_name:
                continue
            if q.model is not None and meta.get("model") != q.model:
                continue
            if q.tags:
                row_tags = meta.get("tags") or []
                if not set(q.tags).intersection(row_tags):
                    continue

            similarity = 0.0
            if query_emb is not None:
                if not emb_bytes:
                    continue
                row_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                row_norm = float(np.linalg.norm(row_emb))
                if row_norm == 0:
                    continue
                similarity = float(np.dot(query_emb, row_emb) / (query_norm * row_norm))
            candidates.append((similarity, rid, content, meta, created))

        if query_emb is not None:
            candidates.sort(key=lambda x: x[0], reverse=True)
        # Without similarity, the initial ORDER BY created_at DESC already
        # gave us recency order.

        if not candidates:
            return []

        # Pre-slice candidates before the batch fetch. Two reasons:
        #  1. SQLite has a bound-parameter limit (commonly 999); a very
        #     large corpus would exceed it.
        #  2. Most of ``candidates`` past the q.limit-th will get dropped
        #     by the min_score filter anyway, so fetching their scores
        #     is wasted work.
        # The factor-of-10 cushion covers min_score rejections; if the
        # filter is aggressive enough that we undershoot q.limit, that's
        # acceptable for v1 — properly chunking with early exit is a
        # follow-up when corpora grow past ~100 results.
        window = min(len(candidates), max(q.limit * 10, 50), 900)
        candidates = candidates[:window]

        # Batch-fetch scores for every surviving candidate with a single
        # IN-list SELECT. Eliminates the per-row N+1 that would grow
        # linearly with corpus size.
        rids = [rid for _, rid, _, _, _ in candidates]
        placeholders = ",".join(["?"] * len(rids))
        score_rows = await (await db.execute(
            f"SELECT result_id, dimension, value, source FROM scores "
            f"WHERE result_id IN ({placeholders})",
            rids,
        )).fetchall()
        scores_by_rid: dict[str, list[Score]] = {}
        for rid, dim, val, src in score_rows:
            scores_by_rid.setdefault(rid, []).append(
                Score(dimension=dim, value=val, source=ScoreSource(src))
            )

        results: list[ScoredResult] = []
        for _sim, rid, content, meta, created in candidates:
            scores = scores_by_rid.get(rid, [])
            if not scores:
                continue
            if q.source is not None:
                scores = [s for s in scores if s.source == q.source]
                if not scores:
                    continue
            avg = sum(s.value for s in scores) / len(scores)
            if q.min_score is not None and avg < q.min_score:
                continue

            results.append(ScoredResult(
                result_id=rid,
                content=content,
                scores=scores,
                avg_score=avg,
                metadata=meta,
                created_at=created,
            ))
            if len(results) >= q.limit:
                break

        return results

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError(
                    f"export_eval_set limit must be a non-negative integer, got {limit!r}"
                )
            if limit == 0:
                return []
        if (
            min_score is not None
            and max_score is not None
            and min_score > max_score
        ):
            raise ValueError(
                f"min_score ({min_score}) must not exceed max_score ({max_score})"
            )

        db = await self._get_db()
        sql = "SELECT id, content, input, metadata, created_at FROM results"
        params: list[Any] = []
        if since:
            sql += " WHERE created_at >= ?"
            params.append(since.isoformat())
        sql += " ORDER BY created_at DESC"

        # Full result set is materialized before filtering; acceptable for current corpus
        # sizes. A windowed/streaming approach (like query()'s factor-of-10 window) is a
        # known follow-up for large DBs.
        rows = await (await db.execute(sql, params)).fetchall()

        cases: list[EvalCase] = []
        for rid, content, inp, meta_json, _ in rows:
            score_rows = await (await db.execute(
                "SELECT dimension, value, source FROM scores WHERE result_id = ?",
                (rid,),
            )).fetchall()

            if not score_rows:
                # Unscored rows don't qualify when any score filter is active;
                # also skip them unconditionally (consistent with query()).
                continue

            avg = sum(v for _, v, _ in score_rows) / len(score_rows)
            if min_score is not None and avg < min_score:
                continue
            if max_score is not None and avg > max_score:
                continue

            meta = json_loads(meta_json) if meta_json else {}
            meta["avg_score"] = avg
            cases.append(
                EvalCase(
                    input=inp,
                    expected=content,
                    metadata=meta,
                )
            )
            if limit is not None and len(cases) >= limit:
                break

        return cases

    async def close(self) -> None:
        await self._conn.close()
