from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import aiosqlite
import numpy as np

from jig._embed import ollama_embed
from jig.core.types import (
    EvalCase,
    FeedbackLoop,
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


class SQLiteFeedbackLoop(FeedbackLoop):
    def __init__(
        self,
        db_path: str = "jig_feedback.db",
        embed_model: str = "nomic-embed-text",
        ollama_host: str | None = None,
    ):
        self._db_path = db_path
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.executescript(_SCHEMA)
        return self._db

    async def _embed(self, text: str) -> np.ndarray:
        return await ollama_embed(text, self._embed_model, self._ollama_host)

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        db = await self._get_db()
        result_id = str(uuid.uuid4())
        embedding = await self._embed(input_text)
        await db.execute(
            "INSERT INTO results (id, content, input, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result_id,
                content,
                input_text,
                json.dumps(metadata or {}),
                embedding.tobytes(),
                datetime.now().isoformat(),
            ),
        )
        await db.commit()
        return result_id

    async def score(self, result_id: str, scores: list[Score]) -> None:
        db = await self._get_db()
        now = datetime.now().isoformat()
        for s in scores:
            await db.execute(
                "INSERT INTO scores (result_id, dimension, value, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (result_id, s.dimension, s.value, s.source.value, now),
            )
        await db.commit()

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        db = await self._get_db()
        query_emb = await self._embed(query)
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return []

        cursor = await db.execute(
            "SELECT id, content, input, metadata, embedding, created_at FROM results"
        )
        rows = await cursor.fetchall()

        candidates: list[tuple[float, str, str, dict[str, Any], datetime]] = []
        for row in rows:
            rid, content, inp, meta_json, emb_bytes, created_str = row
            if not emb_bytes:
                continue
            row_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            row_norm = float(np.linalg.norm(row_emb))
            if row_norm == 0:
                continue
            sim = float(np.dot(query_emb, row_emb) / (query_norm * row_norm))
            candidates.append(
                (sim, rid, content, json.loads(meta_json), datetime.fromisoformat(created_str))
            )

        candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[ScoredResult] = []
        for sim, rid, content, meta, created in candidates[:limit * 2]:
            score_cursor = await db.execute(
                "SELECT dimension, value, source FROM scores WHERE result_id = ?", (rid,)
            )
            score_rows = await score_cursor.fetchall()
            scores = [
                Score(dimension=d, value=v, source=ScoreSource(s))
                for d, v, s in score_rows
            ]
            if not scores:
                continue

            if source:
                scores = [s for s in scores if s.source == source]
                if not scores:
                    continue

            avg = sum(s.value for s in scores) / len(scores)
            if min_score is not None and avg < min_score:
                continue

            results.append(
                ScoredResult(
                    result_id=rid,
                    content=content,
                    scores=scores,
                    avg_score=avg,
                    metadata=meta,
                    created_at=created,
                )
            )
            if len(results) >= limit:
                break

        return results

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        db = await self._get_db()
        query = "SELECT id, content, input, metadata, created_at FROM results"
        params: list[Any] = []
        if since:
            query += " WHERE created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        cases: list[EvalCase] = []
        for rid, content, inp, meta_json, _ in rows:
            score_cursor = await db.execute(
                "SELECT dimension, value, source FROM scores WHERE result_id = ?", (rid,)
            )
            score_rows = await score_cursor.fetchall()
            if not score_rows:
                continue

            avg = sum(v for _, v, _ in score_rows) / len(score_rows)
            if min_score is not None and avg < min_score:
                continue
            if max_score is not None and avg > max_score:
                continue

            meta = json.loads(meta_json)
            meta["avg_score"] = avg
            cases.append(
                EvalCase(
                    input=inp,
                    expected=content,
                    metadata=meta,
                )
            )

        return cases

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
