from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite
import numpy as np

from jig._embed import ollama_embed
from jig._sqlite import LazyConnection, json_dumps, json_loads, parse_aware_utc
from jig.feedback.validation import validate_scores
from jig.core.types import (
    EffectiveScore,
    EffectiveScoreFilter,
    EvalCase,
    FeedbackLoop,
    FeedbackQuery,
    HumanExample,
    HumanExampleDimension,
    HumanExampleSet,
    HumanFeedbackPromptConfig,
    Score,
    ScoreSource,
    ScoredResult,
)

# Sources recognized in effective-score resolution. Exact-value match only —
# LLM_JUDGE, GROUND_TRUTH, and any unrecognized source string never become an
# effective quality judgment; they remain visible only in the full history.
_EFFECTIVE_SOURCES = (ScoreSource.HUMAN, ScoreSource.HEURISTIC)


def resolve_effective_scores(
    rows: list[tuple[str, float, ScoreSource, dict[str, Any] | None, datetime, int]],
) -> dict[str, EffectiveScore]:
    """Pick the effective score per dimension from one result's score rows.

    Public so external callers with their own already-fetched score rows
    (e.g. a CLI reading ``feedback.db`` directly for full-history display)
    can reuse the exact same resolution rule ``query()`` uses internally,
    instead of re-deriving it and risking drift.

    Precedence: the newest ``human`` row if any exists; otherwise the
    newest ``heuristic`` row; otherwise the dimension has no effective
    score. "Newest" breaks ties on ``created_at`` then rowid, both
    descending. A human row always outranks every heuristic row for its
    dimension, even one appended later — human judgment is only ever
    superseded by a later human judgment.
    """
    best: dict[str, tuple[tuple[int, datetime, int], EffectiveScore]] = {}
    for dim, value, source, metadata, created_at, rowid in rows:
        if source not in _EFFECTIVE_SOURCES:
            continue
        rank = (1 if source is ScoreSource.HUMAN else 0, created_at, rowid)
        current = best.get(dim)
        if current is not None and rank <= current[0]:
            continue
        best[dim] = (
            rank,
            EffectiveScore(
                dimension=dim,
                value=value,
                source=source,
                created_at=created_at,
                metadata=metadata,
            ),
        )
    return {dim: es for dim, (_, es) in best.items()}


def _passes_effective_filters(
    effective: dict[str, EffectiveScore],
    filters: list[EffectiveScoreFilter],
) -> bool:
    """AND across all filters; a missing effective dimension fails its filter."""
    for f in filters:
        es = effective.get(f.dimension)
        if es is None:
            return False
        if f.min_value is not None and es.value < f.min_value:
            return False
        if f.max_value is not None and es.value > f.max_value:
            return False
    return True


def _classify_human_example(
    effective: dict[str, EffectiveScore],
    dimensions: tuple[str, ...],
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[str, list[HumanExampleDimension]] | None:
    """Classify one result as a positive/negative exemplar, or omit it.

    Only ``source=human`` effective entries qualify — a dimension resolved
    to a heuristic row (no human grade yet) can never qualify a result,
    even though it's still visible in ``effective`` for other purposes.
    Negative precedence: any selected dimension at or below
    ``negative_threshold`` puts the whole result in the negative section,
    listing only the negative-crossing dimensions (never mixed with
    positive-crossing ones from the same result). Only when no dimension
    crosses negative does a positive-crossing dimension qualify the result
    as positive. A result crossing neither threshold on any dimension is
    omitted from both sections.
    """
    negative_dims: list[HumanExampleDimension] = []
    positive_dims: list[HumanExampleDimension] = []
    for dim in dimensions:
        es = effective.get(dim)
        if es is None or es.source is not ScoreSource.HUMAN:
            continue
        note = (es.metadata or {}).get("note") if es.metadata else None
        if es.value <= negative_threshold:
            negative_dims.append(HumanExampleDimension(dimension=dim, value=es.value, note=note))
        elif es.value >= positive_threshold:
            positive_dims.append(HumanExampleDimension(dimension=dim, value=es.value, note=note))
    if negative_dims:
        return "negative", negative_dims
    if positive_dims:
        return "positive", positive_dims
    return None


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
    created_at TEXT NOT NULL,
    metadata JSON
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
        self._db_lock: asyncio.Lock | None = None
        self._scores_metadata_migrated = False

    async def _get_db(self) -> aiosqlite.Connection:
        return await self._conn.get()

    def _get_db_lock(self) -> asyncio.Lock:
        if self._db_lock is None:
            self._db_lock = asyncio.Lock()
        return self._db_lock

    async def _ensure_scores_metadata_column(self, db: aiosqlite.Connection) -> None:
        """Idempotently add the ``scores.metadata`` column to older databases.

        Caller must already hold ``self._db_lock``. Fresh databases get the
        column from ``_SCHEMA`` directly; this covers databases created
        before the column existed, leaving every historical row NULL.
        """
        if self._scores_metadata_migrated:
            return
        cursor = await db.execute("PRAGMA table_info(scores)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "metadata" not in columns:
            await db.execute("ALTER TABLE scores ADD COLUMN metadata JSON")
            await db.commit()
        self._scores_metadata_migrated = True

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
        async with self._get_db_lock():
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
                        datetime.now(UTC).isoformat(),
                    ),
                )
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise
        return result_id

    async def score(self, result_id: str, scores: list[Score]) -> None:
        validate_scores(scores)
        # Serialize metadata before opening the batch transaction: a
        # non-JSON-serializable value must reject the whole call without
        # inserting any score, not fail partway through the batch.
        serialized_metadata: list[str | None] = []
        for s in scores:
            if s.metadata is None:
                serialized_metadata.append(None)
                continue
            try:
                serialized_metadata.append(json_dumps(s.metadata))
            except TypeError as exc:
                raise ValueError(
                    f"Score.metadata for dimension {s.dimension!r} is not "
                    f"JSON-serializable: {exc}"
                ) from exc
        now = datetime.now(UTC).isoformat()
        async with self._get_db_lock():
            db = await self._get_db()
            await self._ensure_scores_metadata_column(db)
            # Explicit BEGIN pins the batch boundary before the first insert.
            # The database lock keeps other feedback operations from
            # committing, rolling back, or reading this shared connection
            # until the batch is complete.
            try:
                await db.execute("BEGIN")
                for s, meta_json in zip(scores, serialized_metadata):
                    await db.execute(
                        "INSERT INTO scores (result_id, dimension, value, source, created_at, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (result_id, s.dimension, s.value, s.source.value, now, meta_json),
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
        # Compute the query embedding before taking the database lock —
        # it can call out to an external embedding service and must not
        # block writers/other readers while it does.
        query_emb: np.ndarray | None = None
        query_norm = 0.0
        if q.similar_to is not None:
            query_emb = await self._embed(q.similar_to)
            query_norm = float(np.linalg.norm(query_emb))
            if query_norm == 0:
                return []

        base_sql = (
            "SELECT id, content, input, metadata, embedding, created_at "
            "FROM results"
        )
        params: list[Any] = []
        if q.max_age is not None:
            cutoff_iso = (datetime.now(UTC) - q.max_age).isoformat()
            base_sql += " WHERE created_at >= ?"
            params.append(cutoff_iso)
        base_sql += " ORDER BY created_at DESC"

        async with self._get_db_lock():
            db = await self._get_db()
            await self._ensure_scores_metadata_column(db)
            try:
                await db.execute("BEGIN")
                rows = await (await db.execute(base_sql, params)).fetchall()

                # (similarity, rid, content, meta, created_at) — similarity=0 when not ranking
                candidates: list[tuple[float, str, str, dict[str, Any], datetime]] = []
                for rid, content, _inp, meta_json, emb_bytes, created_str in rows:
                    created = parse_aware_utc(created_str)
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

                want_effective = q.resolve_effective or q.effective_filters is not None
                score_columns = "result_id, dimension, value, source, metadata"
                if want_effective:
                    score_columns += ", created_at, rowid"
                score_rows: list[tuple[Any, ...]] = []
                if candidates:
                    # Pre-slice candidates before the batch fetch. Two reasons:
                    #  1. SQLite has a bound-parameter limit (commonly 999); a
                    #     very large corpus would exceed it.
                    #  2. Most of ``candidates`` past the q.limit-th will get
                    #     dropped by the min_score/effective filters anyway,
                    #     so fetching their scores is wasted work.
                    # The factor-of-10 cushion covers filter rejections; if
                    # the filter is aggressive enough that we undershoot
                    # q.limit, that's acceptable for v1 — properly chunking
                    # with early exit is a follow-up when corpora grow past
                    # ~100 results. Effective filters are applied below to
                    # this bounded window before the final q.limit truncation;
                    # aggressive filters can still undershoot q.limit because
                    # candidates outside the window are not scanned.
                    window = min(len(candidates), max(q.limit * 10, 50), 900)
                    candidates = candidates[:window]

                    # Batch-fetch scores for every surviving candidate with a
                    # single IN-list SELECT. Eliminates the per-row N+1 that
                    # would grow linearly with corpus size. Fetch created_at
                    # and rowid only when effective-score resolution needs them.
                    rids = [rid for _, rid, _, _, _ in candidates]
                    placeholders = ",".join(["?"] * len(rids))
                    score_rows = await (await db.execute(
                        f"SELECT {score_columns} "
                        f"FROM scores WHERE result_id IN ({placeholders}) ORDER BY rowid ASC",
                        rids,
                    )).fetchall()
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise

        if not candidates:
            return []

        scores_by_rid: dict[str, list[Score]] = {}
        effective_rows_by_rid: dict[
            str, list[tuple[str, float, ScoreSource, dict[str, Any] | None, datetime, int]]
        ] = {}
        for row in score_rows:
            rid, dim, val, src, score_meta_json = row[:5]
            score_meta = json_loads(score_meta_json) if score_meta_json else None
            source = ScoreSource(src)
            scores_by_rid.setdefault(rid, []).append(
                Score(dimension=dim, value=val, source=source, metadata=score_meta)
            )
            if want_effective:
                created_str, rowid = row[5:]
                effective_rows_by_rid.setdefault(rid, []).append(
                    (dim, val, source, score_meta, parse_aware_utc(created_str), rowid)
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

            effective_scores: dict[str, EffectiveScore] | None = None
            if want_effective:
                effective_scores = resolve_effective_scores(effective_rows_by_rid.get(rid, []))
                if q.effective_filters and not _passes_effective_filters(
                    effective_scores, q.effective_filters
                ):
                    continue

            results.append(ScoredResult(
                result_id=rid,
                content=content,
                scores=scores,
                avg_score=avg,
                metadata=meta,
                created_at=created,
                effective_scores=effective_scores,
            ))
            if len(results) >= q.limit:
                break

        return results

    async def get_human_examples(
        self,
        task_input: str,
        config: HumanFeedbackPromptConfig,
    ) -> HumanExampleSet:
        """Task-similar, human-graded positive/negative exemplars.

        Independent implementation from :meth:`query` rather than a thin
        wrapper: this needs the raw similarity score (to break exact ties
        deterministically on score recency, then result_id) and the input
        text (never returned by ``ScoredResult``), neither of which
        ``query()`` exposes without changing its public contract for every
        existing caller.

        Ranking: embedding similarity to ``task_input`` first (rows without
        a stored embedding are skipped — never substituted with recency).
        ``config.eligibility_filters`` (the safe-exemplar gate, e.g. a
        plausibility floor) is applied on the *effective* score before
        classification, so a result excluded there can never enter either
        section. Classification then looks only at ``source=human``
        effective scores for ``config.dimensions`` — see
        :func:`_classify_human_example` for the negative-precedence rule.
        Each result lands in at most one section; sections are independently
        capped at ``config.positive_limit`` / ``config.negative_limit``.
        """
        if not config.dimensions:
            return HumanExampleSet(positive=[], negative=[])

        query_emb = await self._embed(task_input)
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return HumanExampleSet(positive=[], negative=[])

        async with self._get_db_lock():
            db = await self._get_db()
            await self._ensure_scores_metadata_column(db)
            try:
                await db.execute("BEGIN")
                rows = await (await db.execute(
                    "SELECT id, content, input, embedding, created_at FROM results "
                    "ORDER BY created_at DESC"
                )).fetchall()

                # (similarity, result_id, output, input_text)
                candidates: list[tuple[float, str, str, str]] = []
                for rid, content, inp, emb_bytes, _created_str in rows:
                    if not emb_bytes:
                        continue
                    row_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    row_norm = float(np.linalg.norm(row_emb))
                    if row_norm == 0:
                        continue
                    similarity = float(np.dot(query_emb, row_emb) / (query_norm * row_norm))
                    candidates.append((similarity, rid, content, inp))

                candidates.sort(key=lambda c: c[0], reverse=True)

                # Same bounded-window rationale as query(): most candidates
                # past a generous multiple of the requested section limits
                # would be dropped by classification/eligibility anyway.
                window = min(
                    len(candidates),
                    max((config.positive_limit + config.negative_limit) * 10, 50),
                    900,
                )
                candidates = candidates[:window]

                score_rows: list[tuple[str, str, float, str, str | None, str, int]] = []
                if candidates:
                    rids = [c[1] for c in candidates]
                    placeholders = ",".join(["?"] * len(rids))
                    score_rows = await (await db.execute(
                        f"SELECT result_id, dimension, value, source, metadata, created_at, rowid "
                        f"FROM scores WHERE result_id IN ({placeholders}) ORDER BY rowid ASC",
                        rids,
                    )).fetchall()
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise

        effective_rows_by_rid: dict[
            str, list[tuple[str, float, ScoreSource, dict[str, Any] | None, datetime, int]]
        ] = {}
        for rid, dim, val, src, meta_json, created_str, rowid in score_rows:
            meta = json_loads(meta_json) if meta_json else None
            effective_rows_by_rid.setdefault(rid, []).append(
                (dim, val, ScoreSource(src), meta, parse_aware_utc(created_str), rowid)
            )

        eligibility_filters = list(config.eligibility_filters)
        # (similarity, score_recency_ts, result_id, HumanExample)
        positive_ranked: list[tuple[float, float, str, HumanExample]] = []
        negative_ranked: list[tuple[float, float, str, HumanExample]] = []
        for similarity, rid, content, inp in candidates:
            effective = resolve_effective_scores(effective_rows_by_rid.get(rid, []))
            if eligibility_filters and not _passes_effective_filters(effective, eligibility_filters):
                continue
            classified = _classify_human_example(
                effective, config.dimensions, config.positive_threshold, config.negative_threshold,
            )
            if classified is None:
                continue
            classification, dims = classified
            recency_ts = max(effective[d.dimension].created_at for d in dims).timestamp()
            example = HumanExample(
                result_id=rid,
                input_text=inp,
                output=content,
                classification=classification,
                dimensions=dims,
            )
            entry = (similarity, recency_ts, rid, example)
            if classification == "negative":
                negative_ranked.append(entry)
            else:
                positive_ranked.append(entry)

        def _sort_key(entry: tuple[float, float, str, HumanExample]) -> tuple[float, float, str]:
            similarity, recency_ts, rid, _example = entry
            return (-similarity, -recency_ts, rid)

        positive_ranked.sort(key=_sort_key)
        negative_ranked.sort(key=_sort_key)

        return HumanExampleSet(
            positive=[e[3] for e in positive_ranked[: config.positive_limit]],
            negative=[e[3] for e in negative_ranked[: config.negative_limit]],
        )

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
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "export_eval_set since must be an aware datetime, got a naive one"
            )

        # Single LEFT JOIN ordered by result creation then score rowid,
        # replacing the former per-result N+1 score SELECT. Rows are
        # materialized inside one locked read transaction so a concurrent
        # score() batch can't produce a mixed before/after view.
        sql = (
            "SELECT r.id, r.content, r.input, r.metadata, r.created_at, "
            "s.dimension, s.value, s.source, s.metadata, s.rowid "
            "FROM results r LEFT JOIN scores s ON s.result_id = r.id"
        )
        params: list[Any] = []
        if since:
            sql += " WHERE r.created_at >= ?"
            params.append(since.astimezone(UTC).isoformat())
        sql += " ORDER BY r.created_at DESC, s.rowid ASC"

        async with self._get_db_lock():
            db = await self._get_db()
            await self._ensure_scores_metadata_column(db)
            try:
                await db.execute("BEGIN")
                rows = await (await db.execute(sql, params)).fetchall()
                await db.commit()
            except BaseException:
                await _rollback_safely(db)
                raise

        # Group joined rows by result, preserving SQL order (result creation
        # desc, then score rowid asc within each result).
        order: list[str] = []
        grouped: dict[str, dict[str, Any]] = {}
        for rid, content, inp, meta_json, _created_at, dim, val, src, score_meta_json, _score_rowid in rows:
            if rid not in grouped:
                grouped[rid] = {"content": content, "input": inp, "meta_json": meta_json, "scores": []}
                order.append(rid)
            if dim is not None:
                grouped[rid]["scores"].append((dim, val, src, score_meta_json))

        cases: list[EvalCase] = []
        for rid in order:
            entry = grouped[rid]
            score_rows_for_result = entry["scores"]
            if not score_rows_for_result:
                # Unscored rows don't qualify when any score filter is active;
                # also skip them unconditionally (consistent with query()).
                continue

            avg = sum(val for _, val, _, _ in score_rows_for_result) / len(score_rows_for_result)
            if min_score is not None and avg < min_score:
                continue
            if max_score is not None and avg > max_score:
                continue

            meta = json_loads(entry["meta_json"]) if entry["meta_json"] else {}
            meta["avg_score"] = avg
            meta["scores"] = [
                {
                    "dimension": dim,
                    "value": val,
                    "source": src,
                    "metadata": json_loads(score_meta_json) if score_meta_json else None,
                }
                for dim, val, src, score_meta_json in score_rows_for_result
            ]
            cases.append(
                EvalCase(
                    input=entry["input"],
                    expected=entry["content"],
                    metadata=meta,
                )
            )
            if limit is not None and len(cases) >= limit:
                break

        return cases

    async def close(self) -> None:
        await self._conn.close()
