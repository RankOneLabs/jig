from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
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


# The transforms below are the backend-independent half of every feedback
# read. Each backend fetches rows its own way and hands them here, so the
# ranking, filtering and assembly rules exist once and cannot drift between
# SQLiteFeedbackLoop and PostgresFeedbackLoop.


@dataclass(frozen=True, slots=True)
class _ScoreRow:
    """One stored score as every feedback backend reads it.

    ``created_at`` and ``seq`` are fetched only when effective-score
    resolution needs them. ``seq`` is the backend's insertion order (SQLite
    rowid, PostgreSQL identity) — the documented tie-breaker.
    """

    result_id: str
    dimension: str
    value: float
    source: ScoreSource
    metadata: dict[str, Any] | None
    created_at: datetime | None = None
    seq: int | None = None


@dataclass(frozen=True, slots=True)
class _EvalRow:
    """One result-joined-to-score row for ``export_eval_set``.

    The score fields are all ``None`` for a result with no scores (the
    outer join's unmatched side).
    """

    result_id: str
    content: str
    input: str
    metadata: dict[str, Any]
    dimension: str | None
    value: float | None
    source: str | None
    score_metadata: dict[str, Any] | None


# (similarity, result_id, content, metadata, created_at) — similarity=0 when not ranking
type _QueryCandidate = tuple[float, str, str, dict[str, Any], datetime]
# (similarity, result_id, output, input_text)
type _ExampleCandidate = tuple[float, str, str, str]


def _cosine_similarity(
    query_emb: np.ndarray, query_norm: float, emb_bytes: bytes | None,
) -> float | None:
    """Cosine similarity to a stored float32 embedding, or ``None`` to skip.

    A missing embedding, a zero vector, or a dimension mismatch (older row,
    or a since-changed embed model) all mean "not rankable" — the row is
    skipped rather than substituted with recency or allowed to crash the run.
    """
    if not emb_bytes:
        return None
    row_emb = np.frombuffer(emb_bytes, dtype=np.float32)
    if row_emb.shape != query_emb.shape:
        return None
    row_norm = float(np.linalg.norm(row_emb))
    if row_norm == 0:
        return None
    return float(np.dot(query_emb, row_emb) / (query_norm * row_norm))


def _query_window(candidate_count: int, limit: int) -> int:
    """How many ranked candidates ``query()`` fetches scores for.

    Most candidates past the q.limit-th get dropped by the min_score /
    effective filters anyway, so fetching their scores is wasted work; the
    factor-of-10 cushion covers filter rejections. 900 stays under SQLite's
    bound-parameter limit (commonly 999). Aggressive filters can still
    undershoot q.limit because candidates outside the window are not scanned
    — acceptable for v1; proper chunking with early exit is a follow-up when
    corpora grow past ~100 results.
    """
    return min(candidate_count, max(limit * 10, 50), 900)


def _human_examples_window(candidate_count: int, config: HumanFeedbackPromptConfig) -> int:
    """Same bounded-window rationale as :func:`_query_window`, sized by the
    requested section limits."""
    return min(
        candidate_count,
        max((config.positive_limit + config.negative_limit) * 10, 50),
        900,
    )


def _effective_rows(
    score_rows: Sequence[_ScoreRow],
) -> dict[str, list[tuple[str, float, ScoreSource, dict[str, Any] | None, datetime, int]]]:
    """Group score rows by result into :func:`resolve_effective_scores` input."""
    grouped: dict[
        str, list[tuple[str, float, ScoreSource, dict[str, Any] | None, datetime, int]]
    ] = {}
    for row in score_rows:
        if row.created_at is None or row.seq is None:
            raise ValueError(
                f"score row for result {row.result_id!r} lacks created_at/seq, "
                "which effective-score resolution requires"
            )
        grouped.setdefault(row.result_id, []).append(
            (row.dimension, row.value, row.source, row.metadata, row.created_at, row.seq)
        )
    return grouped


def _assemble_query_results(
    candidates: Sequence[_QueryCandidate],
    score_rows: Sequence[_ScoreRow],
    q: FeedbackQuery,
) -> list[ScoredResult]:
    """Turn ranked candidates and their score rows into ``query()``'s answer.

    Candidates arrive already ranked (similarity, else recency). Results
    with no scores — or none from ``q.source`` — are skipped; ``min_score``
    applies to the average over the source-filtered scores; effective
    filters apply before the final ``q.limit`` truncation.
    """
    want_effective = q.resolve_effective or q.effective_filters is not None
    scores_by_rid: dict[str, list[Score]] = {}
    for row in score_rows:
        scores_by_rid.setdefault(row.result_id, []).append(
            Score(dimension=row.dimension, value=row.value, source=row.source, metadata=row.metadata)
        )
    effective_rows_by_rid = _effective_rows(score_rows) if want_effective else {}

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


def _rank_human_examples(
    candidates: Sequence[_ExampleCandidate],
    score_rows: Sequence[_ScoreRow],
    config: HumanFeedbackPromptConfig,
) -> HumanExampleSet:
    """Classify and rank similarity-ordered candidates into exemplar sections.

    ``config.eligibility_filters`` gate on the effective score before
    classification; ties on similarity break on score recency, then
    result_id. Each section is capped independently.
    """
    effective_rows_by_rid = _effective_rows(score_rows)
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


def _validate_export_limit(limit: int | None) -> None:
    """Reject a malformed ``export_eval_set`` limit. Checked first: a valid
    ``limit=0`` returns ``[]`` before any other argument is examined."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError(
            f"export_eval_set limit must be a non-negative integer, got {limit!r}"
        )


def _validate_export_range(
    since: datetime | None,
    min_score: float | None,
    max_score: float | None,
) -> None:
    """Reject a malformed ``export_eval_set`` score range or cutoff."""
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


def _assemble_eval_cases(
    rows: Sequence[_EvalRow],
    min_score: float | None,
    max_score: float | None,
    limit: int | None,
) -> list[EvalCase]:
    """Group joined rows by result into eval cases, preserving row order.

    Rows arrive ordered by result creation descending, then score insertion
    order within each result. Unscored results are always skipped
    (consistent with ``query()``).
    """
    order: list[str] = []
    grouped: dict[str, list[_EvalRow]] = {}
    for row in rows:
        if row.result_id not in grouped:
            grouped[row.result_id] = []
            order.append(row.result_id)
        if row.dimension is not None:
            grouped[row.result_id].append(row)

    first_row = {row.result_id: row for row in reversed(rows)}
    cases: list[EvalCase] = []
    for rid in order:
        scored = grouped[rid]
        if not scored:
            continue
        values = [r.value for r in scored if r.value is not None]
        avg = sum(values) / len(values)
        if min_score is not None and avg < min_score:
            continue
        if max_score is not None and avg > max_score:
            continue

        head = first_row[rid]
        meta = dict(head.metadata)
        meta["avg_score"] = avg
        meta["scores"] = [
            {
                "dimension": r.dimension,
                "value": r.value,
                "source": r.source,
                "metadata": r.score_metadata,
            }
            for r in scored
        ]
        cases.append(EvalCase(input=head.input, expected=head.content, metadata=meta))
        if limit is not None and len(cases) >= limit:
            break

    return cases


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

                candidates: list[_QueryCandidate] = []
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
                        ranked = _cosine_similarity(query_emb, query_norm, emb_bytes)
                        if ranked is None:
                            continue
                        similarity = ranked
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
                    # Pre-slice before the batch fetch — see _query_window.
                    candidates = candidates[:_query_window(len(candidates), q.limit)]

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

        parsed_rows = [
            _ScoreRow(
                result_id=row[0],
                dimension=row[1],
                value=row[2],
                source=ScoreSource(row[3]),
                metadata=json_loads(row[4]) if row[4] else None,
                created_at=parse_aware_utc(row[5]) if want_effective else None,
                seq=row[6] if want_effective else None,
            )
            for row in score_rows
        ]
        return _assemble_query_results(candidates, parsed_rows, q)

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
        if config.positive_limit == 0 and config.negative_limit == 0:
            # Every result would be sliced away by [:0] regardless of what
            # qualifies — skip the embed call and full table scan entirely
            # rather than doing that work just to discard it.
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

                candidates: list[_ExampleCandidate] = []
                for rid, content, inp, emb_bytes, _created_str in rows:
                    # Missing, zero or dimension-mismatched embeddings are
                    # skipped (still gradeable, never a prompt example).
                    similarity = _cosine_similarity(query_emb, query_norm, emb_bytes)
                    if similarity is None:
                        continue
                    candidates.append((similarity, rid, content, inp))

                candidates.sort(key=lambda c: c[0], reverse=True)
                candidates = candidates[:_human_examples_window(len(candidates), config)]

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

        parsed_rows = [
            _ScoreRow(
                result_id=rid,
                dimension=dim,
                value=val,
                source=ScoreSource(src),
                metadata=json_loads(meta_json) if meta_json else None,
                created_at=parse_aware_utc(created_str),
                seq=rowid,
            )
            for rid, dim, val, src, meta_json, created_str, rowid in score_rows
        ]
        return _rank_human_examples(candidates, parsed_rows, config)

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

        eval_rows = [
            _EvalRow(
                result_id=rid,
                content=content,
                input=inp,
                metadata=json_loads(meta_json) if meta_json else {},
                dimension=dim,
                value=val,
                source=src,
                score_metadata=json_loads(score_meta_json) if score_meta_json else None,
            )
            for rid, content, inp, meta_json, _created_at, dim, val, src, score_meta_json, _score_rowid in rows
        ]
        return _assemble_eval_cases(eval_rows, min_score, max_score, limit)

    async def close(self) -> None:
        await self._conn.close()
