"""Run the feedback-loop test suites against every backend.

``open_feedback_loop`` builds the loop a parametrized ``feedback_db`` fixture
asked for. The raw-row helpers exist for the handful of tests that must plant
state ``store_result`` / ``score`` cannot produce — an explicit score
timestamp, a missing or mismatched embedding — and write it the way each
backend actually stores it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jig._sqlite import json_dumps
from jig.core.types import FeedbackLoop
from jig.feedback.loop import SQLiteFeedbackLoop

BACKENDS = ("sqlite", "postgres")


def open_feedback_loop(request: pytest.FixtureRequest, tmp_path: Path) -> FeedbackLoop:
    if request.param == "sqlite":
        return SQLiteFeedbackLoop(db_path=str(tmp_path / "test.db"))
    from jig.feedback.postgres import PostgresFeedbackLoop

    return PostgresFeedbackLoop(
        request.getfixturevalue("pg_dsn"), schema=request.getfixturevalue("pg_schema"),
    )


async def insert_score_row(
    loop: FeedbackLoop,
    result_id: str,
    dimension: str,
    value: float,
    source: str,
    created_at: str,
    metadata: dict[str, Any] | None,
) -> None:
    """Append a score row with an explicit ``created_at``; insertion order
    still drives the backend's tie-breaker (rowid / identity)."""
    meta_json = json_dumps(metadata) if metadata is not None else None
    if isinstance(loop, SQLiteFeedbackLoop):
        db = await loop._get_db()
        await loop._ensure_scores_metadata_column(db)
        await db.execute(
            "INSERT INTO scores (result_id, dimension, value, source, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (result_id, dimension, value, source, created_at, meta_json),
        )
        await db.commit()
        return
    conn = await loop._conn.get()  # type: ignore[attr-defined]
    await conn.execute(
        f"INSERT INTO {loop._scores} "  # type: ignore[attr-defined]
        "(result_id, dimension, value, source, created_at, metadata) "
        "VALUES (%s, %s, %s, %s, %s::timestamptz, %s::jsonb)",
        (result_id, dimension, value, source, created_at, meta_json),
    )


async def insert_result_row(
    loop: FeedbackLoop,
    result_id: str,
    content: str,
    input_text: str,
    embedding: bytes | None,
    created_at: str,
) -> None:
    """Insert a results row directly — e.g. a legacy row with no embedding."""
    if isinstance(loop, SQLiteFeedbackLoop):
        db = await loop._get_db()
        await db.execute(
            "INSERT INTO results (id, content, input, metadata, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (result_id, content, input_text, "{}", embedding, created_at),
        )
        await db.commit()
        return
    conn = await loop._conn.get()  # type: ignore[attr-defined]
    await conn.execute(
        f"INSERT INTO {loop._results} "  # type: ignore[attr-defined]
        "(id, namespace, content, input, metadata, embedding, created_at) "
        "VALUES (%s, %s, %s, %s, '{}'::jsonb, %s, %s::timestamptz)",
        (result_id, loop._namespace, content, input_text, embedding, created_at),  # type: ignore[attr-defined]
    )


async def set_embedding(loop: FeedbackLoop, result_id: str, embedding: bytes) -> None:
    """Overwrite a stored embedding (e.g. with a mismatched dimension)."""
    if isinstance(loop, SQLiteFeedbackLoop):
        db = await loop._get_db()
        await db.execute(
            "UPDATE results SET embedding = ? WHERE id = ?", (embedding, result_id),
        )
        await db.commit()
        return
    conn = await loop._conn.get()  # type: ignore[attr-defined]
    await conn.execute(
        f"UPDATE {loop._results} SET embedding = %s WHERE id = %s",  # type: ignore[attr-defined]
        (embedding, result_id),
    )
