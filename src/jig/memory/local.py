"""Local SQLite-backed memory: ``SqliteStore`` + ``DenseRetriever``.

This is the split of the old monolithic ``LocalMemory`` class. The store
owns persistence (content + metadata + pre-computed embeddings + session
history). The retriever owns the ranking strategy.

For the common "just use local memory" case, :func:`LocalMemory`
instantiates both and returns them as a pair::

    store, retriever = LocalMemory(db_path="agent.db")
    config = AgentConfig(..., store=store, retriever=retriever)

For experimentation, swap the retriever without touching the corpus::

    store, dense = LocalMemory(db_path="agent.db")
    hybrid = HybridRetriever(dense, BM25Retriever(store))  # future
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable

import aiosqlite
import numpy as np

from jig._embed import ollama_embed
from jig.core.types import MemoryEntry, MemoryStore, Message, Retriever, Role, ToolCall

logger = logging.getLogger(__name__)

Embedder = Callable[[str], Awaitable[np.ndarray]]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSON,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    tool_calls JSON,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id, created_at);
"""


class SqliteStore(MemoryStore):
    """Persistence for memory entries with pre-computed embeddings.

    Embeddings are computed on ``add`` via the configured embedder and
    stored alongside the row so retrievers can fetch them without
    re-embedding the corpus. :class:`DenseRetriever` uses this
    capability — retrievers that don't need embeddings (BM25) can
    operate on ``content`` alone.
    """

    def __init__(
        self,
        db_path: str = "jig_memory.db",
        embed_model: str = "nomic-embed-text",
        ollama_host: str | None = None,
        embedder: Embedder | None = None,
    ):
        self._db_path = db_path
        self._embed_model = embed_model
        self._ollama_host = ollama_host
        self._custom_embedder = embedder
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.executescript(_SCHEMA)
        return self._db

    async def embed(self, text: str) -> np.ndarray:
        """Public so :class:`DenseRetriever` can use the same embedder."""
        if self._custom_embedder is not None:
            return await self._custom_embedder(text)
        return await ollama_embed(text, self._embed_model, self._ollama_host)

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        db = await self._get_db()
        entry_id = str(uuid.uuid4())
        embedding = await self.embed(content)
        await db.execute(
            "INSERT INTO memories (id, content, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                entry_id,
                content,
                json.dumps(metadata or {}),
                embedding.tobytes(),
                datetime.now().isoformat(),
            ),
        )
        await db.commit()
        return entry_id

    async def get(self, id: str) -> MemoryEntry | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT content, metadata, created_at FROM memories WHERE id = ?",
            (id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        content, meta_json, created_str = row
        return MemoryEntry(
            id=id,
            content=content,
            metadata=json.loads(meta_json) if meta_json else {},
            created_at=datetime.fromisoformat(created_str),
        )

    async def all(self) -> list[MemoryEntry]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT id, content, metadata, created_at FROM memories ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            MemoryEntry(
                id=rid,
                content=content,
                metadata=json.loads(meta_json) if meta_json else {},
                created_at=datetime.fromisoformat(created_str),
            )
            for rid, content, meta_json, created_str in rows
        ]

    async def delete(self, id: str) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM memories WHERE id = ?", (id,))
        await db.commit()

    async def iter_entries_with_embeddings(
        self,
    ) -> list[tuple[MemoryEntry, np.ndarray]]:
        """Fetch entries paired with their stored embeddings.

        The lower-level hook :class:`DenseRetriever` uses — we return a
        list rather than an async iterator because sqlite fetches the
        full result set into memory anyway, and retrievers need to sort
        globally over the result set.
        """
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT id, content, metadata, embedding, created_at FROM memories"
        )
        rows = await cursor.fetchall()
        out: list[tuple[MemoryEntry, np.ndarray]] = []
        for rid, content, meta_json, emb_bytes, created_str in rows:
            if not emb_bytes:
                continue
            entry = MemoryEntry(
                id=rid,
                content=content,
                metadata=json.loads(meta_json) if meta_json else {},
                created_at=datetime.fromisoformat(created_str),
            )
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            out.append((entry, emb))
        return out

    async def get_session(self, session_id: str) -> list[Message]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT role, content, tool_call_id, tool_calls "
            "FROM sessions WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        messages: list[Message] = []
        for role_str, content, tool_call_id, tool_calls_json in rows:
            tool_calls = None
            if tool_calls_json:
                raw = json.loads(tool_calls_json)
                tool_calls = [ToolCall(**tc) for tc in raw]
            messages.append(
                Message(
                    role=Role(role_str),
                    content=content,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls,
                )
            )
        return messages

    async def add_to_session(self, session_id: str, message: Message) -> None:
        db = await self._get_db()
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps(
                [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                 for tc in message.tool_calls]
            )
        await db.execute(
            "INSERT INTO sessions (session_id, role, content, tool_call_id, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                message.role.value,
                message.content,
                message.tool_call_id,
                tool_calls_json,
                datetime.now().isoformat(),
            ),
        )
        await db.commit()

    async def clear(
        self,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> None:
        db = await self._get_db()
        if session_id:
            if before:
                await db.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND created_at < ?",
                    (session_id, before.isoformat()),
                )
            else:
                await db.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
        elif before:
            iso = before.isoformat()
            await db.execute("DELETE FROM memories WHERE created_at < ?", (iso,))
            await db.execute("DELETE FROM sessions WHERE created_at < ?", (iso,))
        else:
            await db.execute("DELETE FROM memories")
            await db.execute("DELETE FROM sessions")
        await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class DenseRetriever(Retriever):
    """Cosine-similarity ranking over :class:`SqliteStore` embeddings.

    The retriever borrows the store's embedder so query and stored
    embeddings match. Pass a different ``embedder`` to experiment with
    alternative embedding models without rebuilding the store — useful
    as a sweep axis.
    """

    def __init__(self, store: SqliteStore, embedder: Embedder | None = None):
        self._store = store
        self._embedder = embedder

    async def _embed(self, text: str) -> np.ndarray:
        if self._embedder is not None:
            return await self._embedder(text)
        return await self._store.embed(text)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        query_emb = await self._embed(query)
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return []

        # Optional metadata filter, passed via context for experimentation
        filter_ = (context or {}).get("filter")

        candidates = await self._store.iter_entries_with_embeddings()
        scored: list[tuple[float, MemoryEntry]] = []
        for entry, row_emb in candidates:
            if filter_:
                if not all(entry.metadata.get(key) == val for key, val in filter_.items()):
                    continue
            row_norm = float(np.linalg.norm(row_emb))
            if row_norm == 0:
                continue
            sim = float(np.dot(query_emb, row_emb) / (query_norm * row_norm))
            entry.score = sim
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:k]]


def LocalMemory(
    db_path: str = "jig_memory.db",
    embed_model: str = "nomic-embed-text",
    ollama_host: str | None = None,
) -> tuple[SqliteStore, DenseRetriever]:
    """Convenience factory: the canonical local-memory pair.

    Equivalent to::

        store = SqliteStore(db_path=..., embed_model=..., ollama_host=...)
        retriever = DenseRetriever(store)
        return store, retriever

    Use this for the common case; instantiate the pieces directly when
    you want to swap retrievers or share a store across retrievers.
    """
    store = SqliteStore(
        db_path=db_path,
        embed_model=embed_model,
        ollama_host=ollama_host,
    )
    retriever = DenseRetriever(store)
    return store, retriever
