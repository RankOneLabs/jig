from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import aiosqlite
import numpy as np

from jig.core.types import AgentMemory, MemoryEntry, Message, Role, ToolCall

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    OllamaAsyncClient = None  # type: ignore[assignment, misc]

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


class LocalMemory(AgentMemory):
    def __init__(
        self,
        db_path: str = "jig_memory.db",
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
        if OllamaAsyncClient is None:
            raise ImportError("Install ollama: pip install 'jig[ollama]'")
        client = OllamaAsyncClient(host=self._ollama_host)
        response = await client.embed(model=self._embed_model, input=text)
        return np.array(response["embeddings"][0], dtype=np.float32)

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        db = await self._get_db()
        entry_id = str(uuid.uuid4())
        embedding = await self._embed(content)
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

    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        db = await self._get_db()
        query_emb = await self._embed(query)
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []

        cursor = await db.execute("SELECT id, content, metadata, embedding, created_at FROM memories")
        rows = await cursor.fetchall()

        scored: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            row_id, content, meta_json, emb_bytes, created_str = row
            meta = json.loads(meta_json)

            if filter:
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            row_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            row_norm = np.linalg.norm(row_emb)
            if row_norm == 0:
                continue

            similarity = float(np.dot(query_emb, row_emb) / (query_norm * row_norm))
            entry = MemoryEntry(
                id=row_id,
                content=content,
                metadata=meta,
                score=similarity,
                created_at=datetime.fromisoformat(created_str),
            )
            scored.append((similarity, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    async def get_session(self, session_id: str) -> list[Message]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM sessions WHERE session_id = ? ORDER BY created_at ASC",
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
                [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in message.tool_calls]
            )
        await db.execute(
            "INSERT INTO sessions (session_id, role, content, tool_call_id, tool_calls, created_at) VALUES (?, ?, ?, ?, ?, ?)",
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

    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        db = await self._get_db()
        if session_id:
            if before:
                await db.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND created_at < ?",
                    (session_id, before.isoformat()),
                )
            else:
                await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        elif before:
            await db.execute("DELETE FROM memories WHERE created_at < ?", (before.isoformat(),))
            await db.execute("DELETE FROM sessions WHERE created_at < ?", (before.isoformat(),))
        else:
            await db.execute("DELETE FROM memories")
            await db.execute("DELETE FROM sessions")
        await db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
