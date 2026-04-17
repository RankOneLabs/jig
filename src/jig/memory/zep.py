from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from jig.core.types import AgentMemory, MemoryEntry, Message, Role

try:
    from zep_python.client import AsyncZep
except ImportError:
    AsyncZep = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

_ROLE_TO_ZEP = {Role.USER: "human", Role.ASSISTANT: "ai", Role.TOOL: "tool"}
_ZEP_TO_ROLE = {"human": Role.USER, "ai": Role.ASSISTANT, "tool": Role.TOOL}


class ZepMemory(AgentMemory):
    def __init__(self, session_id: str, **client_kwargs: Any):
        if AsyncZep is None:
            raise ImportError("Install zep: pip install 'jig[zep]'")
        self._client = AsyncZep(**client_kwargs)
        self._default_session_id = session_id

    async def _ensure_session(self, session_id: str) -> None:
        try:
            await self._client.memory.get(session_id)
        except Exception:
            await self._client.memory.add(session_id, messages=[])

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        entry_id = str(uuid.uuid4())
        await self._ensure_session(self._default_session_id)
        await self._client.memory.add(
            self._default_session_id,
            messages=[{"role": "system", "content": content, "metadata": metadata or {}}],
        )
        return entry_id

    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        sid = session_id or self._default_session_id
        try:
            results = await self._client.memory.search(sid, query, limit=limit)
        except Exception as e:
            logger.warning("Zep search failed (session_id=%s): %s", sid, e, exc_info=False)
            return []

        return [
            MemoryEntry(
                id=str(uuid.uuid4()),
                content=r.message.get("content", "") if isinstance(r.message, dict) else r.message.content,
                metadata=r.metadata or {},
                score=r.score,
            )
            for r in results
        ]

    async def get_session(self, session_id: str) -> list[Message]:
        try:
            memory = await self._client.memory.get(session_id)
        except Exception as e:
            logger.warning("Zep session lookup failed (session_id=%s): %s", session_id, e, exc_info=False)
            return []

        messages: list[Message] = []
        for m in memory.messages or []:
            role = _ZEP_TO_ROLE.get(m.role, Role.USER)
            messages.append(Message(role=role, content=m.content))
        return messages

    async def add_to_session(self, session_id: str, message: Message) -> None:
        await self._ensure_session(session_id)
        zep_role = _ROLE_TO_ZEP.get(message.role, "human")
        await self._client.memory.add(
            session_id,
            messages=[{"role": zep_role, "content": message.content}],
        )

    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        sid = session_id or self._default_session_id
        await self._client.memory.delete(sid)
