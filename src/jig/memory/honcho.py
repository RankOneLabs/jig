from __future__ import annotations

from datetime import datetime
from typing import Any

from jig.core.types import AgentMemory, MemoryEntry, Message, Role

try:
    from honcho import AsyncHoncho
except ImportError:
    AsyncHoncho = None  # type: ignore[assignment, misc]


class HonchoMemory(AgentMemory):
    def __init__(
        self,
        app_id: str,
        user_id: str,
        collection_name: str = "default",
        **client_kwargs: Any,
    ):
        if AsyncHoncho is None:
            raise ImportError("Install honcho: pip install 'jig[honcho]'")
        self._client = AsyncHoncho(**client_kwargs)
        self._app_id = app_id
        self._user_id = user_id
        self._collection_name = collection_name

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        collection = await self._client.apps.users.collections.get_or_create(
            app_id=self._app_id,
            user_id=self._user_id,
            name=self._collection_name,
        )
        doc = await self._client.apps.users.collections.documents.create(
            app_id=self._app_id,
            user_id=self._user_id,
            collection_id=collection.id,
            content=content,
            metadata=metadata or {},
        )
        return str(doc.id)

    async def query(
        self,
        query: str,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        try:
            collection = await self._client.apps.users.collections.get_by_name(
                app_id=self._app_id,
                user_id=self._user_id,
                name=self._collection_name,
            )
        except Exception:
            return []

        results = await self._client.apps.users.collections.documents.query(
            app_id=self._app_id,
            user_id=self._user_id,
            collection_id=collection.id,
            query=query,
            top_k=limit,
            filter=filter or {},
        )
        return [
            MemoryEntry(
                id=str(doc.id),
                content=doc.content,
                metadata=doc.metadata or {},
                score=1.0 - getattr(doc, "distance", 0.0),
            )
            for doc in results
        ]

    async def get_session(self, session_id: str) -> list[Message]:
        try:
            msgs = await self._client.apps.users.sessions.messages.list(
                app_id=self._app_id,
                user_id=self._user_id,
                session_id=session_id,
            )
        except Exception:
            return []

        return [
            Message(
                role=Role.USER if m.is_user else Role.ASSISTANT,
                content=m.content,
            )
            for m in msgs
        ]

    async def add_to_session(self, session_id: str, message: Message) -> None:
        await self._client.apps.users.sessions.messages.create(
            app_id=self._app_id,
            user_id=self._user_id,
            session_id=session_id,
            content=message.content,
            is_user=message.role == Role.USER,
        )

    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        if session_id:
            await self._client.apps.users.sessions.delete(
                app_id=self._app_id,
                user_id=self._user_id,
                session_id=session_id,
            )
