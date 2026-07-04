from __future__ import annotations

import logging
from datetime import datetime
from inspect import isawaitable
from typing import Any

from jig.core.types import MemoryEntry, MemoryStore, Message, Retriever, Role

try:
    from honcho import AsyncHoncho
except ImportError:
    AsyncHoncho = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class HonchoMemory(MemoryStore, Retriever):
    """Honcho-backed memory. Implements both ``MemoryStore`` and
    ``Retriever`` — the managed service owns retrieval, so the split
    collapses into one object. Instantiate once and pass to both
    ``store=`` and ``retriever=`` fields of :class:`AgentConfig`.
    """

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
        self._closed = False

    async def close(self) -> None:
        """Release the owned Honcho SDK client, if it exposes a close hook."""
        if self._closed:
            return

        aclose = getattr(self._client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if isawaitable(result):
                await result
            self._closed = True
            return

        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
        self._closed = True

    async def aclose(self) -> None:
        await self.close()

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

    async def get(self, id: str) -> MemoryEntry | None:
        try:
            doc = await self._client.apps.users.collections.documents.get(
                app_id=self._app_id,
                user_id=self._user_id,
                document_id=id,
            )
        except Exception:
            logger.warning("Honcho get failed (id=%s)", id, exc_info=True)
            return None
        return MemoryEntry(id=str(doc.id), content=doc.content, metadata=doc.metadata or {})

    async def all(self) -> list[MemoryEntry]:
        # Honcho doesn't expose a cheap "list everything" — return empty
        # rather than paging through the whole collection. Callers that
        # need iteration should use a MemoryStore backend that supports it.
        return []

    async def delete(self, id: str) -> None:
        try:
            await self._client.apps.users.collections.documents.delete(
                app_id=self._app_id,
                user_id=self._user_id,
                document_id=id,
            )
        except Exception:
            logger.warning("Honcho delete failed (id=%s)", id, exc_info=True)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        filter_ = (context or {}).get("filter")
        try:
            collection = await self._client.apps.users.collections.get_by_name(
                app_id=self._app_id,
                user_id=self._user_id,
                name=self._collection_name,
            )
            results = await self._client.apps.users.collections.documents.query(
                app_id=self._app_id,
                user_id=self._user_id,
                collection_id=collection.id,
                query=query,
                top_k=k,
                filter=filter_ or {},
            )
        except Exception:
            logger.warning(
                "Honcho query failed (collection_name=%r)",
                self._collection_name,
                exc_info=True,
            )
            return []

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
            logger.warning("Honcho session lookup failed (session_id=%s)", session_id, exc_info=True)
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
