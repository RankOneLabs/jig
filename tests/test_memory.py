"""Tests for the MemoryStore / Retriever split (phase 6)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

import numpy as np

from jig import MemoryStore, Retriever
from jig.core.types import MemoryEntry
from jig.memory.local import DenseRetriever, LocalMemory, SqliteStore


async def _fake_embed(text: str) -> np.ndarray:
    """Deterministic per-text embedding — no Ollama required."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(128, dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(db_path=str(tmp_path / "m.db"))
    s._custom_embedder = _fake_embed
    return s


@pytest.fixture
def retriever(store):
    return DenseRetriever(store)


@pytest.mark.asyncio
class TestSqliteStore:
    async def test_add_and_get(self, store):
        entry_id = await store.add("hello", {"tag": "a"})
        got = await store.get(entry_id)
        assert got is not None
        assert got.content == "hello"
        assert got.metadata == {"tag": "a"}

    async def test_get_missing_returns_none(self, store):
        assert await store.get("does-not-exist") is None

    async def test_all_returns_everything_recent_first(self, store):
        await store.add("first", {})
        await store.add("second", {})
        entries = await store.all()
        assert [e.content for e in entries] == ["second", "first"]

    async def test_delete(self, store):
        eid = await store.add("delete me", {})
        await store.delete(eid)
        assert await store.get(eid) is None

    async def test_session_append_and_readback(self, store):
        from jig.core.types import Message, Role

        await store.add_to_session("s1", Message(role=Role.USER, content="hi"))
        await store.add_to_session("s1", Message(role=Role.ASSISTANT, content="hey"))
        msgs = await store.get_session("s1")
        assert [m.content for m in msgs] == ["hi", "hey"]
        assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]

    async def test_clear_session(self, store):
        from jig.core.types import Message, Role

        await store.add_to_session("s1", Message(role=Role.USER, content="x"))
        await store.clear(session_id="s1")
        assert await store.get_session("s1") == []


@pytest.mark.asyncio
class TestDenseRetriever:
    async def test_retrieves_by_similarity(self, store, retriever):
        # Seed with three entries; one uses identical embedder hash as the query.
        await store.add("alpha", {})
        await store.add("beta", {})
        await store.add("gamma", {})

        results = await retriever.retrieve("alpha", k=3)
        assert len(results) == 3
        # "alpha" query should rank "alpha" entry first (identical embedding)
        assert results[0].content == "alpha"
        # Scores are monotonic
        assert results[0].score >= results[1].score >= results[2].score

    async def test_k_caps_results(self, store, retriever):
        for i in range(5):
            await store.add(f"item-{i}", {})
        out = await retriever.retrieve("query", k=2)
        assert len(out) == 2

    async def test_empty_store_empty_results(self, store, retriever):
        assert await retriever.retrieve("anything") == []

    async def test_filter_via_context(self, store, retriever):
        await store.add("keep", {"kind": "a"})
        await store.add("drop", {"kind": "b"})
        await store.add("also_keep", {"kind": "a"})

        results = await retriever.retrieve(
            "query", k=5, context={"filter": {"kind": "a"}}
        )
        contents = {r.content for r in results}
        assert contents == {"keep", "also_keep"}

    async def test_custom_embedder_overrides_store(self, store):
        """Passing a different embedder to DenseRetriever lets you sweep
        retrieval-embedding choice without rebuilding the corpus."""
        await store.add("content", {})

        async def weird_embedder(text: str) -> np.ndarray:
            # Orthogonal to stored embeddings → all similarities ~0
            rng = np.random.default_rng(42)
            return rng.standard_normal(128).astype(np.float32)

        alt = DenseRetriever(store, embedder=weird_embedder)
        results = await alt.retrieve("content", k=1)
        # Returns results but with different scores than the default
        assert len(results) == 1


@pytest.mark.asyncio
class TestLocalMemoryFactory:
    async def test_returns_both_pieces(self, tmp_path):
        store, retriever = LocalMemory(db_path=str(tmp_path / "lm.db"))
        assert isinstance(store, MemoryStore)
        assert isinstance(retriever, Retriever)
        # They share backing — entries added via store are visible to the
        # retriever (when they have embeddings)
        retriever._embedder = _fake_embed  # type: ignore[method-assign]
        store._custom_embedder = _fake_embed  # type: ignore[method-assign]
        await store.add("hello", {})
        assert await retriever.retrieve("hello", k=1)


class TestProtocolConformance:
    """Concrete classes must satisfy the abstract protocol."""

    def test_sqlite_store_is_memory_store(self, tmp_path):
        s = SqliteStore(db_path=str(tmp_path / "conformance.db"))
        assert isinstance(s, MemoryStore)

    def test_dense_retriever_is_retriever(self, tmp_path):
        s = SqliteStore(db_path=str(tmp_path / "conformance2.db"))
        r = DenseRetriever(s)
        assert isinstance(r, Retriever)
