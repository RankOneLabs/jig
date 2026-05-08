"""Tests for jig._embed.ollama_embed response handling.

ollama-python >= 0.4 (the floor declared in pyproject's ``ollama`` extra)
returns a typed EmbedResponse pydantic model from AsyncClient.embed().
ollama_embed accesses ``.embeddings`` directly — older dict-shaped
responses are no longer supported.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from jig._embed import ollama_embed


@pytest.mark.asyncio
async def test_returns_float32_vector_from_embed_response():
    """ollama-python returns a typed EmbedResponse with .embeddings."""
    fake_response = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
    fake_client = AsyncMock()
    fake_client.embed = AsyncMock(return_value=fake_response)

    with patch("jig._embed.OllamaAsyncClient", return_value=fake_client):
        result = await ollama_embed("hello", model="nomic-embed-text")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.asyncio
async def test_raises_when_embeddings_empty():
    """If the response's embeddings list is empty, surface a clear error."""
    fake_response = SimpleNamespace(embeddings=[])
    fake_client = AsyncMock()
    fake_client.embed = AsyncMock(return_value=fake_response)

    with patch("jig._embed.OllamaAsyncClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="missing 'embeddings'"):
            await ollama_embed("hello", model="nomic-embed-text")
