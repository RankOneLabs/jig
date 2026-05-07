"""Tests for jig._embed.ollama_embed response-shape compatibility.

ollama-python >= 0.4 returns an EmbedResponse pydantic model from
AsyncClient.embed(); older releases returned a dict. ollama_embed must
handle both, otherwise LocalMemory and SQLiteFeedbackLoop break the
moment a downstream installs a current ollama client.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from jig._embed import ollama_embed


@pytest.mark.asyncio
async def test_handles_pydantic_embed_response():
    """Modern ollama-python returns a typed EmbedResponse with .embeddings."""
    fake_response = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
    fake_client = AsyncMock()
    fake_client.embed = AsyncMock(return_value=fake_response)

    with patch("jig._embed.OllamaAsyncClient", return_value=fake_client):
        result = await ollama_embed("hello", model="nomic-embed-text")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.asyncio
async def test_handles_legacy_dict_response():
    """Older ollama-python returned a dict — keep it working."""
    fake_response = {"embeddings": [[0.4, 0.5, 0.6]]}
    fake_client = AsyncMock()
    fake_client.embed = AsyncMock(return_value=fake_response)

    with patch("jig._embed.OllamaAsyncClient", return_value=fake_client):
        result = await ollama_embed("hello", model="nomic-embed-text")

    np.testing.assert_array_equal(result, np.array([0.4, 0.5, 0.6], dtype=np.float32))


@pytest.mark.asyncio
async def test_raises_when_embeddings_missing():
    """If the response has neither attr nor key, surface a clear error."""
    fake_response = SimpleNamespace(other_field="x")
    fake_client = AsyncMock()
    fake_client.embed = AsyncMock(return_value=fake_response)

    with patch("jig._embed.OllamaAsyncClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="missing 'embeddings'"):
            await ollama_embed("hello", model="nomic-embed-text")
