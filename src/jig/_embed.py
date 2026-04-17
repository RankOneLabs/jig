from __future__ import annotations

import numpy as np

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    OllamaAsyncClient = None  # type: ignore[assignment, misc]


async def ollama_embed(
    text: str,
    model: str = "nomic-embed-text",
    host: str | None = None,
) -> np.ndarray:
    """Embed `text` via a local Ollama instance and return a float32 vector.

    Shared by LocalMemory and SQLiteFeedbackLoop so a single place owns the
    client construction + response shape.
    """
    if OllamaAsyncClient is None:
        raise ImportError("Install ollama: pip install 'jig[ollama]'")
    client = OllamaAsyncClient(host=host)
    response = await client.embed(model=model, input=text)
    return np.array(response["embeddings"][0], dtype=np.float32)
