from __future__ import annotations

import logging

import numpy as np

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    OllamaAsyncClient = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


async def ollama_embed(
    text: str,
    model: str = "nomic-embed-text",
    host: str | None = None,
) -> np.ndarray:
    """Embed `text` via a local Ollama instance and return a float32 vector.

    Shared by LocalMemory and SQLiteFeedbackLoop so a single place owns the
    client construction + response shape. Requires ollama-python >= 0.4
    (the pyproject ``ollama`` extra pins this floor); the response is a
    typed ``EmbedResponse`` pydantic model.
    """
    if OllamaAsyncClient is None:
        raise ImportError("Install ollama: pip install 'jig[ollama]'")
    client = OllamaAsyncClient(host=host)
    primary_exc: BaseException | None = None
    try:
        response = await client.embed(model=model, input=text)
        embeddings = response.embeddings
        if not embeddings:
            raise RuntimeError(f"Ollama embed response missing 'embeddings' (model={model})")
        return np.array(embeddings[0], dtype=np.float32)
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        inner = getattr(client, "_client", None)
        if inner is not None and hasattr(inner, "aclose"):
            try:
                await inner.aclose()
            except Exception:
                if primary_exc is None:
                    raise
                logger.warning("Ollama embed client cleanup failed", exc_info=True)
