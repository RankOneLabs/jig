"""OpenRouter adapter — OpenAI-compatible chat completions via openrouter.ai.

OpenRouter exposes hundreds of models (Anthropic, OpenAI, Google, Meta,
Mistral, ...) behind one OpenAI-compatible endpoint. Auth is a single
``OPENROUTER_API_KEY``; model names are vendor-namespaced slugs like
``anthropic/claude-3.5-sonnet``.

Cost: rather than maintain a per-slug pricing table that mirrors
OpenRouter's catalog, we ask OpenRouter to include the exact charge in
the response (``extra_body={"usage": {"include": True}}``) and read it
off ``response.usage.cost``. The local ``pricing.py`` table is unused
for OpenRouter slugs.

App attribution: pass ``http_referer`` / ``x_title`` to populate the
``HTTP-Referer`` and ``X-Title`` headers OpenRouter uses on its public
rankings page (https://openrouter.ai/rankings). Both optional.
"""
from __future__ import annotations

import os
from typing import Any

from jig.llm.openai import OpenAIClient


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAIClient):
    _provider_label = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter requires an API key. Set OPENROUTER_API_KEY or "
                "pass api_key= to OpenRouterClient."
            )
        default_headers = dict(client_kwargs.pop("default_headers", {}) or {})
        if http_referer:
            default_headers.setdefault("HTTP-Referer", http_referer)
        if x_title:
            default_headers.setdefault("X-Title", x_title)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
            **client_kwargs,
        )

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"extra_body": {"usage": {"include": True}}}

    def _inline_cost(self, response: Any) -> float | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cost = getattr(usage, "cost", None)
        if cost is None:
            extra = getattr(usage, "model_extra", None) or {}
            cost = extra.get("cost")
        if cost is None:
            return None
        try:
            return float(cost)
        except (TypeError, ValueError):
            return None
