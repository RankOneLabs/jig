"""The HTTP boundary for TypeSafe Jev evaluations."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Sequence
from uuid import uuid4

import httpx

from jig.jev.errors import JevError, JevErrorKind
from jig.jev.models import JevJson, JevQuestion, JevResult
from jig.jev.wire import build_request_body, parse_response

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_BACKOFF_SECONDS = 0.1


class JevClient:
    """Evaluate Jev questions over HTTP.

    A missing API key is a caller error and raises ValueError before any IO.
    Injected HTTP clients remain owned by their caller.
    """

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
        max_retries: int = 0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("Jev requires an API key. Set TYPESAFE_API_KEY or pass api_key=.")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.model = model
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self._owns_http = http is None
        self._http = http if http is not None else httpx.AsyncClient()

    async def evaluate(
        self, state: JevJson, questions: Sequence[JevQuestion]
    ) -> JevResult:
        """Submit one logical call, including any bounded HTTP retries."""
        body = build_request_body(state, questions, self.model)
        call_id = uuid4().hex
        started = time.monotonic()
        deadline = started + self.timeout
        attempts = 0

        def elapsed_ms() -> float:
            return (time.monotonic() - started) * 1000

        def failure(kind: JevErrorKind, status: int | None = None) -> JevError:
            # Never interpolate response content, request data, URL, or an
            # httpx exception: each can contain credentials or post content.
            detail = f"{kind} (HTTP {status})" if status is not None else kind
            return JevError(kind, status, call_id, None, detail, attempts, elapsed_ms())

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise failure("timeout")
            attempts += 1
            try:
                async with asyncio.timeout(remaining):
                    response = await self._http.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                        timeout=remaining,
                    )
            except (httpx.TimeoutException, TimeoutError):
                raise failure("timeout") from None
            except httpx.HTTPError:
                raise failure("transport") from None

            status = response.status_code
            if status != 200:
                status_kinds: dict[int, JevErrorKind] = {
                    401: "auth", 422: "invalid_request", 429: "rate_limited",
                    529: "overloaded",
                }
                kind = status_kinds.get(status, "invalid_response")
                if kind in ("rate_limited", "overloaded") and attempts <= self.max_retries:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        delay = min(_BACKOFF_SECONDS * (2 ** (attempts - 1)), remaining)
                        await asyncio.sleep(delay)
                        continue
                raise failure(kind, status)

            try:
                payload = response.json()
            except ValueError:
                raise failure("invalid_response", status) from None
            try:
                return parse_response(payload, call_id, elapsed_ms(), attempts, questions)
            except JevError:
                # The protocol validator may include provider supplied IDs
                # in its diagnostic. Keep that data out of exception text.
                raise failure("invalid_response", status) from None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> JevClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()
