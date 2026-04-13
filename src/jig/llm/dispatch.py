"""Smithers dispatch adapter — routes inference to the Springfield homelab fleet."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from jig.core.errors import JigLLMError
from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Role,
    Usage,
)

logger = logging.getLogger(__name__)

# Smithers job statuses that mean "still working"
_PENDING_STATUSES = frozenset({"queued", "waking_machine", "dispatched", "running"})


class DispatchClient(LLMClient):
    """jig LLMClient that submits inference jobs to a smithers dispatch server.

    Three levels of specificity:
        DispatchClient()                              # router picks model + machine
        DispatchClient(model="llama-70b")             # router picks machine
        DispatchClient(model="llama-70b", machine="mcbain")  # explicit
    """

    def __init__(
        self,
        model: str | None = None,
        machine: str | None = None,
        dispatch_url: str = "http://willie:8900",
        requester: str = "jig",
        timeout_seconds: int = 300,
        poll_interval: float = 0.5,
        poll_max_interval: float = 5.0,
    ) -> None:
        self._model = model
        self._machine = machine
        self._dispatch_url = dispatch_url.rstrip("/")
        self._requester = requester
        self._timeout_seconds = timeout_seconds
        self._poll_interval = poll_interval
        self._poll_max_interval = poll_max_interval
        self._http = httpx.AsyncClient(timeout=30.0)

    def _build_payload(self, params: CompletionParams) -> dict[str, Any]:
        """Convert CompletionParams to smithers executor payload format."""
        if params.tools:
            raise JigLLMError(
                "Tool use is not supported through smithers dispatch",
                "dispatch",
            )

        messages: list[dict[str, str]] = []
        if params.system:
            messages.append({"role": "system", "content": params.system})
        for msg in params.messages:
            if msg.role == Role.SYSTEM:
                continue
            messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict[str, Any] = {"messages": messages}
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        return payload

    async def complete(self, params: CompletionParams) -> LLMResponse:
        payload = self._build_payload(params)
        start = time.time()

        # Submit job
        submission = {
            "task_type": "inference",
            "payload": payload,
            "requester": self._requester,
            "priority": "normal",
            "timeout_seconds": self._timeout_seconds,
        }
        if self._model is not None:
            submission["model"] = self._model
        if self._machine is not None:
            submission["machine"] = self._machine

        try:
            resp = await self._http.post(f"{self._dispatch_url}/jobs", json=submission)
            resp.raise_for_status()
        except httpx.ConnectError:
            raise JigLLMError(
                f"Cannot reach dispatch server at {self._dispatch_url}",
                "dispatch",
                retryable=True,
            )
        except httpx.HTTPStatusError as e:
            raise JigLLMError(
                f"Dispatch submission failed: {e.response.status_code} {e.response.text}",
                "dispatch",
            )
        except httpx.RequestError as e:
            raise JigLLMError(
                f"Dispatch request error: {e}",
                "dispatch",
                retryable=True,
            )

        try:
            job_id = resp.json()["job_id"]
        except (ValueError, KeyError) as e:
            raise JigLLMError(
                f"Unexpected dispatch response: {resp.text}",
                "dispatch",
            )

        logger.info(f"Dispatch job {job_id} submitted (model={self._model})")

        # Poll for completion
        interval = self._poll_interval
        while True:
            remaining = self._timeout_seconds - (time.time() - start)
            if remaining <= 0:
                raise JigLLMError(
                    f"Dispatch job {job_id} timed out after {self._timeout_seconds}s",
                    "dispatch",
                    retryable=True,
                )

            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * 2, self._poll_max_interval)

            try:
                poll = await self._http.get(f"{self._dispatch_url}/jobs/{job_id}")
                poll.raise_for_status()
            except httpx.ConnectError:
                logger.warning(f"Lost connection polling job {job_id}, retrying...")
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise JigLLMError(
                        f"Dispatch job {job_id} not found (expired or invalid)",
                        "dispatch",
                    )
                logger.warning(f"HTTP {e.response.status_code} polling job {job_id}, retrying...")
                continue
            except httpx.RequestError:
                logger.warning(f"Request error polling job {job_id}, retrying...")
                continue

            try:
                data = poll.json()
            except ValueError:
                logger.warning(f"Malformed JSON polling job {job_id}, retrying...")
                continue
            status = data.get("status", "") if isinstance(data, dict) else ""

            if status in _PENDING_STATUSES:
                continue

            latency_ms = (time.time() - start) * 1000

            if status == "complete":
                result = data.get("result") or {}
                content = result.get("content", "")
                model = data.get("model") or self._model or "dispatch"
                logger.info(f"Dispatch job {job_id} complete ({latency_ms:.0f}ms)")
                return LLMResponse(
                    content=content,
                    tool_calls=None,
                    usage=Usage(
                        input_tokens=0,
                        output_tokens=0,
                        cost=None,
                    ),
                    latency_ms=latency_ms,
                    model=model,
                )

            if status == "failed":
                error = data.get("error") or "Unknown dispatch error"
                raise JigLLMError(error, "dispatch")

            if status == "cancelled":
                raise JigLLMError("Dispatch job was cancelled", "dispatch")

            # Unknown status — keep polling
            logger.warning(f"Unexpected job status: {status}")
