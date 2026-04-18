"""Submit-then-poll primitive, shared by LLM and function dispatch paths.

``_submit_and_poll`` is internal — callers go through :func:`run` for
function dispatch or :class:`jig.llm.DispatchClient` for inference.
Unifying the polling loop, exponential backoff, and error mapping means
fixes to one path benefit both.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from jig.core.errors import JigError

logger = logging.getLogger(__name__)

# Smithers job statuses that mean "still working"
_PENDING_STATUSES = frozenset({"queued", "waking_machine", "dispatched", "running"})


class DispatchError(JigError):
    """A smithers dispatch call failed terminally (non-retryable)."""

    def __init__(
        self,
        message: str,
        *,
        job_id: str | None = None,
        status: str | None = None,
    ):
        super().__init__(message)
        self.job_id = job_id
        self.status = status


class JobTimeoutError(DispatchError):
    """Dispatched job exceeded the configured timeout."""

    def __init__(self, message: str, *, job_id: str, timeout_seconds: int):
        super().__init__(message, job_id=job_id, status="timeout")
        self.timeout_seconds = timeout_seconds


@dataclass
class _PollConfig:
    """Knobs for the polling loop — tests override these to run fast."""

    timeout_seconds: int = 300
    poll_interval: float = 0.5
    poll_max_interval: float = 5.0


async def _submit_and_poll(
    *,
    http: httpx.AsyncClient,
    dispatch_url: str,
    task_type: str,
    payload: dict[str, Any],
    requester: str = "jig",
    model: str | None = None,
    machine: str | None = None,
    trace_context: dict[str, Any] | None = None,
    poll_config: _PollConfig | None = None,
) -> dict[str, Any]:
    """Submit a job to smithers, poll until terminal, return the job data.

    Returns the full job dict from smithers on ``status == "complete"``.
    Raises :class:`DispatchError` / :class:`JobTimeoutError` on terminal
    failure. Transient polling errors (connection drops, malformed JSON)
    are logged and retried within the timeout window.
    """
    cfg = poll_config or _PollConfig()
    url = dispatch_url.rstrip("/")
    start = time.time()

    submission: dict[str, Any] = {
        "task_type": task_type,
        "payload": payload,
        "requester": requester,
        "priority": "normal",
        "timeout_seconds": cfg.timeout_seconds,
    }
    if model is not None:
        submission["model"] = model
    if machine is not None:
        submission["machine"] = machine
    if trace_context is not None:
        # Phase 9 will have workers read this and reparent their spans.
        # For now we just propagate the fields through the payload so the
        # protocol is established.
        submission["trace_context"] = trace_context

    # --- Submit ---
    try:
        resp = await http.post(f"{url}/jobs", json=submission)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise DispatchError(
            f"Cannot reach dispatch server at {url}",
        ) from e
    except httpx.HTTPStatusError as e:
        raise DispatchError(
            f"Dispatch submission failed: {e.response.status_code} {e.response.text}",
        ) from e
    except httpx.RequestError as e:
        raise DispatchError(f"Dispatch request error: {e}") from e

    try:
        job_id = resp.json()["job_id"]
    except (ValueError, KeyError) as e:
        raise DispatchError(
            f"Unexpected dispatch response: {resp.text}",
        ) from e

    logger.info("Dispatch job %s submitted (task_type=%s)", job_id, task_type)

    # --- Poll ---
    interval = cfg.poll_interval
    while True:
        remaining = cfg.timeout_seconds - (time.time() - start)
        if remaining <= 0:
            raise JobTimeoutError(
                f"Dispatch job {job_id} timed out after {cfg.timeout_seconds}s",
                job_id=job_id,
                timeout_seconds=cfg.timeout_seconds,
            )

        await asyncio.sleep(min(interval, remaining))
        interval = min(interval * 2, cfg.poll_max_interval)

        try:
            poll = await http.get(f"{url}/jobs/{job_id}")
            poll.raise_for_status()
        except httpx.ConnectError:
            logger.warning("Lost connection polling job %s, retrying...", job_id)
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise DispatchError(
                    f"Dispatch job {job_id} not found (expired or invalid)",
                    job_id=job_id,
                    status="not_found",
                )
            logger.warning(
                "HTTP %s polling job %s, retrying...",
                e.response.status_code, job_id,
            )
            continue
        except httpx.RequestError:
            logger.warning("Request error polling job %s, retrying...", job_id)
            continue

        try:
            data = poll.json()
        except ValueError:
            logger.warning("Malformed JSON polling job %s, retrying...", job_id)
            continue

        if not isinstance(data, dict):
            logger.warning("Non-object JSON polling job %s, retrying...", job_id)
            continue

        status = data.get("status", "")
        if status in _PENDING_STATUSES:
            continue

        if status == "complete":
            logger.info(
                "Dispatch job %s complete (%.0fms)",
                job_id, (time.time() - start) * 1000,
            )
            return data

        if status == "failed":
            raise DispatchError(
                data.get("error") or f"Dispatch job {job_id} failed",
                job_id=job_id,
                status="failed",
            )

        if status == "cancelled":
            raise DispatchError(
                f"Dispatch job {job_id} was cancelled",
                job_id=job_id,
                status="cancelled",
            )

        logger.warning("Unexpected status %r for job %s, continuing poll", status, job_id)


# Shared httpx client for one-off run() calls. Most callers should use
# an LLMClient-style instance with its own lifecycle; this exists so
# tool-level dispatch (Tool(dispatch=True)) can fire off calls without
# asking each tool to manage its own transport.
_shared_http: httpx.AsyncClient | None = None


def _get_shared_http() -> httpx.AsyncClient:
    global _shared_http
    if _shared_http is None:
        _shared_http = httpx.AsyncClient(timeout=30.0)
    return _shared_http


async def run(
    fn_ref: str,
    payload: dict[str, Any] | None = None,
    *,
    dispatch_url: str = "http://willie:8900",
    requester: str = "jig",
    machine: str | None = None,
    trace_context: dict[str, Any] | None = None,
    timeout_seconds: int = 300,
    poll_interval: float = 0.5,
    poll_max_interval: float = 5.0,
    http: httpx.AsyncClient | None = None,
) -> Any:
    """Execute ``fn_ref`` on a smithers worker, await the result.

    ``fn_ref`` is the ``"package.module:function"`` identifier the
    worker's function registry knows (populated via the
    ``jig.smithers_fn`` entry-point group). ``payload`` becomes the
    function's kwargs.

    Returns whatever the worker put in ``job.result["value"]``. Raises
    :class:`DispatchError` on failure, :class:`JobTimeoutError` on
    timeout. Use this for deterministic steps you want offloaded —
    backtests, embeddings, reindexes — while LLM calls go through
    :class:`jig.llm.DispatchClient`.
    """
    transport = http or _get_shared_http()
    data = await _submit_and_poll(
        http=transport,
        dispatch_url=dispatch_url,
        task_type="function",
        payload={"fn_ref": fn_ref, "args": payload or {}},
        requester=requester,
        machine=machine,
        trace_context=trace_context,
        poll_config=_PollConfig(
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            poll_max_interval=poll_max_interval,
        ),
    )
    result = data.get("result") or {}
    if isinstance(result, dict) and "value" in result:
        return result["value"]
    return result
