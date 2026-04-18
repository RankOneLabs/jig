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
    """A smithers dispatch call failed.

    ``retryable=True`` signals a transient failure the caller may retry —
    primarily submission-time connection / network errors. Worker-side
    terminal outcomes (``failed``, ``cancelled``, HTTP 4xx, malformed
    response) stay non-retryable because retrying them just reproduces
    the same result. :class:`JobTimeoutError` is retryable by default
    since a hung worker can succeed on a fresh submission.
    """

    def __init__(
        self,
        message: str,
        *,
        job_id: str | None = None,
        status: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.job_id = job_id
        self.status = status
        self.retryable = retryable


class JobTimeoutError(DispatchError):
    """Dispatched job exceeded the configured timeout."""

    def __init__(self, message: str, *, job_id: str, timeout_seconds: int):
        super().__init__(
            message, job_id=job_id, status="timeout", retryable=True,
        )
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
        # Transient: dispatch server may be momentarily unreachable
        # (restart, network blip). Mark retryable so agent loops can
        # retry instead of terminating the run.
        raise DispatchError(
            f"Cannot reach dispatch server at {url}",
            retryable=True,
        ) from e
    except httpx.HTTPStatusError as e:
        # 4xx = terminal (bad submission), 5xx = transient (server
        # overloaded / misconfigured, worth a retry).
        retryable = e.response.status_code >= 500
        raise DispatchError(
            f"Dispatch submission failed: {e.response.status_code} {e.response.text}",
            retryable=retryable,
        ) from e
    except httpx.RequestError as e:
        # Timeouts, DNS failures, mid-request disconnects — all transient.
        raise DispatchError(
            f"Dispatch request error: {e}",
            retryable=True,
        ) from e

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
                ) from e
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
# asking each tool to manage its own transport. Callers that run
# multiple event loops in a process lifetime (tests calling
# ``asyncio.run`` repeatedly, long-running services that tear down and
# rebuild) should invoke :func:`aclose` at shutdown; otherwise the
# interpreter reaps the client at exit.
_shared_http: httpx.AsyncClient | None = None
_shared_http_loop: asyncio.AbstractEventLoop | None = None


def _get_shared_http() -> httpx.AsyncClient:
    """Return a lazily-created shared httpx client.

    Rebinds the client when the running event loop differs from the one
    the client was created on — an ``AsyncClient`` bound to a closed
    loop raises on use, so ``asyncio.run`` being called twice in the
    same process would otherwise break dispatch on the second call.
    """
    global _shared_http, _shared_http_loop
    loop = asyncio.get_running_loop()
    if _shared_http is not None and _shared_http_loop is not loop:
        # Loop changed out from under us. Detach the old client rather
        # than awaiting its close here — we're in a sync helper and the
        # old loop may already be closed. Let GC handle the transport.
        _shared_http = None
    if _shared_http is None:
        _shared_http = httpx.AsyncClient(timeout=30.0)
        _shared_http_loop = loop
    return _shared_http


async def aclose() -> None:
    """Close the shared httpx client used by :func:`run`.

    Safe to call multiple times and safe when nothing was ever
    dispatched. Tests should call this in teardown; long-running
    services should call it at shutdown. No-op in short-lived scripts
    (interpreter exit tears the client down anyway).
    """
    global _shared_http, _shared_http_loop
    client = _shared_http
    _shared_http = None
    _shared_http_loop = None
    if client is not None:
        await client.aclose()


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
    # Don't ``or {}`` here — that rewrites legitimate falsy returns
    # (``0``, ``False``, ``[]``, ``""``) into an empty dict and breaks
    # any dispatched function whose natural result is falsy.
    result = data.get("result")
    if isinstance(result, dict) and "value" in result:
        return result["value"]
    return result
