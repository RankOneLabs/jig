"""Submit-then-poll primitive, shared by LLM and function dispatch paths.

``_submit_and_poll`` is internal — callers go through :func:`run` for
function dispatch or :class:`jig.llm.DispatchClient` for inference.
Unifying the polling loop, exponential backoff, and error mapping means
fixes to one path benefit both.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import httpx

from jig.core.errors import JigError

logger = logging.getLogger(__name__)

# Smithers job statuses that mean "still working"
_PENDING_STATUSES = frozenset({"queued", "waking_machine", "dispatched", "running"})

# Dispatch is a deployment-specific backend, so the endpoint is configured
# rather than hardcoded: pass ``dispatch_url=`` explicitly, set the
# ``JIG_DISPATCH_URL`` environment variable, or fall back to a local server.
_DISPATCH_URL_ENV = "JIG_DISPATCH_URL"
_DEFAULT_DISPATCH_URL = "http://localhost:8900"


def default_dispatch_url() -> str:
    """Resolve the dispatch URL from ``JIG_DISPATCH_URL`` or the local default."""
    return os.getenv(_DISPATCH_URL_ENV) or _DEFAULT_DISPATCH_URL


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


class DispatchBusinessError(JigError):
    """A dispatched function ran to completion and reported its own
    domain-level failure via the reserved ``__jig_tool_error__`` result
    key (see :func:`jig.dispatch.tool_error`).

    Deliberately **not** a :class:`DispatchError` subclass: ``DispatchError``
    and :class:`JobTimeoutError` mean the job itself never produced a
    result (submission failed, the worker crashed, the poll timed out).
    This means the opposite — the job returned normally and the *business
    outcome* it reports is a failure. ``ToolRegistry`` synthesizes this
    exception so both kinds can flow through the same
    ``Tool.on_dispatch_error`` hook (a single reconciliation
    implementation can handle both), while
    ``isinstance(error, DispatchBusinessError)`` lets a consumer branch on
    which one actually happened — exactly the distinction gecko's attempt
    reconciliation needs between "the study never ran" and "the study ran
    and failed".
    """

    def __init__(self, message: str, *, payload: dict[str, Any]):
        super().__init__(message)
        # The sibling fields the worker attached alongside the reserved
        # key (e.g. a partial summary) — same dict ``ToolResult.output``
        # is built from, so a consumer that wants more than the message
        # doesn't have to re-parse JSON.
        self.payload = payload


@dataclass
class _PollConfig:
    """Knobs for the polling loop — tests override these to run fast."""

    timeout_seconds: int = 300
    cleanup_grace_seconds: float = 10.0
    cancel_on_timeout: bool = True
    poll_interval: float = 0.5
    poll_max_interval: float = 5.0


# Cancellation is the path a caller takes when it has already given up, so it
# gets an explicit bound rather than inheriting whatever the caller's client
# was built with — a client with ``timeout=None`` would otherwise let a stalled
# cancellation hold the caller's exception hostage indefinitely.
_CANCEL_TIMEOUT_SECONDS = 10.0

# What smithers said when asked to cancel a job by id. "gone" covers a 404:
# smithers has no such job, so there is nothing left to stop. "already_terminal"
# is a 409 — the job stopped on its own, and *how* it stopped is a separate
# question this call does not answer.
_DeleteOutcome = Literal["accepted", "gone", "already_terminal"]


async def _send(
    request: Callable[..., Awaitable[httpx.Response]],
    url: str,
    *,
    what: str,
    job_id: str | None = None,
    error_statuses: Mapping[int, str] | None = None,
    passthrough: frozenset[int] = frozenset(),
    timeout: float | None = None,
) -> httpx.Response:
    """Send one smithers request, mapping HTTP and transport failures.

    ``what`` names the request for error messages ("Dispatch job j-1
    cancellation"). Status codes in ``passthrough`` come back as responses
    rather than raising, for callers that branch on them; any other non-2xx
    raises a :class:`DispatchError` whose ``status`` is looked up in
    ``error_statuses`` and which is retryable on 5xx. ``timeout`` is only
    passed when set, so an unbounded read keeps the client's own default.
    """
    kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
    try:
        response = await request(url, **kwargs)
        if response.status_code not in passthrough:
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        raise DispatchError(
            f"{what} failed: {code} {exc.response.text}",
            job_id=job_id,
            status=(error_statuses or {}).get(code),
            retryable=code >= 500,
        ) from exc
    except httpx.RequestError as exc:
        raise DispatchError(
            f"{what} failed: {exc}", job_id=job_id, retryable=True,
        ) from exc
    return response


def _json_object(
    response: httpx.Response,
    *,
    what: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Decode a smithers response body that must be a JSON object."""
    try:
        data = response.json()
    except ValueError as exc:
        raise DispatchError(
            f"{what} returned malformed JSON", job_id=job_id,
        ) from exc
    if not isinstance(data, dict):
        raise DispatchError(
            f"{what} returned a non-object response", job_id=job_id,
        )
    return data


async def _delete_remote_job(
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
) -> _DeleteOutcome:
    """Cancel one job by id, raising when the cancellation is unacknowledged.

    Smithers waits for its worker cancellation before marking the job
    cancelled, so awaiting this endpoint also creates the ordering guarantee
    callers need before they submit a retry. A terminal or missing job (409 /
    404) means no work survives this call, which is all a caller that has
    given up needs to know; callers that must distinguish *cancelled* from
    *ran to completion* branch on the returned outcome. Anything else is
    reported as a :class:`DispatchError` so the caller can decide whether an
    unacknowledged cancellation matters on its path.

    The body of an accepted cancellation is not read: the status code is the
    acknowledgement, and a malformed body must not turn a cancellation that
    happened into one reported as failed.
    """
    response = await _send(
        http.delete,
        f"{url}/jobs/{job_id}",
        what=f"Dispatch job {job_id} cancellation",
        job_id=job_id,
        passthrough=frozenset({404, 409}),
        timeout=_CANCEL_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return "gone"
    if response.status_code == 409:
        return "already_terminal"
    return "accepted"


async def _cancel_remote_job(
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
) -> None:
    """Best-effort cancellation for a request/response caller that gave up.

    Failures are logged rather than raised because on the timeout and
    caller-cancellation paths the cancellation must not replace the original
    timeout/cancellation exception. Paths where an unacknowledged
    cancellation is itself a correctness problem use
    :func:`_fence_rejected_submission` instead.
    """
    try:
        await _delete_remote_job(http, url, job_id)
    except Exception as exc:
        logger.warning("Could not cancel dispatch job %s: %s", job_id, exc)


async def _cancel_remote_job_drained(
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
) -> asyncio.CancelledError | None:
    """Run :func:`_cancel_remote_job` to completion, even if cancelled.

    Returns the cancellation that arrived meanwhile, if any, for the caller
    to re-raise once the cancellation has landed — see :func:`_drain`.
    """
    return await _drain(
        asyncio.create_task(_cancel_remote_job(http, url, job_id)),
    )


# A rejected registration is rare and the retry is cheap, so spend a few
# attempts on a retryable cancellation before giving up and reporting it.
_FENCE_ATTEMPTS = 3
_FENCE_RETRY_SECONDS = 0.5


@dataclass(frozen=True)
class _FenceCancelled:
    """Smithers confirmed the job stopped before producing an outcome."""

    kind: Literal["cancelled"] = "cancelled"


@dataclass(frozen=True)
class _FenceRanToTerminal:
    """The job reached a terminal state of its own before cancellation landed.

    Cancelling a job that already finished is a no-op on its *effects*: the
    worker ran, and whatever it wrote stays written. Reporting that as a clean
    record-or-cancel rejection would tell the caller the opposite of the truth,
    so it is its own outcome. ``status`` is the smithers status when it could
    be read (``complete``, ``failed``) and ``None`` when only the fact of
    terminality is known.
    """

    status: str | None
    kind: Literal["ran_to_terminal"] = "ran_to_terminal"


@dataclass(frozen=True)
class _FenceUnacknowledged:
    """Smithers never confirmed the cancellation; the job may still run."""

    error: DispatchError
    kind: Literal["unacknowledged"] = "unacknowledged"


_FenceOutcome = _FenceCancelled | _FenceRanToTerminal | _FenceUnacknowledged

# Smithers statuses that mean cancellation did its job. Anything else terminal
# means the worker got there first.
_CANCELLED_STATUSES = frozenset({"cancelled", "cancelling"})


def _fence_outcome_for_status(
    status: object,
    *,
    when_unknown: _FenceOutcome,
) -> _FenceOutcome:
    """Classify the status a fence attempt surfaced.

    ``when_unknown`` is what an absent or unreadable status means on the path
    that read it: a 200 fence is a cancellation whether or not the body
    bothers to name a status, while a 409 is terminality of an unknown kind
    and must not be reported as a clean stop.
    """
    if not isinstance(status, str) or not status:
        return when_unknown
    if status in _CANCELLED_STATUSES:
        return _FenceCancelled()
    return _FenceRanToTerminal(status=status)


async def _terminal_outcome_by_job_id(
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
) -> _FenceOutcome:
    """Read back why a job was already terminal when cancellation reached it.

    Neither ``DELETE`` names the terminal state behind its 409 in a field —
    smithers answers both with only a ``detail`` string — so this costs one
    extra read on a path that only runs when a
    registration hook has already rejected — rare, and the answer decides
    whether the caller is told work completed behind its back.

    Bounded for the same reason the cancellation itself is: this runs while a
    hook exception is waiting to propagate, and ``get_job`` would otherwise
    inherit a caller client built with ``timeout=None``. A read that times out
    leaves the outcome terminal-of-unknown-kind, which is what not knowing the
    status means anywhere else on this path.
    """
    try:
        job = await asyncio.wait_for(
            get_job(job_id, dispatch_url=url, http=http),
            timeout=_CANCEL_TIMEOUT_SECONDS,
        )
    except (DispatchError, asyncio.TimeoutError) as exc:
        logger.warning(
            "Could not read the terminal status of dispatch job %s: %s",
            job_id,
            exc,
        )
        return _FenceRanToTerminal(status=None)
    return _fence_outcome_for_status(
        job.get("status"), when_unknown=_FenceRanToTerminal(status=None),
    )


async def _fence_rejected_submission(
    *,
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
    idempotency_key: str | None,
) -> _FenceOutcome:
    """Durably stop an accepted job whose registration hook rejected it.

    Returns the outcome rather than a bare success flag, because "cancelled"
    is only one of three things that can happen and the other two both matter
    to the caller. An unacknowledged cancellation means the job may still be
    running, so a retry under the same idempotency key can reuse live work —
    exactly what the record-or-cancel boundary exists to prevent. A job that
    ran to a terminal state ran with nobody recording it, and whatever effects
    it had stand — which the caller must not read as a clean rejection either.

    ``cancel_or_fence`` is preferred when the submission carried an
    idempotency key, because smithers records a tombstone against the key
    even when no job is visible yet. The per-job ``DELETE`` only speaks to
    the attempt in hand.
    """
    last_error: DispatchError | None = None
    for attempt in range(1, _FENCE_ATTEMPTS + 1):
        try:
            if idempotency_key is not None:
                was_terminal, fenced = await _cancel_or_fence_detailed(
                    idempotency_key, dispatch_url=url, http=http,
                )
                if was_terminal and not fenced.get("status"):
                    # Smithers answers a keyed 409 with only a ``detail``
                    # string, so which terminal state the fence hit has to
                    # be read back — by the job id already in hand.
                    return await _terminal_outcome_by_job_id(
                        http, url, job_id,
                    )
                return _fence_outcome_for_status(
                    fenced.get("status"), when_unknown=_FenceCancelled(),
                )
            outcome = await _delete_remote_job(http, url, job_id)
            if outcome == "already_terminal":
                return await _terminal_outcome_by_job_id(http, url, job_id)
            return _FenceCancelled()
        except DispatchError as exc:
            last_error = exc
            if not exc.retryable or attempt == _FENCE_ATTEMPTS:
                break
            logger.warning(
                "Cancellation of dispatch job %s unacknowledged "
                "(attempt %d/%d): %s",
                job_id,
                attempt,
                _FENCE_ATTEMPTS,
                exc,
            )
            await asyncio.sleep(_FENCE_RETRY_SECONDS)
    if last_error is None:
        # Only reachable with _FENCE_ATTEMPTS configured to zero: nothing was
        # attempted, so nothing was fenced, and saying so beats reporting a
        # cancellation that never happened.
        last_error = DispatchError(
            f"Dispatch job {job_id} cancellation was never attempted",
            job_id=job_id,
        )
    return _FenceUnacknowledged(error=last_error)


def _fence_warning(job_id: str, outcome: _FenceOutcome) -> str | None:
    """The one-line caller-visible warning for a fence that wasn't clean."""
    if isinstance(outcome, _FenceCancelled):
        return None
    if isinstance(outcome, _FenceRanToTerminal):
        reached = (
            f"reached status {outcome.status!r}"
            if outcome.status is not None
            else "was already terminal"
        )
        return (
            f"Dispatch job {job_id} {reached} before it could be cancelled, so "
            f"it ran unrecorded and any effects it had stand even though this "
            f"hook rejected it."
        )
    return (
        f"Dispatch job {job_id} was not confirmed cancelled after this hook "
        f"rejected it and may still be running: {outcome.error}"
    )


async def _drain(task: asyncio.Task[Any]) -> asyncio.CancelledError | None:
    """Await ``task`` to completion, surviving repeated cancellation.

    ``asyncio.shield`` keeps a task running when its awaiter is cancelled,
    but it does not protect the ``await`` on it — a single ``cancel()`` would
    otherwise abandon the task mid-flight, skip the awaiter's cleanup, and
    leave a detached task to outlive the HTTP client it is still using. So a
    cancellation arriving here is recorded and the wait resumed; the caller
    decides what to do with it once the task has actually landed, and should
    re-raise it — absorbing a cancellation tells the canceller this coroutine
    stopped when it did not.

    The caller also owns inspecting the completed task for failures. Draining
    retains any caller cancellation even when the task itself raises.
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            # shield propagates task failures too. Leave their interpretation
            # to the caller, retaining any cancellation already received.
            break
    return cancellation


async def _await_fence(
    fence_task: asyncio.Task[_FenceOutcome],
    job_id: str,
) -> tuple[_FenceOutcome, asyncio.CancelledError | None]:
    """Drain a fence to its outcome, plus any cancellation that arrived.

    A fence that fails in some way it did not anticipate is reported as an
    unacknowledged cancellation rather than raised, because the exception this
    whole path exists to propagate is the hook's, not this one's.
    """
    cancellation = await _drain(fence_task)

    if fence_task.cancelled():
        # Nothing here cancels the shielded task, so this is the event loop
        # being torn down underneath it. The fence did not land.
        return (
            _FenceUnacknowledged(
                error=DispatchError(
                    f"Dispatch job {job_id} cancellation was itself cancelled",
                    job_id=job_id,
                    retryable=True,
                ),
            ),
            cancellation,
        )
    fence_error = fence_task.exception()
    if fence_error is not None:
        return (
            _FenceUnacknowledged(
                error=DispatchError(
                    f"Dispatch job {job_id} cancellation failed: {fence_error}",
                    job_id=job_id,
                    retryable=True,
                ),
            ),
            cancellation,
        )
    return fence_task.result(), cancellation


async def _wait_for_terminal(
    *,
    http: httpx.AsyncClient,
    url: str,
    job_id: str,
    cfg: _PollConfig,
    wait_timeout_seconds: float,
    started_at: float,
    listener: Any,
    callback_nonce: str | None,
    callback_future: Any,
) -> dict[str, Any]:
    """Wait for one submitted job without owning its cancellation policy."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_timeout_seconds

    def timeout_error(suffix: str = "") -> JobTimeoutError:
        grace = cfg.cleanup_grace_seconds
        grace_note = (
            f" plus {grace:g}s cleanup grace" if grace > 0 else ""
        )
        return JobTimeoutError(
            f"Dispatch job {job_id} timed out waiting for a terminal status after "
            f"{cfg.timeout_seconds}s execution timeout{grace_note}{suffix}",
            job_id=job_id,
            timeout_seconds=cfg.timeout_seconds,
        )

    # On non-timeout listener trouble (listener stopped mid-flight,
    # malformed callback body) fall through to polling — smithers still has
    # the job_id and the poll endpoint can recover the result.
    callback_data: dict[str, Any] | None = None
    if listener is not None and callback_future is not None:
        try:
            callback_data = await asyncio.wait_for(
                callback_future, timeout=max(0.0, deadline - loop.time()),
            )
        except asyncio.TimeoutError as exc:
            raise timeout_error(" (callback not received)") from exc
        except Exception as exc:
            logger.warning(
                "Callback future for job %s failed (%s) — falling back to polling",
                job_id,
                exc,
            )
            if callback_nonce is not None:
                listener.unregister(callback_nonce)
            callback_data = None

    if callback_data is not None:
        if not isinstance(callback_data, dict):
            raise DispatchError(
                f"Callback for job {job_id} delivered non-object body: {callback_data!r}",
                job_id=job_id,
            )

        status = callback_data.get("status", "")
        if status == "complete":
            logger.info(
                "Dispatch job %s complete via callback (%.0fms)",
                job_id,
                (time.time() - started_at) * 1000,
            )
            # Callback bodies aren't guaranteed to carry the id — make the
            # success-path job dict always self-identifying.
            callback_data.setdefault("job_id", job_id)
            return callback_data
        if status == "failed":
            raise DispatchError(
                callback_data.get("error") or f"Dispatch job {job_id} failed",
                job_id=job_id,
                status="failed",
            )
        if status == "cancelled":
            raise DispatchError(
                f"Dispatch job {job_id} was cancelled",
                job_id=job_id,
                status="cancelled",
            )
        raise DispatchError(
            f"Unexpected callback status {status!r} for job {job_id}",
            job_id=job_id,
            status=status,
        )

    interval = cfg.poll_interval
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise timeout_error()

        await asyncio.sleep(min(interval, remaining))
        interval = min(interval * 2, cfg.poll_max_interval)

        try:
            poll = await http.get(f"{url}/jobs/{job_id}")
            poll.raise_for_status()
        except httpx.ConnectError:
            logger.warning("Lost connection polling job %s, retrying...", job_id)
            continue
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise DispatchError(
                    f"Dispatch job {job_id} not found (expired or invalid)",
                    job_id=job_id,
                    status="not_found",
                ) from exc
            logger.warning(
                "HTTP %s polling job %s, retrying...",
                exc.response.status_code,
                job_id,
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
                job_id,
                (time.time() - started_at) * 1000,
            )
            data.setdefault("job_id", job_id)
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
    idempotency_key: str | None = None,
    poll_config: _PollConfig | None = None,
    listener: Any = None,  # CallbackListener | None — typed via Any to keep
                            # jig.dispatch.listener import optional
    on_submitted: Callable[[str], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """Submit a job to smithers, wait for a terminal status, return the job data.

    ``on_submitted``, when provided, is invoked with the smithers job id
    immediately after submission is accepted — before the terminal wait —
    so callers can durably record the correlation while the job runs.
    The hook may be synchronous or asynchronous.  It completes before Jig
    begins waiting for the result.  If it rejects, Jig cancels the accepted
    remote job before propagating the hook exception, providing a safe
    record-or-cancel boundary for durable correlation stores.  That
    cancellation goes through the durable idempotency-key fence when the
    submission carried a key, and a cancellation smithers never acknowledges
    is retried and then reported as a note on the propagated hook exception
    — never silently swallowed, because the job may still be running.

    When ``listener`` is provided (and its health check passes), the wait
    is an ``asyncio.Future`` resolved by a smithers HTTP callback —
    callers don't pay for a polling coroutine. Health-check failure or
    any other listener trouble falls back to the polling path silently,
    so the caller still gets an answer.

    Returns the full job dict from smithers on ``status == "complete"``.
    Raises :class:`DispatchError` / :class:`JobTimeoutError` on terminal
    failure. Transient polling errors (connection drops, malformed JSON)
    are logged and retried within the timeout window.
    """
    cfg = poll_config or _PollConfig()
    url = dispatch_url.rstrip("/")
    start = time.time()

    # Health-probe the listener once up front. If it doesn't respond the
    # caller still gets an answer via the poll path.
    if listener is not None:
        try:
            await listener.health_check()
        except Exception as e:
            logger.info(
                "Callback listener unhealthy (%s) — falling back to polling",
                e,
            )
            listener = None

    callback_nonce: str | None = None
    callback_future: Any = None  # asyncio.Future[dict[str, Any]]
    if listener is not None:
        callback_nonce, callback_future = listener.register()

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
        # Phase 9 has workers read this and reparent their spans.
        submission["trace_context"] = trace_context
    if idempotency_key is not None:
        submission["idempotency_key"] = idempotency_key
    if listener is not None and callback_nonce is not None:
        submission["callback_url"] = listener.url_for(callback_nonce)

    # --- Submit ---
    try:
        resp = await http.post(f"{url}/jobs", json=submission)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        if listener is not None and callback_nonce is not None:
            listener.unregister(callback_nonce)
        # Transient: dispatch server may be momentarily unreachable
        # (restart, network blip). Mark retryable so agent loops can
        # retry instead of terminating the run.
        raise DispatchError(
            f"Cannot reach dispatch server at {url}",
            retryable=True,
        ) from e
    except httpx.HTTPStatusError as e:
        if listener is not None and callback_nonce is not None:
            listener.unregister(callback_nonce)
        # 4xx = terminal (bad submission), 5xx = transient (server
        # overloaded / misconfigured, worth a retry).
        retryable = e.response.status_code >= 500
        raise DispatchError(
            f"Dispatch submission failed: {e.response.status_code} {e.response.text}",
            retryable=retryable,
        ) from e
    except httpx.RequestError as e:
        if listener is not None and callback_nonce is not None:
            listener.unregister(callback_nonce)
        # Timeouts, DNS failures, mid-request disconnects — all transient.
        raise DispatchError(
            f"Dispatch request error: {e}",
            retryable=True,
        ) from e

    try:
        job_id = resp.json()["job_id"]
    except (ValueError, KeyError) as e:
        if listener is not None and callback_nonce is not None:
            listener.unregister(callback_nonce)
        raise DispatchError(
            f"Unexpected dispatch response: {resp.text}",
        ) from e

    logger.info("Dispatch job %s submitted (task_type=%s)", job_id, task_type)
    if on_submitted is not None:
        try:
            hook_result = on_submitted(job_id)
            if inspect.isawaitable(hook_result):
                await hook_result
        except BaseException as hook_error:
            # Acceptance already happened.  Do not leave uncorrelated work
            # running when its durable registration fence rejects.
            fence_task = asyncio.create_task(
                _fence_rejected_submission(
                    http=http,
                    url=url,
                    job_id=job_id,
                    idempotency_key=idempotency_key,
                ),
            )
            try:
                fence_outcome, cancellation = await _await_fence(
                    fence_task, job_id,
                )
            finally:
                # In a finally so the nonce cannot leak if draining the fence
                # raises something _await_fence didn't anticipate.
                if listener is not None and callback_nonce is not None:
                    listener.unregister(callback_nonce)
            warning = _fence_warning(job_id, fence_outcome)
            if warning is not None:
                # The hook failure stays the raised exception — it is what the
                # caller asked about — but it must not read as a clean
                # record-or-cancel rejection when the job may still be live,
                # or when it already ran.
                logger.error("%s (on_submitted hook rejected it)", warning)
                hook_error.add_note(warning)
            if cancellation is not None:
                # This caller was cancelled while the fence was in flight.
                # Cancellation wins: a coroutine that absorbs it and returns
                # some other exception has told its canceller it stopped when
                # it did not, which breaks task groups and every other
                # structured-concurrency caller. The hook failure rides along
                # as the cause, and the fence warning is restated as a note so
                # it survives a caller that only reads ``str(exc)``.
                cancellation.add_note(
                    f"Cancelled while propagating a rejected on_submitted hook "
                    f"for dispatch job {job_id}: {hook_error!r}",
                )
                if warning is not None:
                    cancellation.add_note(warning)
                raise cancellation from hook_error
            raise
    wait_timeout_seconds = max(
        0.0,
        cfg.timeout_seconds + cfg.cleanup_grace_seconds,
    )
    try:
        return await _wait_for_terminal(
            http=http,
            url=url,
            job_id=job_id,
            cfg=cfg,
            wait_timeout_seconds=wait_timeout_seconds,
            started_at=start,
            listener=listener,
            callback_nonce=callback_nonce,
            callback_future=callback_future,
        )
    except asyncio.CancelledError:
        # The local request/response owner has abandoned the call. Wait for
        # smithers to propagate cancellation to its worker before allowing a
        # retry to occupy that same worker slot. Drained rather than merely
        # shielded, so a second cancel() cannot abandon it mid-flight.
        await _cancel_remote_job_drained(http, url, job_id)
        raise
    except JobTimeoutError:
        if cfg.cancel_on_timeout:
            cancellation = await _cancel_remote_job_drained(http, url, job_id)
            if cancellation is not None:
                # Cancelled while cancelling the timed-out job: cancellation
                # wins, with the timeout as its context.
                raise cancellation
        raise
    finally:
        if listener is not None and callback_nonce is not None:
            listener.unregister(callback_nonce)


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


async def get_job(
    job_id: str,
    *,
    dispatch_url: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return the current smithers representation of ``job_id``.

    This performs one ``GET`` and does not wait for a terminal state.
    """
    transport = http or _get_shared_http()
    url = (dispatch_url or default_dispatch_url()).rstrip("/")
    what = f"Dispatch job {job_id} status request"
    response = await _send(
        transport.get,
        f"{url}/jobs/{job_id}",
        what=what,
        job_id=job_id,
        error_statuses={404: "not_found"},
    )
    return _json_object(response, what=what, job_id=job_id)


async def cancel_job(
    job_id: str,
    *,
    dispatch_url: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Cancel ``job_id`` and return smithers' acknowledged job state.

    Smithers only succeeds after an active worker acknowledges cancellation.
    A 409 means the job was already terminal; a 503 means it remains active.
    """
    transport = http or _get_shared_http()
    url = (dispatch_url or default_dispatch_url()).rstrip("/")
    what = f"Dispatch job {job_id} cancellation"
    response = await _send(
        transport.delete,
        f"{url}/jobs/{job_id}",
        what=what,
        job_id=job_id,
        error_statuses={404: "not_found", 409: "already_terminal"},
    )
    return _json_object(response, what=what, job_id=job_id)


async def get_job_by_idempotency_key(
    idempotency_key: str,
    *,
    dispatch_url: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return the job associated with ``idempotency_key``.

    This is a single status read. The key is URL-encoded as one path segment;
    a missing job is reported as ``DispatchError(status="not_found")``.
    """
    transport = http or _get_shared_http()
    url = (dispatch_url or default_dispatch_url()).rstrip("/")
    what = f"Dispatch idempotency key {idempotency_key!r} status request"
    response = await _send(
        transport.get,
        f"{url}/jobs/by-idempotency/{quote(idempotency_key, safe='')}",
        what=what,
        error_statuses={404: "not_found"},
    )
    return _json_object(response, what=what)


async def cancel_or_fence(
    idempotency_key: str,
    *,
    dispatch_url: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Durably prevent work for ``idempotency_key`` from remaining active.

    Smithers records a cancellation tombstone even when no matching job is
    visible yet. A 200 response means cancellation completed (and may include
    a ``job_id``). A 409 means the job was already terminal, but the tombstone
    is still durable, so its response is also returned as a successful fence.
    A 503 means worker cancellation was not acknowledged and is retryable.

    Both codes fence, so both are returned the same way here. Callers that
    must tell "I stopped it" from "it had already finished" — the
    record-or-cancel boundary is the one that must — use
    :func:`_cancel_or_fence_detailed`.
    """
    _, data = await _cancel_or_fence_detailed(
        idempotency_key, dispatch_url=dispatch_url, http=http,
    )
    return data


async def _cancel_or_fence_detailed(
    idempotency_key: str,
    *,
    dispatch_url: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> tuple[bool, dict[str, Any]]:
    """:func:`cancel_or_fence`, plus whether the job was already terminal.

    The flag is the 409, kept separate from the body because a 200 fence is a
    cancellation whether or not smithers names a status in its response — so
    the body alone cannot distinguish the two.
    """
    transport = http or _get_shared_http()
    url = (dispatch_url or default_dispatch_url()).rstrip("/")
    what = f"Dispatch idempotency key {idempotency_key!r} fencing"
    response = await _send(
        transport.delete,
        f"{url}/jobs/by-idempotency/{quote(idempotency_key, safe='')}",
        what=what,
        passthrough=frozenset({409}),
        timeout=_CANCEL_TIMEOUT_SECONDS,
    )
    return response.status_code == 409, _json_object(response, what=what)


async def aclose() -> None:
    """Close the shared httpx client and the callback listener, if any.

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

    # Tear down the listener too so `aclose()` is the single shutdown
    # entry point for dispatch state. Import here to avoid pulling
    # aiohttp in for callers that never start a listener.
    try:
        from jig.dispatch import listener as _listener_mod
    except ImportError:
        return
    await _listener_mod.stop()


def _current_listener() -> Any:
    """Return the active callback listener (if any) without forcing the
    aiohttp import on callers who never started one.

    Returns ``None`` when aiohttp isn't installed, the listener module
    hasn't been imported yet, or no listener is running.
    """
    try:
        from jig.dispatch import listener as _listener_mod
    except ImportError:
        return None
    return _listener_mod._active_listener()


async def run(
    fn_ref: str,
    payload: dict[str, Any] | None = None,
    *,
    dispatch_url: str | None = None,
    requester: str = "jig",
    machine: str | None = None,
    trace_context: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    timeout_seconds: int = 300,
    cleanup_grace_seconds: float = 10.0,
    cancel_on_timeout: bool = True,
    poll_interval: float = 0.5,
    poll_max_interval: float = 5.0,
    http: httpx.AsyncClient | None = None,
    on_submitted: Callable[[str], Awaitable[None] | None] | None = None,
) -> Any:
    """Execute ``fn_ref`` on a smithers worker, await the result.

    ``on_submitted``, when provided, receives the smithers job id at
    acceptance time (before the result wait). Async hooks are awaited. If
    the hook rejects, the accepted job is cancelled before that exception
    is propagated.

    ``idempotency_key`` identifies this logical submission to smithers. If a
    transient response loss makes the caller retry with the same key, smithers
    returns the existing job and Jig resumes polling it instead of creating a
    duplicate. Omitting it preserves the original submission behavior.

    ``fn_ref`` is the ``"package.module:function"`` identifier the
    worker's function registry knows (populated via the
    ``jig.smithers_fn`` entry-point group). ``payload`` becomes the
    function's kwargs.

    If :func:`jig.dispatch.listen` is running, the wait uses the
    callback listener — no per-call polling coroutine. Otherwise falls
    back to polling. The choice is automatic; callers don't opt in
    per-call.

    ``timeout_seconds`` is the execution deadline sent to smithers. Jig waits
    an additional ``cleanup_grace_seconds`` for smithers to publish the
    terminal result. If the caller cancels this coroutine, Jig asks smithers
    to cancel the remote job before propagating cancellation. A client-side
    timeout does the same by default; set ``cancel_on_timeout=False`` only
    when the remote job is intentionally durable beyond this request.

    Returns whatever the worker put in ``job.result["value"]``. Raises
    :class:`DispatchError` on failure, :class:`JobTimeoutError` on timeout.
    Use this for deterministic steps you want offloaded — backtests,
    embeddings, reindexes — while LLM calls go through
    :class:`jig.llm.DispatchClient`.
    """
    transport = http or _get_shared_http()
    data = await _submit_and_poll(
        http=transport,
        dispatch_url=dispatch_url or default_dispatch_url(),
        task_type="function",
        payload={"fn_ref": fn_ref, "args": payload or {}},
        requester=requester,
        machine=machine,
        trace_context=trace_context,
        idempotency_key=idempotency_key,
        poll_config=_PollConfig(
            timeout_seconds=timeout_seconds,
            cleanup_grace_seconds=max(0.0, cleanup_grace_seconds),
            cancel_on_timeout=cancel_on_timeout,
            poll_interval=poll_interval,
            poll_max_interval=poll_max_interval,
        ),
        listener=_current_listener(),
        on_submitted=on_submitted,
    )
    # Don't ``or {}`` here — that rewrites legitimate falsy returns
    # (``0``, ``False``, ``[]``, ``""``) into an empty dict and breaks
    # any dispatched function whose natural result is falsy.
    result = data.get("result")
    if isinstance(result, dict) and "value" in result:
        return result["value"]
    return result
