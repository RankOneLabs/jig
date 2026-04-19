"""Callback listener for smithers dispatch.

Replaces per-job polling when smithers can reach back into the jig
process. Each dispatched job registers an :class:`asyncio.Future`
keyed by a callback nonce; a small :mod:`aiohttp.web` HTTP server
resolves those futures when smithers POSTs job results. Sweep-style
workloads now cost one HTTP receiver per process instead of N
polling coroutines.

Opt-in via ``pip install 'jig[callback]'`` — importing this module
without ``aiohttp`` raises with an actionable install hint. Callers
that only need the polling path never trigger the import.

Auth is a shared secret in the callback URL query string
(``?token=...``). Not HMAC-grade — the attack model is accidental
callbacks crossing wires between jig processes, not an active
attacker on the Tailscale network. See phase-10 plan for the full
threat-model rationale.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import uuid
from typing import Any

from jig.core.errors import JigError
from jig.core.retry import with_retry

try:
    from aiohttp import web
    import aiohttp
except ImportError as exc:
    raise ImportError(
        "jig.dispatch.listener requires aiohttp. "
        "Install with: pip install 'jig[callback]'"
    ) from exc

logger = logging.getLogger(__name__)


class ListenerError(JigError):
    """Callback listener failed to start, bind, or dispatch a callback."""


class CallbackListener:
    """HTTP receiver that resolves per-job futures on smithers callback.

    Not instantiated directly — call :func:`listen` to get the
    singleton. The class is public so tests and type hints can
    reference it.
    """

    def __init__(self) -> None:
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._bound_port: int | None = None
        self._base_url: str | None = None
        self._token: str | None = None
        # nonce → future for in-flight callbacks
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    @property
    def url(self) -> str:
        """Base URL smithers should POST callbacks to, including port."""
        if self._base_url is None:
            raise ListenerError("Listener not started — call start() first")
        return self._base_url

    @property
    def port(self) -> int:
        if self._bound_port is None:
            raise ListenerError("Listener not started — call start() first")
        return self._bound_port

    def url_for(self, nonce: str) -> str:
        """Full callback URL for a given nonce, including the auth token."""
        if self._base_url is None or self._token is None:
            raise ListenerError("Listener not started")
        return f"{self._base_url}/callbacks/{nonce}?token={self._token}"

    def register(self) -> tuple[str, asyncio.Future[dict[str, Any]]]:
        """Reserve a (nonce, future) pair before submitting a job.

        Registration happens before submit so we're never racing against
        a callback that arrives faster than the registration. The nonce
        is a fresh UUID4 per call — collisions are not a concern.
        """
        nonce = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[nonce] = future
        return nonce, future

    def unregister(self, nonce: str) -> None:
        """Drop a pending future. Caller should invoke this if the
        submit fails or the caller decides to fall back to polling.
        No-op if the nonce isn't registered."""
        self._pending.pop(nonce, None)

    async def start(
        self,
        *,
        port: int = 0,
        host: str = "0.0.0.0",
        base_url: str | None = None,
    ) -> None:
        """Bind the HTTP receiver. Idempotent — subsequent calls no-op."""
        if self._runner is not None:
            return

        self._token = os.getenv("JIG_CALLBACK_SECRET") or secrets.token_urlsafe(32)

        app = web.Application()
        app.router.add_post("/callbacks/{nonce}", self._handle_callback)
        app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()

        # `port=0` lets the OS pick; recover the actual port from the
        # server sockets rather than trusting the kwarg.
        bound_port = port
        if bound_port == 0:
            server = site._server  # aiohttp exposes no public accessor
            if server is not None and server.sockets:
                bound_port = server.sockets[0].getsockname()[1]

        self._runner = runner
        self._site = site
        self._bound_port = bound_port
        self._base_url = base_url or self._resolve_base_url(bound_port)
        logger.info(
            "Callback listener bound to %s:%d (base_url=%s)",
            host, bound_port, self._base_url,
        )

    async def stop(self) -> None:
        """Tear down the receiver and fail any pending futures.

        Idempotent. Pending futures resolve with :class:`ListenerError`
        so awaiters unblock cleanly — a listener shutdown is a terminal
        event for every in-flight job; callers who want resumability
        should fall back to polling before stopping."""
        for nonce, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(
                    ListenerError(f"Listener stopped before callback for {nonce}")
                )
        self._pending.clear()

        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

        self._bound_port = None
        self._base_url = None
        self._token = None

    async def health_check(
        self,
        *,
        http: "aiohttp.ClientSession | None" = None,
    ) -> None:
        """Self-probe GET /health. Raises on failure; returns silently on success.

        Used by :func:`jig.dispatch.run` pre-submit to confirm the
        listener still serves before committing to the callback path.
        """
        if self._base_url is None:
            raise ListenerError("Listener not started")
        url = f"{self._base_url}/health"
        owns = http is None
        session = http or aiohttp.ClientSession()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status != 200:
                    raise ListenerError(f"Health check got status {resp.status}")
        except aiohttp.ClientError as e:
            raise ListenerError(f"Health check failed: {e}") from e
        finally:
            if owns:
                await session.close()

    # --- HTTP handlers ---

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_callback(self, request: web.Request) -> web.Response:
        nonce = request.match_info["nonce"]
        token = request.query.get("token", "")

        if not self._token or not secrets.compare_digest(token, self._token):
            logger.warning("Callback for %s rejected — token mismatch", nonce)
            return web.json_response({"error": "unauthorized"}, status=401)

        future = self._pending.pop(nonce, None)
        if future is None:
            # Smithers may retry a callback after we already timed out
            # the future locally, or an operator may cross wires. Either
            # way, a 404 tells smithers to stop retrying this callback.
            logger.warning("Callback for %s rejected — unknown nonce", nonce)
            return web.json_response({"error": "unknown nonce"}, status=404)

        try:
            body = await request.json()
        except Exception as e:
            logger.warning("Callback for %s had invalid JSON: %s", nonce, e)
            future.set_exception(ListenerError(f"Malformed callback body: {e}"))
            return web.json_response({"error": "invalid body"}, status=400)

        if not future.done():
            future.set_result(body)
        return web.json_response({"status": "accepted"})

    # --- Helpers ---

    @staticmethod
    def _resolve_base_url(port: int) -> str:
        """Return a URL smithers should reach jig on.

        Precedence:
          1. ``JIG_CALLBACK_HOST`` env var.
          2. ``socket.gethostname()``.

        On the Springfield Tailscale network MagicDNS resolves short
        hostnames (e.g. ``otto``) from any node, so the hostname
        default just works. In CI or loopback-only environments,
        callers pass ``base_url="http://127.0.0.1"`` (the hostname
        fallback won't be reachable).
        """
        host = os.getenv("JIG_CALLBACK_HOST") or socket.gethostname()
        return f"http://{host}:{port}"


# --- Module-level singleton ---

_active: CallbackListener | None = None


async def listen(
    *,
    port: int = 0,
    host: str = "0.0.0.0",
    base_url: str | None = None,
) -> CallbackListener:
    """Start (or return) the process-wide callback listener.

    Idempotent — a second call returns the already-running listener.
    Start-up is retried up to 3 times with 1s/2s backoff for transient
    bind errors; exhausting retries surfaces :class:`ListenerError`.
    """
    global _active
    if _active is not None and _active._runner is not None:
        return _active

    listener = _active or CallbackListener()

    async def _start() -> None:
        await listener.start(port=port, host=host, base_url=base_url)

    await with_retry(
        _start,
        max_attempts=3,
        base_delay=1.0,
        retryable=lambda e: isinstance(e, OSError),
    )

    _active = listener
    return listener


async def stop() -> None:
    """Tear down the process-wide listener, if any. Idempotent."""
    global _active
    if _active is None:
        return
    try:
        await _active.stop()
    finally:
        _active = None


def _active_listener() -> CallbackListener | None:
    """Internal: return the current listener if running, else None.

    Used by :mod:`jig.dispatch.client` to decide whether to use the
    callback path or fall back to polling.
    """
    if _active is None or _active._runner is None:
        return None
    return _active


__all__ = [
    "CallbackListener",
    "ListenerError",
    "listen",
    "stop",
]
