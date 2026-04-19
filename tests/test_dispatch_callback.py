"""Integration tests for listener-aware dispatch.run().

Each test runs a real :class:`CallbackListener` plus a fake smithers
HTTP server so the callback loop is exercised end to end. The fake
smithers accepts ``POST /jobs``, returns a job_id, then POSTs the
"completed" job record back to the ``callback_url`` carried in the
submission — no polling coroutine needed.
"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import httpx
import pytest
from aiohttp import web

from jig.dispatch import listener as listener_mod
from jig.dispatch.client import DispatchError, JobTimeoutError, run


class FakeSmithers:
    """Minimal smithers stand-in that posts a callback a moment after submit.

    ``submit_behavior`` controls the response to ``POST /jobs``:
      - ``"accept"``: 200 with a job_id and schedule a callback.
      - ``"http_500"``: return 500.
    ``callback_status`` controls the job status POSTed back.
    """

    def __init__(
        self,
        *,
        submit_behavior: str = "accept",
        callback_status: str = "complete",
        callback_result: Any = None,
        callback_error: str | None = None,
        callback_delay: float = 0.05,
    ) -> None:
        self.submit_behavior = submit_behavior
        self.callback_status = callback_status
        self.callback_result = callback_result if callback_result is not None else {"value": 42}
        self.callback_error = callback_error
        self.callback_delay = callback_delay
        self.submissions: list[dict] = []
        self.callbacks_sent: list[str] = []
        self._runner: web.AppRunner | None = None
        self._port: int | None = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/jobs", self._handle_submit)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host="127.0.0.1", port=0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"http://127.0.0.1:{self._port}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_submit(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.submissions.append(body)

        if self.submit_behavior == "http_500":
            return web.json_response({"error": "overloaded"}, status=500)

        job_id = f"job-{len(self.submissions)}"
        callback_url = body.get("callback_url")

        if callback_url:
            asyncio.create_task(
                self._deliver_callback(callback_url, job_id)
            )

        return web.json_response({"job_id": job_id})

    async def _deliver_callback(self, url: str, job_id: str) -> None:
        await asyncio.sleep(self.callback_delay)
        payload: dict[str, Any] = {"job_id": job_id, "status": self.callback_status}
        if self.callback_status == "complete":
            payload["result"] = self.callback_result
        elif self.callback_status == "failed":
            payload["error"] = self.callback_error or "worker blew up"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                self.callbacks_sent.append(f"{resp.status}:{job_id}")


@pytest.fixture
async def fake_smithers():
    server = FakeSmithers()
    url = await server.start()
    try:
        yield server, url
    finally:
        await server.stop()


@pytest.fixture
async def listener_on():
    """Start the process-wide listener on loopback, tear down after."""
    listener = await listener_mod.listen(port=0, host="127.0.0.1")
    # Force loopback URL for tests — gethostname() might not round-trip
    # locally (see test_dispatch_listener.py's live_listener rationale).
    listener._base_url = f"http://127.0.0.1:{listener.port}"
    yield listener
    await listener_mod.stop()


@pytest.fixture(autouse=True)
async def reset_dispatch_state():
    """Clear listener + shared-http state between tests.

    Each test's event loop closes at teardown; a shared httpx.AsyncClient
    or CallbackListener bound to that loop becomes poison for the next
    test's `aclose()` call. `jig.dispatch.aclose()` handles both.
    """
    import jig.dispatch
    await jig.dispatch.aclose()
    yield
    await jig.dispatch.aclose()


class TestCallbackHappyPath:
    async def test_run_uses_callback_when_listener_active(
        self, fake_smithers, listener_on
    ):
        server, smithers_url = fake_smithers

        result = await run(
            "demo.fn:echo",
            {"x": 1},
            dispatch_url=smithers_url,
        )

        # run() unwraps result["value"] per its existing contract.
        assert result == 42
        # Smithers received exactly one submission with a callback_url.
        assert len(server.submissions) == 1
        submission = server.submissions[0]
        assert "callback_url" in submission
        assert submission["callback_url"].startswith(listener_on.url)
        assert "token=" in submission["callback_url"]
        # Callback delivery happens in a separate task on the fake
        # smithers side; yield a tick so its POST bookkeeping completes
        # before we assert on it.
        await asyncio.sleep(0.05)
        assert len(server.callbacks_sent) == 1
        assert server.callbacks_sent[0].startswith("200:")


class TestCallbackFailsGracefully:
    async def test_listener_unhealthy_falls_back_to_poll(
        self, fake_smithers, listener_on
    ):
        server, smithers_url = fake_smithers

        # Stop the listener's HTTP runner so the health check fails,
        # but leave the singleton pointer set. `run()` should notice and
        # fall back to polling. The fake smithers has polling disabled
        # (no GET /jobs/{id} handler) so we expect a DispatchError when
        # the poll lookup 404s.
        if listener_on._site is not None:
            await listener_on._site.stop()
            listener_on._site = None

        with pytest.raises(DispatchError):
            await run(
                "demo.fn:echo",
                {},
                dispatch_url=smithers_url,
                timeout_seconds=2,
                poll_interval=0.1,
                poll_max_interval=0.2,
            )

        # Submission went through (poll path is the fallback), but no
        # callback was sent because the listener is dead.
        assert len(server.submissions) == 1
        assert server.callbacks_sent == []


class TestCallbackTerminalStates:
    async def test_failed_status_raises_dispatch_error(self, listener_on):
        server = FakeSmithers(
            callback_status="failed",
            callback_error="worker segfaulted",
        )
        smithers_url = await server.start()
        try:
            with pytest.raises(DispatchError) as exc:
                await run("demo.fn:boom", {}, dispatch_url=smithers_url)
            assert "worker segfaulted" in str(exc.value)
            assert exc.value.status == "failed"
        finally:
            await server.stop()

    async def test_cancelled_status_raises_dispatch_error(self, listener_on):
        server = FakeSmithers(callback_status="cancelled")
        smithers_url = await server.start()
        try:
            with pytest.raises(DispatchError) as exc:
                await run("demo.fn:any", {}, dispatch_url=smithers_url)
            assert exc.value.status == "cancelled"
        finally:
            await server.stop()

    async def test_callback_timeout(self, listener_on):
        # Callback delay longer than the client timeout.
        server = FakeSmithers(callback_delay=2.0)
        smithers_url = await server.start()
        try:
            with pytest.raises(JobTimeoutError) as exc:
                await run(
                    "demo.fn:slow", {},
                    dispatch_url=smithers_url,
                    timeout_seconds=1,
                )
            assert "callback not received" in str(exc.value)
        finally:
            await server.stop()


class TestNoListener:
    """When no listener is registered the behavior matches pre-phase-10."""

    async def test_run_uses_poll_path_by_default(self, fake_smithers):
        # No `listener_on` fixture → no listener active. Our fake smithers
        # doesn't serve GET /jobs/{id}, so polling 404s. That's fine —
        # we just want to confirm the submission went out without a
        # callback_url.
        server, smithers_url = fake_smithers

        with pytest.raises(DispatchError):
            await run(
                "demo.fn:x", {},
                dispatch_url=smithers_url,
                timeout_seconds=2,
                poll_interval=0.1,
                poll_max_interval=0.2,
            )

        assert len(server.submissions) == 1
        assert "callback_url" not in server.submissions[0]


class TestSubmitFailureCleanup:
    async def test_submit_500_unregisters_nonce(self, listener_on):
        server = FakeSmithers(submit_behavior="http_500")
        smithers_url = await server.start()
        try:
            with pytest.raises(DispatchError):
                await run("demo.fn:x", {}, dispatch_url=smithers_url)

            # Nothing pending — the failed submit unregistered the
            # future.
            assert listener_on._pending == {}
        finally:
            await server.stop()
