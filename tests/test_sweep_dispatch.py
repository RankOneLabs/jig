"""Tests for ``sweep(dispatch=...)`` and ``compare(dispatch=...)``.

Covers the lifecycle contract (who owns the listener, who stops it),
the rejection of unknown backends, and an end-to-end sweep whose
configs route LLM calls through a fake smithers server to confirm every
run resolves via the callback path rather than the poll fallback.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from jig.core.runner import AgentConfig
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    LLMResponse,
    Span,
    SpanKind,
    TracingLogger,
    Usage,
)
from jig.dispatch import listener as listener_mod
from jig.llm.dispatch import DispatchClient
from jig.sweep import compare, sweep
from jig.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
async def reset_dispatch_state():
    import jig.dispatch
    await jig.dispatch.aclose()
    yield
    await jig.dispatch.aclose()


class _NullFeedback(FeedbackLoop):
    async def query(self, q: Any) -> list:
        return []

    async def store_result(self, content: str, input_text: str, metadata: dict | None = None) -> str:
        return "noop"

    async def score(self, result_id: str, scores: list) -> None:
        pass

    async def get_signals(self, query: str, limit: int = 3, min_score: float | None = None, source: Any = None) -> list:
        return []

    async def export_eval_set(self, **kwargs: Any) -> list:
        return []


class _NullTracer(TracingLogger):
    """Drop-on-floor tracer that returns real :class:`Span` objects.

    The runner reads ``span.trace_id`` / ``span.id`` during grading and
    store writes; minimal stub objects without those attributes crash
    the run. Match the shape test_sweep.py uses.
    """

    def start_trace(
        self, name: str, metadata: dict | None = None, kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        return Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now())

    def start_span(self, parent_id: str, kind: SpanKind, name: str, input: Any = None) -> Span:
        return Span(
            id="s", trace_id="t", kind=kind, name=name,
            started_at=datetime.now(), parent_id=parent_id,
        )

    def end_span(self, *a: Any, **kw: Any) -> None:
        pass

    async def get_trace(self, trace_id: str) -> list:
        return []

    async def list_traces(self, **kwargs: Any) -> list:
        return []


class TestDispatchArgValidation:
    async def test_unknown_dispatch_backend_rejected(self):
        cfg = _noop_config(name="x")
        with pytest.raises(ValueError, match="Unknown dispatch backend"):
            await sweep(["hi"], [cfg], dispatch="mcclure")

    async def test_dispatch_none_is_default(self):
        """With dispatch=None, no listener should be started."""
        cfg = _noop_config(name="x")
        await sweep(["hi"], [cfg])
        assert listener_mod._active_listener() is None


class TestListenerLifecycle:
    async def test_sweep_starts_and_stops_listener(self):
        cfg = _noop_config(name="x")
        assert listener_mod._active_listener() is None
        await sweep(["hi"], [cfg], dispatch="smithers")
        # Listener was cleaned up after sweep
        assert listener_mod._active_listener() is None

    async def test_sweep_reuses_pre_existing_listener(self):
        pre = await listener_mod.listen(port=0, host="127.0.0.1")
        try:
            cfg = _noop_config(name="x")
            await sweep(["hi"], [cfg], dispatch="smithers")
            # Sweep didn't stop it — operator still owns it
            assert listener_mod._active_listener() is pre
        finally:
            await listener_mod.stop()

    async def test_compare_starts_and_stops_listener(self):
        cfg = _noop_config(name="x")
        await compare("hi", [cfg], dispatch="smithers")
        assert listener_mod._active_listener() is None


# --- End-to-end sweep via fake smithers + real listener ---


class _FakeSmithers:
    """Minimal smithers that returns canned inference responses via callback."""

    def __init__(self) -> None:
        self.submissions: list[dict] = []
        self.polls_received = 0
        self._runner: web.AppRunner | None = None
        self._port: int | None = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/jobs", self._handle_submit)
        app.router.add_get("/jobs/{job_id}", self._handle_poll)
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
        job_id = f"job-{len(self.submissions)}"
        callback_url = body.get("callback_url")
        if callback_url:
            asyncio.create_task(self._post_callback(callback_url, job_id))
        return web.json_response({"job_id": job_id})

    async def _handle_poll(self, request: web.Request) -> web.Response:
        # Guard so the end-to-end test can assert the poll path was
        # never hit — if any call lands here, something went wrong.
        self.polls_received += 1
        return web.json_response({"error": "test rejects polling"}, status=500)

    async def _post_callback(self, url: str, job_id: str) -> None:
        await asyncio.sleep(0.02)
        payload = {
            "job_id": job_id,
            "status": "complete",
            "result": {
                "content": "canned answer",
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost": 0.0},
                "tool_calls": [],
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as _resp:
                pass


def _minimal_agent_config(name: str, llm: Any) -> AgentConfig[Any]:
    """Build an AgentConfig that run_agent will accept.

    Uses in-process drop-on-floor tracer + feedback to keep the test
    focused on dispatch behavior.
    """
    return AgentConfig(
        name=name,
        description="dispatch-backed sweep test agent",
        system_prompt="you are terse",
        llm=llm,
        feedback=_NullFeedback(),
        tracer=_NullTracer(),
        tools=ToolRegistry([]),
        max_tool_calls=1,
        include_memory_in_prompt=False,
        include_feedback_in_prompt=False,
    )


def _noop_config(name: str) -> AgentConfig[Any]:
    """Agent config with a stub LLM that returns a canned reply.

    Used by the lifecycle tests, which need sweep() to complete
    without actually exercising dispatch. Avoids the real DispatchClient
    and so avoids needing a fake smithers for those tests.
    """
    class _StubLLM:
        async def complete(self, params: CompletionParams) -> LLMResponse:
            return LLMResponse(
                content="ok",
                tool_calls=None,
                usage=Usage(input_tokens=1, output_tokens=1, cost=0.0),
                latency_ms=1.0,
                model="stub",
            )
    return _minimal_agent_config(name, _StubLLM())


class TestEndToEndSweep:
    async def test_every_run_uses_callback_path(self):
        fake = _FakeSmithers()
        smithers_url = await fake.start()
        try:
            # Pre-start the listener so we control the base_url (test
            # environments can't use socket.gethostname() round-trip).
            listener = await listener_mod.listen(port=0, host="127.0.0.1")
            listener._base_url = f"http://127.0.0.1:{listener.port}"

            cfgs = [
                _minimal_agent_config(
                    name=f"cfg-{i}",
                    llm=DispatchClient(
                        model=f"model-{i}",
                        dispatch_url=smithers_url,
                        timeout_seconds=5,
                    ),
                )
                for i in range(3)
            ]
            cases = ["case-1", "case-2"]

            # Don't pass dispatch="smithers" because we're managing the
            # listener ourselves — but the DispatchClients will still
            # pick up the active listener automatically via
            # `_current_listener()`.
            result = await sweep(cases, cfgs, concurrency=6)

            # 2 cases × 3 configs = 6 runs
            assert len(result.runs) == 6
            # Fake smithers saw 6 submissions, each with a callback_url
            assert len(fake.submissions) == 6
            for sub in fake.submissions:
                assert "callback_url" in sub
            # Poll endpoint was never hit — every run went through
            # the callback path
            assert fake.polls_received == 0
            # All runs produced the canned content
            for run in result.runs:
                assert run.result.output == "canned answer"
        finally:
            await fake.stop()
            await listener_mod.stop()
