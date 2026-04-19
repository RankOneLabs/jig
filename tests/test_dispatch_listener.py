"""Tests for the callback listener.

These tests bind real HTTP sockets (via aiohttp) and POST against the
running handler — the goal is to catch wiring bugs that a mock transport
would hide. Every listener is torn down in the fixture to avoid
leaking ports across tests.
"""
from __future__ import annotations

import asyncio
import os

import pytest

# Skip the whole module when the optional extra isn't installed.
# `jig.dispatch.listener` itself raises ImportError with an install
# hint; collecting these tests without aiohttp would cascade that.
aiohttp = pytest.importorskip("aiohttp")

from jig.dispatch import listener as listener_mod
from jig.dispatch.listener import CallbackListener, ListenerError


@pytest.fixture
async def live_listener():
    """A fresh CallbackListener bound to loopback on a random port.

    `base_url` pinned to 127.0.0.1 so tests round-trip over loopback
    rather than the host's Tailscale hostname (which may not resolve to
    the same interface the socket is bound on).
    """
    listener = CallbackListener()
    await listener.start(port=0, host="127.0.0.1")
    # Override base_url post-start: we want the URL to point at the
    # loopback address the socket is actually bound on, not socket's
    # gethostname() default that's meant for cross-host reachability.
    listener._base_url = f"http://127.0.0.1:{listener.port}"
    yield listener
    await listener.stop()


@pytest.fixture(autouse=True)
async def reset_module_singleton():
    """Clear the module-level _active between tests so listen()/stop()
    start from a clean slate and don't leak across tests."""
    await listener_mod.stop()
    yield
    await listener_mod.stop()


async def _post_callback(
    url: str, body: dict | None = None, raw: str | None = None
) -> tuple[int, dict]:
    """POST a callback body and return (status, json_or_error_dict)."""
    async with aiohttp.ClientSession() as session:
        if raw is not None:
            async with session.post(url, data=raw) as resp:
                try:
                    return resp.status, await resp.json()
                except Exception:
                    return resp.status, {"raw": await resp.text()}
        else:
            async with session.post(url, json=body or {}) as resp:
                return resp.status, await resp.json()


class TestStartStopLifecycle:
    async def test_start_binds_random_port(self, live_listener):
        assert live_listener.port > 0
        assert live_listener.url.startswith("http://")

    async def test_start_is_idempotent(self, live_listener):
        port_before = live_listener.port
        await live_listener.start(port=0, host="127.0.0.1")
        assert live_listener.port == port_before

    async def test_stop_allows_fresh_start(self):
        listener = CallbackListener()
        await listener.start(port=0, host="127.0.0.1")
        first_port = listener.port
        await listener.stop()

        await listener.start(port=0, host="127.0.0.1")
        assert listener.port != 0
        # Port may or may not be the same; what matters is that the
        # listener works after restart.
        await listener.stop()

    async def test_stop_is_idempotent(self, live_listener):
        await live_listener.stop()
        await live_listener.stop()

    async def test_url_raises_before_start(self):
        listener = CallbackListener()
        with pytest.raises(ListenerError):
            _ = listener.url


class TestHealthEndpoint:
    async def test_health_returns_200(self, live_listener):
        url = f"{live_listener.url}/health"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["status"] == "ok"

    async def test_health_check_succeeds(self, live_listener):
        # Should not raise.
        await live_listener.health_check()

    async def test_health_check_raises_after_stop(self, live_listener):
        await live_listener.stop()
        with pytest.raises(ListenerError):
            await live_listener.health_check()


class TestRegisterResolve:
    async def test_register_returns_unique_nonces(self, live_listener):
        n1, f1 = live_listener.register()
        n2, f2 = live_listener.register()
        assert n1 != n2
        assert not f1.done()
        assert not f2.done()

    async def test_callback_resolves_future(self, live_listener):
        nonce, future = live_listener.register()
        url = live_listener.url_for(nonce)

        status, body = await _post_callback(
            url, body={"job_id": "j-1", "status": "complete", "result": {"value": 42}}
        )

        assert status == 200
        assert body["status"] == "accepted"
        assert future.done()
        resolved = await future
        assert resolved["status"] == "complete"
        assert resolved["result"]["value"] == 42

    async def test_callback_with_mismatched_token_is_rejected(self, live_listener):
        nonce, future = live_listener.register()
        # Build a URL with the wrong token
        bad_url = f"{live_listener.url}/callbacks/{nonce}?token=wrong"

        status, body = await _post_callback(bad_url, body={"status": "complete"})

        assert status == 401
        assert body["error"] == "unauthorized"
        # Future must remain pending — a mismatched token can't resolve it.
        assert not future.done()

    async def test_callback_for_unknown_nonce_returns_404(self, live_listener):
        url = live_listener.url_for("nonexistent")

        status, body = await _post_callback(url, body={"status": "complete"})

        assert status == 404
        assert body["error"] == "unknown nonce"

    async def test_callback_with_malformed_body_fails_future(self, live_listener):
        nonce, future = live_listener.register()
        url = live_listener.url_for(nonce)

        status, _ = await _post_callback(url, raw="not-json")

        assert status == 400
        assert future.done()
        with pytest.raises(ListenerError):
            await future

    async def test_unregister_drops_pending(self, live_listener):
        nonce, future = live_listener.register()
        live_listener.unregister(nonce)

        url = live_listener.url_for(nonce)
        status, _ = await _post_callback(url, body={"status": "complete"})
        assert status == 404  # treated as unknown nonce after unregister
        # Future was never resolved and the caller abandoned it.
        assert not future.done()

    async def test_unregister_unknown_is_noop(self, live_listener):
        live_listener.unregister("never-registered")


class TestStopFailsPending:
    async def test_stop_fails_pending_futures(self, live_listener):
        _, future1 = live_listener.register()
        _, future2 = live_listener.register()

        await live_listener.stop()

        assert future1.done()
        assert future2.done()
        with pytest.raises(ListenerError):
            await future1
        with pytest.raises(ListenerError):
            await future2


class TestBaseURLResolution:
    async def test_explicit_base_url_with_port_wins(self):
        listener = CallbackListener()
        try:
            await listener.start(
                port=0, host="127.0.0.1", base_url="http://explicit.example:9999"
            )
            assert listener.url == "http://explicit.example:9999"
        finally:
            await listener.stop()

    async def test_explicit_base_url_without_port_gets_bound_port(self):
        """Host-only base_url like `http://127.0.0.1` should pick up the
        actual bound port instead of silently defaulting to 80."""
        listener = CallbackListener()
        try:
            await listener.start(port=0, host="127.0.0.1", base_url="http://127.0.0.1")
            assert listener.url == f"http://127.0.0.1:{listener.port}"
        finally:
            await listener.stop()

    async def test_env_var_used_when_no_kwarg(self, monkeypatch):
        monkeypatch.setenv("JIG_CALLBACK_HOST", "env.example")
        listener = CallbackListener()
        try:
            await listener.start(port=0, host="127.0.0.1")
            assert "env.example" in listener.url
        finally:
            await listener.stop()

    async def test_falls_back_to_hostname(self, monkeypatch):
        monkeypatch.delenv("JIG_CALLBACK_HOST", raising=False)
        listener = CallbackListener()
        try:
            await listener.start(port=0, host="127.0.0.1")
            # Whatever socket.gethostname() returns on the test host
            # should be in the URL.
            import socket
            assert socket.gethostname() in listener.url
        finally:
            await listener.stop()


class TestTokenURLEncoding:
    async def test_special_chars_in_token_are_url_encoded(self, monkeypatch):
        """Tokens with `?/&/=/#/space` must not break the callback URL."""
        monkeypatch.setenv("JIG_CALLBACK_SECRET", "a b&c=d?e#f")
        listener = CallbackListener()
        try:
            await listener.start(port=0, host="127.0.0.1")
            url = listener.url_for("test-nonce")
            # Raw reserved chars must not appear after `token=`
            token_part = url.split("token=", 1)[1]
            assert "&" not in token_part
            assert "#" not in token_part
            # The quoted form should be present
            assert "a%20b%26c%3Dd%3Fe%23f" in token_part
        finally:
            await listener.stop()


class TestHealthCheckCaching:
    async def test_health_check_ttl_skips_probe(self, live_listener):
        """When ttl_seconds is 10, two back-to-back calls should only hit
        the HTTP endpoint once — the second should return from cache."""
        # First call primes the cache
        await live_listener.health_check(ttl_seconds=10.0)
        primed_at = live_listener._last_healthy_at

        # Second call should no-op
        await live_listener.health_check(ttl_seconds=10.0)
        assert live_listener._last_healthy_at == primed_at

    async def test_health_check_ttl_zero_forces_fresh_probe(self, live_listener):
        await live_listener.health_check(ttl_seconds=10.0)
        primed_at = live_listener._last_healthy_at

        # A moment later with ttl=0 should record a new timestamp
        import asyncio
        await asyncio.sleep(0.01)
        await live_listener.health_check(ttl_seconds=0.0)
        assert live_listener._last_healthy_at > primed_at


class TestConcurrentListen:
    async def test_concurrent_listen_returns_same_instance(self):
        """Two concurrent `listen()` callers must observe one listener,
        not race to create two live servers."""
        results = await asyncio.gather(
            listener_mod.listen(port=0, host="127.0.0.1"),
            listener_mod.listen(port=0, host="127.0.0.1"),
            listener_mod.listen(port=0, host="127.0.0.1"),
        )
        # All three references point at the same singleton
        assert results[0] is results[1] is results[2]
        assert listener_mod._active_listener() is results[0]
        await listener_mod.stop()


class TestTokenGeneration:
    async def test_token_defaults_to_random(self, live_listener):
        # Not a great assertion for randomness but at least confirms
        # a token exists and url_for embeds it.
        nonce, _ = live_listener.register()
        url = live_listener.url_for(nonce)
        assert "token=" in url
        assert live_listener._token is not None
        assert len(live_listener._token) >= 32

    async def test_env_secret_overrides(self, monkeypatch):
        monkeypatch.setenv("JIG_CALLBACK_SECRET", "fixed-secret-value")
        listener = CallbackListener()
        try:
            await listener.start(port=0, host="127.0.0.1")
            assert listener._token == "fixed-secret-value"
        finally:
            await listener.stop()


class TestModuleSingleton:
    async def test_listen_returns_same_listener(self):
        a = await listener_mod.listen(port=0, host="127.0.0.1")
        b = await listener_mod.listen(port=0, host="127.0.0.1")
        assert a is b
        assert listener_mod._active_listener() is a

    async def test_stop_clears_active(self):
        await listener_mod.listen(port=0, host="127.0.0.1")
        assert listener_mod._active_listener() is not None
        await listener_mod.stop()
        assert listener_mod._active_listener() is None

    async def test_stop_without_active_is_noop(self):
        await listener_mod.stop()

    async def test_active_listener_none_when_stopped(self):
        listener = await listener_mod.listen(port=0, host="127.0.0.1")
        assert listener_mod._active_listener() is listener
        await listener_mod.stop()
        assert listener_mod._active_listener() is None
