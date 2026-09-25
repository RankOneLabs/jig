import asyncio
import json
import time

import httpx
import pytest

from jig.jev import DEFAULT_ENDPOINT, JevClient, JevError, NoulQuestion


QUESTION = NoulQuestion("n", "Is it true?")
ANSWER = {
    "model": "jev-1.13.0", "request_id": "provider-123",
    "answers": {"n": {"type": "noul", "noul": 0.75}},
    "usage": {"input_tokens": 4, "output_tokens": 2},
}


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_missing_key_is_caller_error_before_io(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def forbidden(_request):
        pytest.fail("network request made without API key")

    async with mock_client(forbidden) as http:
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            JevClient(http=http)


async def test_invalid_questions_fail_before_io():
    def forbidden(_request):
        pytest.fail("network request made with invalid questions")

    async with mock_client(forbidden) as http:
        client = JevClient(api_key="key", http=http)
        with pytest.raises(ValueError, match="duplicate question id"):
            await client.evaluate({}, [QUESTION, QUESTION])


async def test_request_and_resolved_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=ANSWER)

    async with mock_client(handler) as http:
        client = JevClient(http=http)
        result = await client.evaluate({"post": "hello"}, [QUESTION])
        await client.aclose()
        await client.aclose()
        assert not http.is_closed
        await client.evaluate({}, [QUESTION])
    assert str(requests[0].url) == DEFAULT_ENDPOINT
    assert requests[0].headers["Authorization"] == "Bearer from-env"
    assert json.loads(requests[0].content) == {
        "state": {"post": "hello"}, "model": "jev-latest",
        "questions": {"n": {"type": "noul", "instructions": "Is it true?"}},
    }
    assert result.model == "jev-1.13.0"
    assert result.provider_request_id == "provider-123"
    assert result.call_id != result.provider_request_id
    assert result.attempts == 1 and result.latency_ms >= 0


async def test_endpoint_override_and_context_manager():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=ANSWER)

    async with mock_client(handler) as http:
        async with JevClient(api_key="key", endpoint="https://example.test/jev", http=http) as client:
            assert (await client.evaluate({}, [QUESTION])).model == "jev-1.13.0"
        assert not http.is_closed
    assert urls == ["https://example.test/jev"]


async def test_owned_context_closes_transport():
    async with JevClient(api_key="key") as client:
        assert not client._http.is_closed
    assert client._http.is_closed
    await client.aclose()


async def test_retry_success_has_one_call_and_elapsed_backoff():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(429) if calls == 1 else httpx.Response(200, json=ANSWER)

    async with mock_client(handler) as http:
        client = JevClient(api_key="key", http=http, max_retries=1)
        start = time.monotonic()
        result = await client.evaluate({}, [QUESTION])
    assert calls == 2
    assert result.attempts == 2
    assert result.latency_ms >= 90
    assert (time.monotonic() - start) * 1000 >= 90
    assert len(result.call_id) == 32


async def test_retry_respects_total_deadline():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    async with mock_client(handler) as http:
        client = JevClient(api_key="key", http=http, timeout=0.04, max_retries=100)
        start = time.monotonic()
        with pytest.raises(JevError) as caught:
            await client.evaluate({}, [QUESTION])
    assert caught.value.kind in ("rate_limited", "timeout")
    assert calls <= 2
    assert time.monotonic() - start < 0.12


async def test_cancel_in_flight_request_propagates():
    entered = asyncio.Event()

    async def handler(_request):
        entered.set()
        await asyncio.Event().wait()

    async with mock_client(handler) as http:
        client = JevClient(api_key="key", http=http)
        task = asyncio.create_task(client.evaluate({}, [QUESTION]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
