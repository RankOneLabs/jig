import httpx
import pytest

from jig.jev import JevClient, JevError, NoulQuestion


QUESTION = NoulQuestion("n", "check")


@pytest.mark.parametrize(
    ("case", "kind", "status"),
    [
        ("auth", "auth", 401),
        ("invalid_request", "invalid_request", 422),
        ("rate_limited", "rate_limited", 429),
        ("overloaded", "overloaded", 529),
        ("read_timeout", "timeout", None),
        ("connect_error", "transport", None),
        ("malformed_json", "invalid_response", 200),
        ("bad_contract", "invalid_response", 200),
    ],
)
async def test_failure_matrix(case, kind, status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if case == "read_timeout":
            raise httpx.ReadTimeout("secret request URL", request=request)
        if case == "connect_error":
            raise httpx.ConnectError("secret request URL", request=request)
        if case == "malformed_json":
            return httpx.Response(200, text="not JSON")
        if case == "bad_contract":
            return httpx.Response(200, json={"answers": {"n": {"type": "noul"}}})
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(api_key="key", http=http, max_retries=3, timeout=1)
        with pytest.raises(JevError) as caught:
            await client.evaluate({}, [QUESTION])
    error = caught.value
    assert error.kind == kind
    assert error.status_code == status
    assert error.attempts == calls
    assert error.elapsed_ms >= 0
    assert len(error.call_id) == 32
    if status in (401, 422):
        assert calls == 1
    elif status in (429, 529):
        assert calls == 4
    else:
        assert calls == 1
