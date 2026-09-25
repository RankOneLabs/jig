import traceback

import httpx
import pytest

from jig.jev import JevClient, JevError, NoulQuestion


KEY = "sk-SENTINEL-DO-NOT-LEAK"
POST = "SENTINEL-POST-BODY-DO-NOT-LEAK"


@pytest.mark.parametrize("case", [
    "auth", "invalid_request", "rate_limited", "overloaded",
    "read_timeout", "connect_error", "malformed_json", "bad_contract",
])
async def test_errors_never_reveal_key_or_post(case, caplog):
    def handler(request):
        if case == "read_timeout":
            raise httpx.ReadTimeout(f"{KEY} {POST}", request=request)
        if case == "connect_error":
            raise httpx.ConnectError(f"{KEY} {POST}", request=request)
        if case == "malformed_json":
            return httpx.Response(200, text=f"{{{KEY} {POST}")
        if case == "bad_contract":
            return httpx.Response(200, json={"answers": {POST: {"type": "noul"}},
                                             "request_id": KEY})
        return httpx.Response({"auth": 401, "invalid_request": 422,
                               "rate_limited": 429, "overloaded": 529}[case],
                              text=f"echo {KEY} {POST}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(api_key=KEY, http=http)
        with pytest.raises(JevError) as caught:
            await client.evaluate({"post": POST}, [NoulQuestion("n", "check")])
    error = caught.value
    visible = "\n".join((str(error), repr(error),
                         "".join(traceback.format_exception(error)), caplog.text))
    assert KEY not in visible
    assert POST not in visible
