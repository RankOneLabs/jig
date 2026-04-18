"""Tests for the smithers DispatchClient."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from jig.core.errors import JigLLMError
from jig.core.types import CompletionParams, Message, Role, ToolDefinition
from jig.llm.dispatch import DispatchClient


def _mock_submit_response(job_id: str = "job-123", status: str = "queued"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"job_id": job_id, "status": status}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_poll_response(
    job_id: str = "job-123",
    status: str = "complete",
    result: dict | None = None,
    model: str | None = "llama-70b",
    error: str | None = None,
):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "job_id": job_id,
        "status": status,
        "result": result,
        "model": model,
        "error": error,
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestBuildPayload:
    """Tests for _build_payload conversion."""

    def test_basic_message(self):
        client = DispatchClient(model="llama-70b")
        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hello")],
        )
        payload = client._build_payload(params)

        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert "temperature" not in payload
        assert "max_tokens" not in payload

    def test_system_prompt(self):
        client = DispatchClient()
        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
            system="You are helpful.",
        )
        payload = client._build_payload(params)

        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    def test_temperature_and_max_tokens(self):
        client = DispatchClient()
        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
            temperature=0.7,
            max_tokens=1024,
        )
        payload = client._build_payload(params)

        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1024

    def test_tools_included_in_payload(self):
        """Phase 7+8: tools pass through in OpenAI-compatible shape."""
        client = DispatchClient()
        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
            tools=[ToolDefinition(
                name="echo",
                description="echoes input",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            )],
        )
        payload = client._build_payload(params)
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "echo"
        assert payload["tools"][0]["function"]["description"] == "echoes input"
        assert payload["tools"][0]["function"]["parameters"]["type"] == "object"

    def test_skips_system_role_in_messages(self):
        client = DispatchClient()
        params = CompletionParams(
            messages=[
                Message(role=Role.SYSTEM, content="system msg"),
                Message(role=Role.USER, content="user msg"),
            ],
            system="Actual system prompt",
        )
        payload = client._build_payload(params)

        # System role message should be skipped; system param used instead
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user"]
        assert payload["messages"][0]["content"] == "Actual system prompt"


class TestComplete:
    """Tests for the complete() method."""

    @pytest.mark.asyncio
    async def test_submit_and_poll_complete(self):
        """Happy path: submit job, poll once, get result."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={"content": "Hello world"},
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)

        assert response.content == "Hello world"
        assert response.model == "llama-70b"
        assert response.latency_ms > 0
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_poll_through_intermediate_states(self):
        """Should keep polling through queued/running states."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.side_effect = [
            _mock_poll_response(status="queued", result=None),
            _mock_poll_response(status="running", result=None),
            _mock_poll_response(status="complete", result={"content": "Done"}),
        ]

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)

        assert response.content == "Done"
        assert client._http.get.call_count == 3

    @pytest.mark.asyncio
    async def test_job_failed(self):
        """Should raise JigLLMError on job failure."""
        client = DispatchClient(poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            status="failed", error="Model not available",
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        with pytest.raises(JigLLMError, match="Model not available"):
            await client.complete(params)

    @pytest.mark.asyncio
    async def test_job_cancelled(self):
        """Should raise JigLLMError on cancellation."""
        client = DispatchClient(poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(status="cancelled")

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        with pytest.raises(JigLLMError, match="cancelled"):
            await client.complete(params)

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Should raise JigLLMError after timeout."""
        client = DispatchClient(
            timeout_seconds=0.1, poll_interval=0.01, poll_max_interval=0.02,
        )

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(status="running", result=None)

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        with pytest.raises(JigLLMError, match="timed out"):
            await client.complete(params)

    @pytest.mark.asyncio
    async def test_submit_connect_error(self):
        """Should raise JigLLMError on connection failure."""
        client = DispatchClient()

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.side_effect = httpx.ConnectError("Connection refused")

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        with pytest.raises(JigLLMError, match="Cannot reach dispatch"):
            await client.complete(params)

    @pytest.mark.asyncio
    async def test_submit_http_error(self):
        """Should raise JigLLMError on HTTP error from dispatch."""
        client = DispatchClient()

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.text = "Bad request"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=resp,
        )
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = resp

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        with pytest.raises(JigLLMError, match="submission failed"):
            await client.complete(params)

    @pytest.mark.asyncio
    async def test_model_in_submission(self):
        """Should include model and machine in job submission."""
        client = DispatchClient(model="qwen-72b", machine="frink", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={"content": "ok"}, model="qwen-72b",
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        await client.complete(params)

        # Check the submission payload
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["model"] == "qwen-72b"
        assert body["machine"] == "frink"
        assert body["task_type"] == "inference"
        assert body["requester"] == "jig"

    @pytest.mark.asyncio
    async def test_cost_from_result_usage(self):
        """Dispatch should surface usage (tokens + cost) when smithers reports it."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={
                "content": "hi",
                "usage": {"input_tokens": 42, "output_tokens": 17, "cost": 0.0},
            },
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)
        assert response.usage.input_tokens == 42
        assert response.usage.output_tokens == 17
        assert response.usage.cost == 0.0

    @pytest.mark.asyncio
    async def test_usage_defaults_when_result_lacks_it(self):
        """Missing smithers usage → tokens=0, cost=None (unknown, not free)."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={"content": "hi"},
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        # None preserves "unknown" vs. 0.0 which would mean "confirmed free";
        # BudgetTracker deliberately ignores None-cost usages.
        assert response.usage.cost is None

    @pytest.mark.asyncio
    async def test_usage_tolerates_malformed_fields(self):
        """Non-numeric or null token/cost fields don't crash the adapter."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={
                "content": "hi",
                "usage": {
                    "input_tokens": None,
                    "output_tokens": "not-a-number",
                    "cost": "garbage",
                },
            },
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        assert response.usage.cost is None

    @pytest.mark.asyncio
    async def test_usage_non_finite_cost_rejected(self):
        """NaN/Inf cost (from string inputs or otherwise) becomes None."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={
                "content": "hi",
                "usage": {"input_tokens": 1, "output_tokens": 1, "cost": "nan"},
            },
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)
        # "nan" parses to float('nan') which is non-finite → treated as unknown.
        assert response.usage.cost is None

    @pytest.mark.asyncio
    async def test_aclose_releases_http(self):
        """aclose closes the underlying httpx client."""
        client = DispatchClient(model="llama-70b")
        client._http = AsyncMock(spec=httpx.AsyncClient)

        await client.aclose()
        client._http.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_usage_numeric_string_cost_accepted(self):
        """Cost reported as a numeric string coerces cleanly to float."""
        client = DispatchClient(model="llama-70b", poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={
                "content": "hi",
                "usage": {"input_tokens": "10", "output_tokens": "5", "cost": "0.125"},
            },
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        response = await client.complete(params)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.cost == 0.125

    @pytest.mark.asyncio
    async def test_no_model_omits_field(self):
        """Should omit model/machine from submission when not specified."""
        client = DispatchClient(poll_interval=0.01)

        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.post.return_value = _mock_submit_response()
        client._http.get.return_value = _mock_poll_response(
            result={"content": "ok"}, model=None,
        )

        params = CompletionParams(
            messages=[Message(role=Role.USER, content="Hi")],
        )
        await client.complete(params)

        body = client._http.post.call_args.kwargs.get("json") or client._http.post.call_args[1].get("json")
        assert "model" not in body
        assert "machine" not in body
