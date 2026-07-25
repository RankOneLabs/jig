"""Tests for §2.3 (soft deadline) and §2.4 (RunControl) of the jig
lifecycle-hooks spec: AgentConfig.soft_deadline_s and the
context.control.request_finalize channel, and the shared loop-top
finalize-trigger mechanism that drives both.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

import jig.core.runner as runner_module
from jig import (
    AgentConfig,
    CompletionParams,
    LLMResponse,
    Message,
    MemoryEntry,
    Role,
    RunControl,
    Score,
    ScoreSource,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolRegistry,
    Usage,
    current_tool_context,
    run_agent,
)
from jig.core.runner import SUBMIT_OUTPUT_TOOL
from jig.core.types import (
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryStore,
    Retriever,
    Span,
    SpanKind,
    TracingLogger,
)

# --- Fakes (trimmed copies of the test_structured_output.py conventions) ---


class FakeLLM(LLMClient):
    """Replays a scripted list of responses. Records every CompletionParams it sees."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[CompletionParams] = []

    async def complete(self, params: CompletionParams) -> LLMResponse:
        self.calls.append(params)
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class FakeNativeLLM(FakeLLM):
    """A FakeLLM that declares response_format support, as a real adapter would."""

    supports_response_format = True


class FakeMemory(MemoryStore, Retriever):
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        return "mem-1"

    async def get(self, id: str) -> MemoryEntry | None:
        return None

    async def all(self) -> list[MemoryEntry]:
        return []

    async def delete(self, id: str) -> None:
        pass

    async def retrieve(self, query: str, k: int = 5, context=None) -> list[MemoryEntry]:
        return []

    async def get_session(self, session_id: str) -> list[Message]:
        return []

    async def add_to_session(self, session_id: str, message: Message) -> None:
        pass

    async def clear(self, session_id=None, before=None) -> None:
        pass


class FakeFeedback(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None):
        return "r-0"

    async def score(self, result_id: str, scores: list[Score]) -> None:
        pass

    async def get_signals(self, query, limit=3, min_score=None, source=None):
        return []

    async def query(self, q):
        return []

    async def export_eval_set(self, since=None, min_score=None, max_score=None, limit=None):
        return []


class FakeTracer(TracingLogger):
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):
        s = Span(id="trace-0", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), metadata=metadata)
        self.spans.append(s)
        return s

    def start_span(self, parent_id, kind, name, input=None, metadata=None):
        s = Span(
            id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name,
            started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata,
        )
        self.spans.append(s)
        return s

    def end_span(self, span_id, output=None, error=None, usage=None):
        for s in self.spans:
            if s.id == span_id:
                s.ended_at = datetime.now()
                s.output = output
                s.error = error

    async def get_trace(self, trace_id):
        return [s for s in self.spans if s.trace_id == trace_id]

    async def list_traces(self, since=None, limit=50, name=None):
        return []

    async def flush(self) -> None:
        pass


# --- Test schema ---


class Output(BaseModel):
    note: str = ""


def _config(
    llm: LLMClient,
    *,
    output_schema: type[BaseModel] | None = None,
    max_parse_retries: int = 2,
    tools: list | None = None,
    structured_output_mode: str = "legacy",
    soft_deadline_s: float | None = None,
    grader: Grader | None = None,
) -> AgentConfig:
    return AgentConfig(
        name="test",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        store=FakeMemory(), retriever=None,
        feedback=FakeFeedback(),
        tracer=FakeTracer(),
        tools=ToolRegistry(tools or []),
        output_schema=output_schema,
        max_parse_retries=max_parse_retries,
        structured_output_mode=structured_output_mode,
        soft_deadline_s=soft_deadline_s,
        grader=grader,
    )


def _submit_response(args: dict[str, Any], call_id: str = "so-1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=SUBMIT_OUTPUT_TOOL, arguments=args)],
        usage=Usage(10, 10),
        latency_ms=10,
        model="fake",
    )


def _content_response(payload: dict[str, Any] | str) -> LLMResponse:
    import json
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMResponse(content=content, tool_calls=None, usage=Usage(10, 10), latency_ms=10, model="fake")


def _tool_call_response(name: str, call_id: str, args: dict[str, Any] | None = None) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args or {})],
        usage=Usage(1, 1), latency_ms=1, model="fake",
    )


class _FinalizeTool:
    """Requests finalize via context.control on its first execution."""

    def __init__(self, reason: str = "done") -> None:
        self._reason = reason
        self.calls = 0
        self.seen_contexts: list[ToolExecutionContext | None] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="do_thing",
            description="does a thing, then requests finalize",
            parameters={"type": "object", "properties": {}},
        )

    async def execute_with_context(
        self, args: dict[str, Any], context: ToolExecutionContext | None = None,
    ) -> str:
        self.calls += 1
        self.seen_contexts.append(context)
        if self.calls == 1:
            assert context is not None
            assert context.control is not None
            context.control.request_finalize(self._reason)
        return f"did thing #{self.calls}"

    async def execute(self, args: dict[str, Any]) -> str:  # pragma: no cover
        return "unused"


class _Echo:
    """An ordinary working tool with no RunControl involvement."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo", description="echoes", parameters={"type": "object", "properties": {}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return "echoed"


# --- RunControl unit tests ---


class TestRunControlUnit:
    def test_starts_with_no_reason(self):
        assert RunControl().finalize_reason is None

    def test_request_finalize_sets_reason(self):
        control = RunControl()
        control.request_finalize("attempt terminal")
        assert control.finalize_reason == "attempt terminal"

    def test_second_request_is_a_noop_first_reason_wins(self):
        control = RunControl()
        control.request_finalize("first")
        control.request_finalize("second")
        assert control.finalize_reason == "first"

    def test_has_slots_no_dict(self):
        """Deliberately not a dataclass: __slots__, no __dict__, mutable."""
        control = RunControl()
        assert not hasattr(control, "__dict__")
        with pytest.raises(AttributeError):
            control.something_else = 1  # type: ignore[attr-defined]


class TestToolExecutionContextControlField:
    def test_defaults_to_none(self):
        ctx = ToolExecutionContext(
            trace_id="t", span_id="s", parent_span_id=None, tool_call_id="c",
        )
        assert ctx.control is None

    def test_accepts_run_control_instance(self):
        control = RunControl()
        ctx = ToolExecutionContext(
            trace_id="t", span_id="s", parent_span_id=None, tool_call_id="c", control=control,
        )
        assert ctx.control is control


# --- AgentConfig.soft_deadline_s validation ---


class TestSoftDeadlineValidation:
    def test_defaults_to_none(self):
        assert _config(FakeLLM([])).soft_deadline_s is None

    def test_accepts_positive_value(self):
        assert _config(FakeLLM([]), soft_deadline_s=30.0).soft_deadline_s == 30.0

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="soft_deadline_s"):
            _config(FakeLLM([]), soft_deadline_s=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="soft_deadline_s"):
            _config(FakeLLM([]), soft_deadline_s=-1.0)


# --- Finalize via RunControl, across all three structured-output modes ---


class TestFinalizeAcrossModes:
    """A tool calling context.control.request_finalize('done') on its first
    execution must cause the run to perform exactly one more LLM turn (the
    finalize turn) and terminate with valid structured output — in every
    structured_output_mode. Legacy is the trap case: tools=None would make
    submit_output uncallable, so the loop must keep offering it while
    dropping ordinary working tools.
    """

    async def test_legacy_mode(self):
        tool = _FinalizeTool()
        llm = FakeLLM([
            _tool_call_response("do_thing", "c1"),
            _submit_response({"note": "wrapped up"}),
        ])
        result = await run_agent(
            _config(llm, output_schema=Output, tools=[tool], structured_output_mode="legacy"),
            "go",
        )

        assert tool.calls == 1
        assert len(llm.calls) == 2  # exactly one more LLM turn after the tool call
        assert result.parsed is not None
        assert result.parsed.note == "wrapped up"
        assert result.error is None
        assert result.finalize_reason == "done"

        # The finalize turn still offers submit_output (the legacy trap:
        # tools=None here would make structured output unreachable) but no
        # longer offers the ordinary working tool.
        finalize_tools = [t.name for t in (llm.calls[1].tools or [])]
        assert SUBMIT_OUTPUT_TOOL in finalize_tools
        assert "do_thing" not in finalize_tools

        # Ordering guarantee: the tool's own result is in messages before
        # the finalize nudge that follows it.
        finalize_messages = llm.calls[1].messages
        tool_result_idx = next(
            i for i, m in enumerate(finalize_messages)
            if m.role == Role.TOOL and m.content == "did thing #1"
        )
        nudge_idx = next(
            i for i, m in enumerate(finalize_messages)
            if m.role == Role.USER and "done" in m.content and SUBMIT_OUTPUT_TOOL in m.content
        )
        assert tool_result_idx < nudge_idx

    async def test_plain_text_mode(self):
        """No output_schema at all: withholding tools is what ends the run,
        but the model still gets told why so its final answer accounts for
        the interruption."""
        tool = _FinalizeTool()
        llm = FakeLLM([
            _tool_call_response("do_thing", "c1"),
            _content_response("wrapped up"),
        ])
        result = await run_agent(_config(llm, tools=[tool]), "go")

        assert tool.calls == 1
        assert len(llm.calls) == 2
        assert result.output == "wrapped up"
        assert result.error is None
        assert result.finalize_reason == "done"
        assert llm.calls[1].tools is None
        assert any(
            m.role == Role.USER and "done" in m.content
            and "final answer" in m.content
            for m in llm.calls[1].messages
        )

    async def test_native_mode(self):
        tool = _FinalizeTool()
        llm = FakeNativeLLM([
            _tool_call_response("do_thing", "c1"),
            _content_response({"note": "wrapped up"}),
        ])
        result = await run_agent(
            _config(llm, output_schema=Output, tools=[tool], structured_output_mode="native"),
            "go",
        )

        assert tool.calls == 1
        assert len(llm.calls) == 2
        assert result.parsed is not None
        assert result.parsed.note == "wrapped up"
        assert result.error is None
        assert result.finalize_reason == "done"

        # Native: no synthetic tool: withhold every tool on the finalize turn.
        assert llm.calls[1].tools is None
        # response_format stays attached (native always sends it).
        assert llm.calls[1].response_format is not None
        # The reason reaches the model, not just the log and AgentResult: the
        # run is ending for a cause the model did not choose.
        assert any(
            m.role == Role.USER and "done" in m.content
            and "structured output" in m.content
            for m in llm.calls[1].messages
        )

    async def test_native_two_phase_mode(self):
        tool = _FinalizeTool()
        llm = FakeNativeLLM([
            _tool_call_response("do_thing", "c1"),
            _content_response({"note": "wrapped up"}),
        ])
        result = await run_agent(
            _config(
                llm, output_schema=Output, tools=[tool],
                structured_output_mode="native_two_phase",
            ),
            "go",
        )

        assert tool.calls == 1
        assert len(llm.calls) == 2  # jumps straight to the finalize call
        assert result.parsed is not None
        assert result.parsed.note == "wrapped up"
        assert result.error is None
        assert result.finalize_reason == "done"

        # Reuses the existing finalize_pending machinery: tools=None,
        # schema-constrained response_format on the finalize turn.
        assert llm.calls[1].tools is None
        assert llm.calls[1].response_format is not None
        # Working turn stayed unconstrained.
        assert llm.calls[0].response_format is None
        # Setting finalize_pending alone would terminate the run but drop the
        # reason on the floor — in this mode the natural finalize path sends
        # its own instruction, so the triggered path must too. This is the
        # mode gecko's sweep runs in, where the reason ("attempt concluded,
        # submit kind='stopped'") is the whole point of the request.
        assert any(
            m.role == Role.USER and "done" in m.content
            and "structured output" in m.content
            for m in llm.calls[1].messages
        )


class TestFinalizeLogging:
    async def test_logs_finalize_reason_at_info(self, caplog: pytest.LogCaptureFixture):
        tool = _FinalizeTool(reason="attempt validation_rejected")
        llm = FakeLLM([
            _tool_call_response("do_thing", "c1"),
            _submit_response({"note": "x"}),
        ])
        with caplog.at_level("INFO", logger="jig.core.runner"):
            await run_agent(
                _config(llm, output_schema=Output, tools=[tool]),
                "go",
            )

        assert any(
            "finalize requested" in r.message and "attempt validation_rejected" in r.message
            for r in caplog.records
        )


# --- Idempotency at the loop level: two calls in the same turn ---


class TestIdempotentAcrossLoop:
    async def test_two_finalize_requests_in_one_turn_first_reason_wins(self):
        class _TwoShotTool:
            def __init__(self) -> None:
                self.calls = 0

            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="two_shot", description="x", parameters={"type": "object", "properties": {}},
                )

            async def execute_with_context(self, args, context=None):
                self.calls += 1
                context.control.request_finalize(f"reason-{self.calls}")
                return f"call-{self.calls}"

            async def execute(self, args):  # pragma: no cover
                return "unused"

        tool = _TwoShotTool()
        # Both tool calls happen within the SAME turn's tool-dispatch loop,
        # before the next loop-top finalize check — so both requests land
        # before the loop even notices finalize_reason is set.
        double_call = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="a", name="two_shot", arguments={}),
                ToolCall(id="b", name="two_shot", arguments={}),
            ],
            usage=Usage(1, 1), latency_ms=1, model="fake",
        )
        llm = FakeLLM([double_call, _submit_response({"note": "ok"})])
        result = await run_agent(
            _config(llm, output_schema=Output, tools=[tool]),
            "go",
        )

        assert tool.calls == 2
        assert result.finalize_reason == "reason-1"


# --- Backward compatibility: no finalize request at all ---


class TestNoFinalizeUnchanged:
    async def test_run_without_request_finalize_is_unchanged(self):
        tool = _Echo()
        llm = FakeLLM([
            _tool_call_response("echo", "c1"),
            _submit_response({"note": "done"}),
        ])
        result = await run_agent(
            _config(llm, output_schema=Output, tools=[tool]),
            "go",
        )

        assert result.finalize_reason is None
        assert result.parsed is not None
        assert result.parsed.note == "done"
        # Every turn kept offering the full tool list (echo + submit_output);
        # no restriction and no nudge were ever injected.
        for params in llm.calls:
            names = [t.name for t in (params.tools or [])]
            assert "echo" in names
            assert SUBMIT_OUTPUT_TOOL in names

    async def test_plain_text_run_without_schema_or_deadline_unchanged(self):
        llm = FakeLLM([
            LLMResponse(content="hello", tool_calls=None, usage=Usage(5, 5), latency_ms=1, model="fake"),
        ])
        result = await run_agent(_config(llm), "hi")

        assert result.output == "hello"
        assert result.finalize_reason is None
        assert result.error is None


# --- Soft deadline ---


class TestSoftDeadline:
    async def test_deadline_stops_new_tool_turns_but_finalize_turn_completes(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """No RunControl call at all here — purely wall-clock. The tool
        call already in flight when the deadline is set completes; the
        very next turn is denied ordinary tools and instead lands
        submit_output."""
        times = iter([100.0, 100.0, 200.0, 200.0, 200.0, 200.0])

        def fake_time() -> float:
            try:
                return next(times)
            except StopIteration:
                return 200.0

        monkeypatch.setattr(runner_module.time, "time", fake_time)

        tool = _Echo()
        llm = FakeLLM([
            _tool_call_response("echo", "c1"),
            _submit_response({"note": "wrapped up"}),
        ])
        result = await run_agent(
            _config(llm, output_schema=Output, tools=[tool], soft_deadline_s=5.0),
            "go",
        )

        assert len(llm.calls) == 2
        assert result.parsed is not None
        assert result.parsed.note == "wrapped up"
        assert result.error is None
        assert result.finalize_reason == "soft deadline exceeded"

        # First (in-flight) turn still saw the ordinary tool.
        first_tools = [t.name for t in (llm.calls[0].tools or [])]
        assert "echo" in first_tools
        # Second turn: deadline already exceeded — only submit_output.
        second_tools = [t.name for t in (llm.calls[1].tools or [])]
        assert "echo" not in second_tools
        assert SUBMIT_OUTPUT_TOOL in second_tools

    async def test_deadline_not_exceeded_leaves_run_unaffected(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        times = iter([100.0, 100.0, 100.0, 100.0])

        def fake_time() -> float:
            try:
                return next(times)
            except StopIteration:
                return 100.0

        monkeypatch.setattr(runner_module.time, "time", fake_time)

        llm = FakeLLM([_submit_response({"note": "fine"})])
        result = await run_agent(
            _config(llm, output_schema=Output, soft_deadline_s=1000.0),
            "go",
        )

        assert result.finalize_reason is None
        assert result.parsed is not None
