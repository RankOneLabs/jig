"""Tests for ``jig.replay`` — recorded-trace replay with canned tool outputs."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from jig import (
    AgentConfig,
    ReplayConfigMissingError,
    ReplayMissError,
    ReplaySchemaMismatchError,
    Span,
    SpanKind,
    ToolCall,
    ToolDefinition,
    ToolResult,
    replay,
    run_agent,
)
from jig.core.types import (
    CompletionParams,
    EvalCase,
    FeedbackLoop,
    LLMClient,
    LLMResponse,
    MemoryEntry,
    MemoryStore,
    Message,
    Retriever,
    Score,
    ScoreSource,
    ScoredResult,
    Tool,
    Usage,
)
from jig.replay.registry import ReplayToolRegistry, _canonical_args
from jig.replay.snapshot import reconstruct_config, serialize_config
from jig.tools import ToolRegistry
from jig.tracing import SQLiteTracer


# --- Inline fakes (each tests module keeps its own for isolation) ---


class FakeLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, params: CompletionParams) -> LLMResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class FakeMemory(MemoryStore, Retriever):
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        return "mem-1"

    async def get(self, id: str) -> MemoryEntry | None:  # noqa: A002
        return None

    async def all(self) -> list[MemoryEntry]:
        return []

    async def delete(self, id: str) -> None:  # noqa: A002
        pass

    async def retrieve(self, query: str, k: int = 5, context: dict[str, Any] | None = None) -> list[MemoryEntry]:
        return []

    async def get_session(self, session_id: str) -> list[Message]:
        return []

    async def add_to_session(self, session_id: str, message: Message) -> None:
        pass

    async def clear(self, session_id: str | None = None, before: datetime | None = None) -> None:
        pass


class FakeFeedback(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None):
        return "r-0"

    async def score(self, result_id: str, scores: list[Score]) -> None:
        pass

    async def get_signals(self, query: str, limit: int = 3, min_score: float | None = None, source: ScoreSource | None = None) -> list[ScoredResult]:
        return []

    async def query(self, q):
        return []

    async def export_eval_set(self, since: datetime | None = None, min_score: float | None = None, max_score: float | None = None, limit: int | None = None) -> list[EvalCase]:
        return []


class EchoTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echoes input",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return args.get("text", "")


# --- Helpers ---


def _make_config(
    llm: FakeLLM,
    tracer: SQLiteTracer,
    *,
    tools: ToolRegistry | None = None,
    output_schema: type[BaseModel] | None = None,
    system_prompt: str = "You are a test agent.",
    max_tool_calls: int = 10,
    name: str = "test",
) -> AgentConfig[Any]:
    return AgentConfig(
        name=name,
        description="test agent",
        system_prompt=system_prompt,
        llm=llm,
        store=FakeMemory(),
        retriever=None,
        feedback=FakeFeedback(),
        tracer=tracer,
        tools=tools if tools is not None else ToolRegistry([EchoTool()]),
        max_tool_calls=max_tool_calls,
        output_schema=output_schema,
    )


def _llm_with_echo_then_final(
    text_to_echo: str = "hello",
    final: str = "done",
) -> FakeLLM:
    return FakeLLM([
        LLMResponse(
            content="",
            tool_calls=[ToolCall(
                id="tc-1", name="echo", arguments={"text": text_to_echo},
            )],
            usage=Usage(5, 5),
            latency_ms=1,
            model="fake",
        ),
        LLMResponse(
            content=final,
            tool_calls=None,
            usage=Usage(5, 5),
            latency_ms=1,
            model="fake",
        ),
    ])


@pytest.fixture
def tracer(tmp_path: Any) -> SQLiteTracer:
    return SQLiteTracer(db_path=str(tmp_path / "traces.db"))


# --- Snapshot ---


def test_serialize_config_captures_state_fields():
    class DummyConfig(AgentConfig):
        pass

    # Minimal construction via the shared helper
    tracer = SQLiteTracer(db_path=":memory:")
    cfg = _make_config(
        FakeLLM([]), tracer, system_prompt="rock solid", max_tool_calls=3,
        name="snap",
    )
    snap = serialize_config(cfg)

    assert snap["agent_name"] == "snap"
    assert snap["system_prompt"] == "rock solid"
    assert snap["system_prompt_is_callable"] is False
    assert snap["max_tool_calls"] == 3
    assert snap["output_schema"] is None


def test_serialize_config_with_callable_system_prompt_marked():
    def build_prompt() -> str:
        return "dynamic"

    tracer = SQLiteTracer(db_path=":memory:")
    cfg = _make_config(FakeLLM([]), tracer, system_prompt="placeholder")
    cfg = cfg.with_(system_prompt=build_prompt)
    snap = serialize_config(cfg)

    assert snap["system_prompt_is_callable"] is True
    assert snap["system_prompt"] is None


class _SchemaA(BaseModel):
    field_a: str


class _SchemaB(BaseModel):
    field_b: int


def test_serialize_config_encodes_output_schema_fqn():
    tracer = SQLiteTracer(db_path=":memory:")
    cfg = _make_config(FakeLLM([]), tracer, output_schema=_SchemaA)
    snap = serialize_config(cfg)

    fqn = snap["output_schema"]
    assert fqn is not None
    assert ":" in fqn
    from jig.replay.snapshot import _resolve_output_schema
    assert _resolve_output_schema(fqn) is _SchemaA


def test_reconstruct_config_round_trip():
    tracer = SQLiteTracer(db_path=":memory:")
    cfg = _make_config(FakeLLM([]), tracer, max_tool_calls=7, name="rt")
    snap = serialize_config(cfg)

    new = reconstruct_config(
        snap,
        None,
        llm=FakeLLM([]),
        tools=ToolRegistry(),
        tracer=tracer,
        feedback=FakeFeedback(),
    )
    assert new.name == "rt"
    assert new.max_tool_calls == 7
    assert new.system_prompt == cfg.system_prompt


def test_reconstruct_config_override_applies():
    tracer = SQLiteTracer(db_path=":memory:")
    cfg = _make_config(FakeLLM([]), tracer, max_tool_calls=7)
    snap = serialize_config(cfg)

    new = reconstruct_config(
        snap,
        {"max_tool_calls": 2},
        llm=FakeLLM([]),
        tools=ToolRegistry(),
        tracer=tracer,
        feedback=FakeFeedback(),
    )
    assert new.max_tool_calls == 2


def test_reconstruct_config_callable_prompt_requires_override():
    snap = {"agent_name": "x", "description": "", "system_prompt": None,
            "system_prompt_is_callable": True, "max_tool_calls": 10,
            "max_llm_calls": 50, "max_llm_retries": 3,
            "max_parse_retries": 2, "include_memory_in_prompt": True,
            "include_feedback_in_prompt": True, "session_id": None,
            "output_schema": None}

    with pytest.raises(ReplaySchemaMismatchError, match="callable"):
        reconstruct_config(
            snap, None,
            llm=FakeLLM([]),
            tools=ToolRegistry(),
            tracer=SQLiteTracer(db_path=":memory:"),
            feedback=FakeFeedback(),
        )

    # Supplying a new prompt satisfies the requirement
    new = reconstruct_config(
        snap, {"system_prompt": "fresh"},
        llm=FakeLLM([]),
        tools=ToolRegistry(),
        tracer=SQLiteTracer(db_path=":memory:"),
        feedback=FakeFeedback(),
    )
    assert new.system_prompt == "fresh"


def test_reconstruct_config_schema_override_mismatch_rejected():
    snap = {
        "agent_name": "x", "description": "", "system_prompt": "p",
        "system_prompt_is_callable": False, "max_tool_calls": 10,
        "max_llm_calls": 50, "max_llm_retries": 3, "max_parse_retries": 2,
        "include_memory_in_prompt": True, "include_feedback_in_prompt": True,
        "session_id": None,
        "output_schema": f"{_SchemaA.__module__}:{_SchemaA.__qualname__}",
    }
    with pytest.raises(ReplaySchemaMismatchError, match="out of replay scope"):
        reconstruct_config(
            snap, {"output_schema": _SchemaB},
            llm=FakeLLM([]),
            tools=ToolRegistry(),
            tracer=SQLiteTracer(db_path=":memory:"),
            feedback=FakeFeedback(),
        )


def test_reconstruct_config_schema_fqn_missing_raises():
    snap = {
        "agent_name": "x", "description": "", "system_prompt": "p",
        "system_prompt_is_callable": False, "max_tool_calls": 10,
        "max_llm_calls": 50, "max_llm_retries": 3, "max_parse_retries": 2,
        "include_memory_in_prompt": True, "include_feedback_in_prompt": True,
        "session_id": None,
        "output_schema": "nonexistent.module:NoSuchClass",
    }
    with pytest.raises(ReplaySchemaMismatchError, match="can't be imported"):
        reconstruct_config(
            snap, None,
            llm=FakeLLM([]),
            tools=ToolRegistry(),
            tracer=SQLiteTracer(db_path=":memory:"),
            feedback=FakeFeedback(),
        )


def test_reconstruct_config_override_rejects_adding_schema():
    """Adding an output_schema to a trace recorded without one is a
    semantic shift (flips whether submit_output is injected)."""
    snap = {
        "agent_name": "x", "description": "", "system_prompt": "p",
        "system_prompt_is_callable": False, "max_tool_calls": 10,
        "max_llm_calls": 50, "max_llm_retries": 3, "max_parse_retries": 2,
        "include_memory_in_prompt": True, "include_feedback_in_prompt": True,
        "session_id": None, "output_schema": None,
    }
    with pytest.raises(ReplaySchemaMismatchError, match="out of replay scope"):
        reconstruct_config(
            snap, {"output_schema": _SchemaA},
            llm=FakeLLM([]),
            tools=ToolRegistry(),
            tracer=SQLiteTracer(db_path=":memory:"),
            feedback=FakeFeedback(),
        )


def test_reconstruct_config_override_rejects_dropping_schema():
    """Removing the recorded output_schema is equally disallowed."""
    snap = {
        "agent_name": "x", "description": "", "system_prompt": "p",
        "system_prompt_is_callable": False, "max_tool_calls": 10,
        "max_llm_calls": 50, "max_llm_retries": 3, "max_parse_retries": 2,
        "include_memory_in_prompt": True, "include_feedback_in_prompt": True,
        "session_id": None,
        "output_schema": f"{_SchemaA.__module__}:{_SchemaA.__qualname__}",
    }
    with pytest.raises(ReplaySchemaMismatchError, match="out of replay scope"):
        reconstruct_config(
            snap, {"output_schema": None},
            llm=FakeLLM([]),
            tools=ToolRegistry(),
            tracer=SQLiteTracer(db_path=":memory:"),
            feedback=FakeFeedback(),
        )


def test_resolve_output_schema_rejects_function_local():
    """PEP 3155: <locals> in qualname means the class isn't reachable
    by getattr walk. Resolver surfaces a clear error instead of a
    generic AttributeError."""
    from jig.replay.snapshot import _resolve_output_schema

    with pytest.raises(ReplaySchemaMismatchError, match="function-local"):
        _resolve_output_schema("some.module:outer.<locals>.Schema")


@pytest.mark.parametrize("fqn", [":Schema", "os:", ":", "foo"])
def test_resolve_output_schema_rejects_malformed_fqn(fqn: str):
    """Empty module_name or qualname must raise ReplaySchemaMismatchError.

    Without the emptiness guard, ``importlib.import_module("")`` raises
    ``ValueError`` (not ``ImportError``) and bypasses the handler, leaking
    a raw exception instead of a typed replay error.
    """
    from jig.replay.snapshot import _resolve_output_schema

    with pytest.raises(ReplaySchemaMismatchError, match="malformed"):
        _resolve_output_schema(fqn)


# --- ReplayToolRegistry ---


def _tool_span(name: str, args: dict[str, Any], output: str, error: str | None = None) -> Span:
    return Span(
        id=f"s-{name}",
        trace_id="t-0",
        kind=SpanKind.TOOL_CALL,
        name=name,
        started_at=datetime.now(),
        input=args,
        output=output,
        error=error,
    )


async def test_replay_registry_strict_hit_returns_canned():
    recorded = [_tool_span("echo", {"text": "hi"}, "hi")]
    reg = ReplayToolRegistry(recorded, strict=True)

    out = await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "hi"}))
    assert out.output == "hi"
    assert out.error is None


async def test_replay_registry_strict_miss_raises():
    recorded = [_tool_span("echo", {"text": "a"}, "a")]
    reg = ReplayToolRegistry(recorded, strict=True)

    with pytest.raises(ReplayMissError):
        await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "z"}))


async def test_replay_registry_lenient_miss_falls_back():
    recorded = [_tool_span("echo", {"text": "a"}, "a")]
    fallback = ToolRegistry([EchoTool()])
    reg = ReplayToolRegistry(recorded, fallback=fallback, strict=False)

    out = await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "z"}))
    assert out.output == "z"  # came from live EchoTool
    assert out.error is None


async def test_replay_registry_lenient_miss_no_fallback_returns_error():
    reg = ReplayToolRegistry([], strict=False)

    out = await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "x"}))
    assert out.output == ""
    assert out.error is not None
    assert "Replay miss" in out.error


async def test_replay_registry_fifo_on_duplicate_keys():
    recorded = [
        _tool_span("echo", {"text": "x"}, "first"),
        _tool_span("echo", {"text": "x"}, "second"),
    ]
    reg = ReplayToolRegistry(recorded, strict=True)

    first = await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "x"}))
    second = await reg.execute(ToolCall(id="c-2", name="echo", arguments={"text": "x"}))
    assert first.output == "first"
    assert second.output == "second"

    with pytest.raises(ReplayMissError):
        await reg.execute(ToolCall(id="c-3", name="echo", arguments={"text": "x"}))


async def test_replay_registry_args_canonicalized_by_key_order():
    """`{a:1, b:2}` and `{b:2, a:1}` must hit the same cache entry."""
    recorded = [_tool_span("echo", {"a": 1, "b": 2}, "matched")]
    reg = ReplayToolRegistry(recorded, strict=True)

    out = await reg.execute(ToolCall(
        id="c-1", name="echo", arguments={"b": 2, "a": 1},
    ))
    assert out.output == "matched"


async def test_replay_registry_advertises_only_observed_tools():
    """``list()`` must only surface tools that appeared in the trace.

    Exposing fallback tools that never ran originally would let the
    replay model pick brand-new tools and diverge silently from the
    recording.
    """
    recorded = [_tool_span("echo", {"text": "x"}, "x")]
    fallback = ToolRegistry([EchoTool()])
    # Extra definition the fallback advertises that was never in the trace.
    extra = ToolDefinition(
        name="brand_new", description="not in trace",
        parameters={"type": "object", "properties": {}},
    )
    reg = ReplayToolRegistry(recorded, definitions=[
        ToolDefinition(
            name="echo", description="e",
            parameters={"type": "object", "properties": {}},
        ),
        extra,
    ], fallback=fallback)

    names = {d.name for d in reg.list()}
    assert names == {"echo"}


async def test_replay_registry_skips_submit_output_span():
    """``submit_output`` spans are runner-internal; replay must ignore them."""
    from jig.core.runner import SUBMIT_OUTPUT_TOOL

    recorded = [
        _tool_span(SUBMIT_OUTPUT_TOOL, {"result": "x"}, "{}"),
        _tool_span("echo", {"text": "hi"}, "hi"),
    ]
    reg = ReplayToolRegistry(recorded, strict=True)
    # Echo should still work; submit_output key must not have been indexed.
    out = await reg.execute(ToolCall(id="c-1", name="echo", arguments={"text": "hi"}))
    assert out.output == "hi"


# --- replay() end-to-end ---


async def test_replay_end_to_end_tool_outputs_pinned(tracer: SQLiteTracer):
    # First run: real LLM → real tool execution, output recorded.
    llm_a = _llm_with_echo_then_final(text_to_echo="hello", final="all done")
    result_a = await run_agent(
        _make_config(llm_a, tracer, name="echo-agent"),
        "please echo",
    )
    await tracer.flush()

    # Second run via replay: different LLM emits the *same* tool args but
    # decides something different for the final answer.
    llm_b = _llm_with_echo_then_final(text_to_echo="hello", final="reinterpreted")
    result_b = await replay(
        trace_id=result_a.trace_id,
        tracer=tracer,
        llm=llm_b,
        feedback=FakeFeedback(),
    )

    assert result_a.output == "all done"
    assert result_b.output == "reinterpreted"
    assert result_a.trace_id != result_b.trace_id

    # Replay's tool span carries the *canned* output, not a re-executed one.
    replay_spans = await tracer.get_trace(result_b.trace_id)
    tool_spans = [s for s in replay_spans if s.kind == SpanKind.TOOL_CALL]
    assert len(tool_spans) == 1
    assert tool_spans[0].output == "hello"


async def test_replay_honors_config_override(tracer: SQLiteTracer):
    llm = _llm_with_echo_then_final()
    result = await run_agent(_make_config(llm, tracer), "go")
    await tracer.flush()

    # Override max_tool_calls: replay cuts off early. Feed an LLM that
    # would want to call the tool again past the cap.
    tool_resp = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "hello"})],
        usage=Usage(1, 1), latency_ms=1, model="fake",
    )
    final = LLMResponse(content="stop", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake")
    replay_llm = FakeLLM([tool_resp, tool_resp, final, final])

    result_r = await replay(
        trace_id=result.trace_id,
        config_override={"max_tool_calls": 1},
        tracer=tracer,
        llm=replay_llm,
        feedback=FakeFeedback(),
    )
    assert result_r.usage["tool_calls"] == 1


async def test_replay_override_cannot_swap_tools(tracer: SQLiteTracer):
    """The core contract: replay must always use ``ReplayToolRegistry``.
    A caller passing ``tools`` in ``config_override`` (or a full
    AgentConfig replacement) must not defeat that — otherwise replay
    silently falls back to real tool execution.

    We use a live tool registry as the override and record what runs:
    if replay honored the override, the real EchoTool would emit a
    different output than what was recorded. The canned recording wins.
    """
    # Record: echo "hello" → recorded output "hello"
    llm_a = _llm_with_echo_then_final(text_to_echo="hello", final="done")
    result_a = await run_agent(_make_config(llm_a, tracer), "go")
    await tracer.flush()

    # Build a "live" registry whose EchoTool emits a sentinel; if replay
    # used it, the replay tool span output would be "LIVE-NOT-CANNED".
    class SentinelEcho(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="echo",
                description="sentinel",
                parameters={"type": "object", "properties": {}},
            )

        async def execute(self, args: dict[str, Any]) -> str:
            return "LIVE-NOT-CANNED"

    live_registry = ToolRegistry([SentinelEcho()])

    llm_b = _llm_with_echo_then_final(text_to_echo="hello", final="done-b")
    result_b = await replay(
        trace_id=result_a.trace_id,
        config_override={"tools": live_registry},  # tries to swap!
        tracer=tracer,
        llm=llm_b,
        feedback=FakeFeedback(),
    )

    replay_spans = await tracer.get_trace(result_b.trace_id)
    tool_spans = [s for s in replay_spans if s.kind == SpanKind.TOOL_CALL]
    assert len(tool_spans) == 1
    # Canned "hello" wins; SentinelEcho was NOT invoked.
    assert tool_spans[0].output == "hello"


async def test_replay_rejects_old_trace_without_snapshot(tracer: SQLiteTracer):
    """A pre-phase-11 trace (no config snapshot) surfaces a clear error."""
    # Plant a bare AGENT_RUN span without a config snapshot.
    span = tracer.start_trace("stale", metadata={"input": "hi"})
    tracer.end_span(span.id, {"output": "hi"})
    await tracer.flush()

    with pytest.raises(ReplayConfigMissingError, match="missing the config snapshot"):
        await replay(
            trace_id=span.trace_id,
            tracer=tracer,
            llm=FakeLLM([]),
            feedback=FakeFeedback(),
        )


async def test_replay_rejects_unknown_trace_id(tracer: SQLiteTracer):
    with pytest.raises(ReplayConfigMissingError, match="No spans"):
        await replay(
            trace_id="does-not-exist",
            tracer=tracer,
            llm=FakeLLM([]),
            feedback=FakeFeedback(),
        )


async def test_replay_strict_raises_on_tool_arg_drift(tracer: SQLiteTracer):
    llm = _llm_with_echo_then_final(text_to_echo="x", final="done")
    result = await run_agent(_make_config(llm, tracer), "go")
    await tracer.flush()

    # Replay with an LLM that emits *different* tool args — strict mode
    # surfaces that as a ReplayMissError.
    drift_llm = FakeLLM([
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-1", name="echo", arguments={"text": "y"})],
            usage=Usage(1, 1), latency_ms=1, model="fake",
        ),
        LLMResponse(content="n/a", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
    ])
    with pytest.raises(ReplayMissError):
        await replay(
            trace_id=result.trace_id,
            tracer=tracer,
            llm=drift_llm,
            feedback=FakeFeedback(),
            strict=True,
        )


# --- Edge cases ---


async def test_replay_trace_with_zero_tool_calls(tracer: SQLiteTracer):
    llm = FakeLLM([
        LLMResponse(content="plain", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
    ])
    result = await run_agent(_make_config(llm, tracer, tools=ToolRegistry()), "hi")
    await tracer.flush()

    replay_llm = FakeLLM([
        LLMResponse(content="plain-again", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
    ])
    result_r = await replay(
        trace_id=result.trace_id,
        tracer=tracer,
        llm=replay_llm,
        feedback=FakeFeedback(),
    )
    assert result_r.output == "plain-again"
    assert result_r.usage["tool_calls"] == 0


def test_canonical_args_sort_order_stable():
    a = _canonical_args({"a": 1, "b": 2})
    b = _canonical_args({"b": 2, "a": 1})
    assert a == b
