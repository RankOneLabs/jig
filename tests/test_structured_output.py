"""Tests for typed agent outputs (AgentConfig.output_schema, submit_output tool)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel, Field

from jig import (
    AgentConfig,
    AgentNativeOutputError,
    CompletionParams,
    LLMResponse,
    MemoryEntry,
    Message,
    Role,
    Score,
    ScoreSource,
    ToolCall,
    ToolDefinition,
    UnsupportedResponseFormatError,
    Usage,
    run_agent,
)
from jig.core.runner import SUBMIT_OUTPUT_TOOL, _normalize_strict_schema
from jig.core.types import (
    MemoryStore,
    Retriever,
    FeedbackLoop,
    Grader,
    LLMClient,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry
from jig.tracing import SQLiteTracer


# --- Fakes ---


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
    """A FakeLLM that declares response_format support, as a real adapter
    (OpenAI, Dispatch, Ollama) would."""

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
    def __init__(self) -> None:
        self.scored: list[tuple[str, list[Score]]] = []

    async def store_result(self, content, input_text, metadata=None):
        return "r-0"

    async def score(self, result_id: str, scores: list[Score]) -> None:
        self.scored.append((result_id, scores))

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
        s = Span(id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, input=input, metadata=metadata)
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


class StrategyOutput(BaseModel):
    strategy_types: list[str] = Field(description="Strategy type keywords")
    best_sharpe: float | None = None
    notes: str = ""


def _config(
    llm: LLMClient,
    *,
    output_schema: type[BaseModel] | None = None,
    grader: Grader | None = None,
    max_parse_retries: int = 2,
    tools: list | None = None,
    tracer: TracingLogger | None = None,
    structured_output_mode: str = "legacy",
) -> AgentConfig:
    return AgentConfig(
        name="test",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        store=FakeMemory(), retriever=None,
        feedback=FakeFeedback(),
        tracer=tracer if tracer is not None else FakeTracer(),
        tools=ToolRegistry(tools or []),
        output_schema=output_schema,
        max_parse_retries=max_parse_retries,
        grader=grader,
        structured_output_mode=structured_output_mode,
    )


def _submit_response(args: dict[str, Any], call_id: str = "so-1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=SUBMIT_OUTPUT_TOOL, arguments=args)],
        usage=Usage(10, 10),
        latency_ms=10,
        model="fake",
    )


# --- Tests ---


class TestHappyPath:
    async def test_model_calls_submit_output_once(self):
        """Valid args on first try → parsed populated, loop ends."""
        llm = FakeLLM([
            _submit_response({
                "strategy_types": ["mean_reversion", "breakout"],
                "best_sharpe": 1.5,
                "notes": "promising",
            }),
        ])
        result = await run_agent(_config(llm, output_schema=StrategyOutput), "go")

        assert isinstance(result.parsed, StrategyOutput)
        assert result.parsed.strategy_types == ["mean_reversion", "breakout"]
        assert result.parsed.best_sharpe == 1.5
        assert '"strategy_types"' in result.output  # JSON-serialized form

    async def test_submit_output_tool_included_in_llm_params(self):
        """Runner injects the submit_output tool on every turn."""
        llm = FakeLLM([
            _submit_response({"strategy_types": ["diffusion"]}),
        ])
        await run_agent(_config(llm, output_schema=StrategyOutput), "go")

        params = llm.calls[0]
        assert params.tools is not None
        tool_names = [t.name for t in params.tools]
        assert SUBMIT_OUTPUT_TOOL in tool_names

    async def test_schema_instruction_appended_to_system(self):
        """System prompt is extended with the schema-completion instruction."""
        llm = FakeLLM([
            _submit_response({"strategy_types": []}),
        ])
        await run_agent(_config(llm, output_schema=StrategyOutput), "go")

        system = llm.calls[0].system
        assert SUBMIT_OUTPUT_TOOL in system
        assert "final answer" in system.lower()


class TestValidationRetry:
    async def test_first_invalid_then_valid(self):
        """Invalid args trigger a retry with validation error in tool result."""
        llm = FakeLLM([
            _submit_response({"best_sharpe": "not a float"}, call_id="a"),  # missing required
            _submit_response({"strategy_types": ["ok"]}, call_id="b"),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=2),
            "go",
        )

        assert result.parsed is not None
        assert result.parsed.strategy_types == ["ok"]
        # Second call should have seen the prior failure in its messages
        second_params = llm.calls[1]
        tool_msgs = [m for m in second_params.messages if m.role == Role.TOOL]
        assert any("Validation failed" in m.content for m in tool_msgs)

    async def test_gives_up_after_max_retries(self):
        """Exceeding max_parse_retries leaves parsed=None and bails out."""
        # max_parse_retries=1 → runner allows the initial attempt + 1 retry;
        # the 2nd validation failure (parse_retries becomes 2 > 1) terminates.
        # Buffer 3 responses but only the first two will be consumed.
        llm = FakeLLM([
            _submit_response({}, call_id="a"),  # missing required field
            _submit_response({}, call_id="b"),
            _submit_response({}, call_id="c"),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=1),
            "go",
        )

        assert result.parsed is None
        assert "validation failed" in result.output.lower()

    async def test_model_ignores_submit_output_then_complies(self):
        """Plain-text response with schema set → nudge back, accept on retry."""
        llm = FakeLLM([
            LLMResponse(
                content="Here's my answer in prose.",
                tool_calls=None,
                usage=Usage(5, 5),
                latency_ms=1,
                model="fake",
            ),
            _submit_response({"strategy_types": ["x"]}),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=2),
            "go",
        )

        assert result.parsed is not None
        # Second call should include a nudge message
        second_params = llm.calls[1]
        nudges = [
            m for m in second_params.messages
            if m.role == Role.USER and SUBMIT_OUTPUT_TOOL in m.content
        ]
        assert len(nudges) >= 1


class TestAmbiguousTurn:
    """submit_output must be the only tool call in its turn, else other
    tool executions would vanish silently on termination."""

    async def test_submit_output_with_other_tool_calls_nudged(self):
        """Same-turn submit_output + other tools → reject, ask to retry."""
        # First turn: model combines a user tool + submit_output.
        # Second turn: model retries with submit_output alone.
        mixed = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "hi"}),
                ToolCall(id="t2", name=SUBMIT_OUTPUT_TOOL,
                         arguments={"strategy_types": ["x"]}),
            ],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        clean = _submit_response({"strategy_types": ["x"]})
        llm = FakeLLM([mixed, clean])

        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=2),
            "go",
        )

        # Terminated on the clean turn, not the mixed one
        assert result.parsed is not None
        assert result.parsed.strategy_types == ["x"]
        # Second turn's history must carry tool results for BOTH calls in
        # the mixed turn (providers reject orphan tool_use blocks).
        second = llm.calls[1]
        tool_msgs = [m for m in second.messages if m.role == Role.TOOL]
        assert any("only tool call" in m.content.lower() for m in tool_msgs)
        assert any("separate turn" in m.content.lower() for m in tool_msgs)

    async def test_multiple_submit_output_in_one_turn_nudged(self):
        """Two submit_output calls in a turn → treated as ambiguous."""
        doubled = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="a", name=SUBMIT_OUTPUT_TOOL,
                         arguments={"strategy_types": ["x"]}),
                ToolCall(id="b", name=SUBMIT_OUTPUT_TOOL,
                         arguments={"strategy_types": ["y"]}),
            ],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        clean = _submit_response({"strategy_types": ["ok"]})
        llm = FakeLLM([doubled, clean])

        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=2),
            "go",
        )
        assert result.parsed is not None
        assert result.parsed.strategy_types == ["ok"]

    async def test_ambiguous_turn_exhausts_budget(self):
        """Repeated ambiguous turns hit max_parse_retries and terminate."""
        mixed = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="echo", arguments={"text": "hi"}),
                ToolCall(id="t2", name=SUBMIT_OUTPUT_TOOL,
                         arguments={"strategy_types": ["x"]}),
            ],
            usage=Usage(1, 1),
            latency_ms=1,
            model="fake",
        )
        llm = FakeLLM([mixed] * 3)
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=1),
            "go",
        )
        assert result.parsed is None
        assert "combined" in result.output


class TestGraderIntegration:
    async def test_typed_grader_receives_parsed_model(self):
        """Grader[T] sees the pydantic instance, not the raw string."""
        seen: list[Any] = []

        class TypedGrader(Grader[StrategyOutput]):
            async def grade(self, input, output, context=None):
                seen.append(output)
                # raw_output must be in context for graders that want both
                assert context is not None
                assert "raw_output" in context
                return [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

        llm = FakeLLM([
            _submit_response({"strategy_types": ["a", "b"]}),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, grader=TypedGrader()),
            "go",
        )

        assert len(seen) == 1
        assert isinstance(seen[0], StrategyOutput)
        assert seen[0].strategy_types == ["a", "b"]
        assert result.scores == [Score(dimension="q", value=1.0, source=ScoreSource.HEURISTIC)]

    async def test_untyped_grader_receives_raw_when_no_schema(self):
        """Without a schema, grader receives the final content string."""
        seen: list[Any] = []

        class StrGrader(Grader[str]):
            async def grade(self, input, output, context=None):
                seen.append(output)
                return [Score(dimension="q", value=0.5, source=ScoreSource.HEURISTIC)]

        llm = FakeLLM([
            LLMResponse(content="plain answer", tool_calls=None, usage=Usage(5, 5), latency_ms=1, model="fake"),
        ])
        await run_agent(_config(llm, grader=StrGrader()), "go")

        assert seen == ["plain answer"]


class TestSchemaValidation:
    async def test_non_basemodel_schema_rejected(self):
        """Passing a plain class as output_schema raises TypeError."""
        class NotAModel:
            pass

        llm = FakeLLM([_submit_response({})])
        with pytest.raises(TypeError, match="BaseModel subclass"):
            await run_agent(
                _config(llm, output_schema=NotAModel),  # type: ignore[arg-type]
                "go",
            )


class TestBackwardCompatibility:
    async def test_no_schema_path_unchanged(self):
        """Without output_schema, behavior matches pre-phase-1 (no submit_output)."""
        llm = FakeLLM([
            LLMResponse(content="hello", tool_calls=None, usage=Usage(5, 5), latency_ms=1, model="fake"),
        ])
        result = await run_agent(_config(llm), "hi")

        assert result.output == "hello"
        assert result.parsed is None
        # No submit_output tool in the params
        params = llm.calls[0]
        if params.tools:
            assert SUBMIT_OUTPUT_TOOL not in [t.name for t in params.tools]


class TestReservedToolName:
    async def test_user_tool_named_submit_output_rejected(self):
        """When schema is set, user tools can't shadow the reserved name."""

        class CollidingTool:
            @property
            def definition(self):
                return ToolDefinition(
                    name=SUBMIT_OUTPUT_TOOL,
                    description="colliding",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, args):
                return ""

        llm = FakeLLM([_submit_response({"strategy_types": []})])
        with pytest.raises(ValueError, match="reserved"):
            await run_agent(
                _config(llm, output_schema=StrategyOutput, tools=[CollidingTool()]),
                "go",
            )

    async def test_user_tool_named_submit_output_allowed_without_schema(self):
        """Without a schema the name isn't reserved — user can own it."""

        class UserSubmit:
            @property
            def definition(self):
                return ToolDefinition(
                    name=SUBMIT_OUTPUT_TOOL,
                    description="user-owned",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, args):
                return "ok"

        llm = FakeLLM([
            LLMResponse(
                content="done",
                tool_calls=None,
                usage=Usage(5, 5),
                latency_ms=1,
                model="fake",
            ),
        ])
        # Should complete without raising
        result = await run_agent(_config(llm, tools=[UserSubmit()]), "hi")
        assert result.output == "done"


class TestFailClosedOnPlainText:
    async def test_plain_text_exhaustion_fails_closed(self):
        """Schema set + model never calls submit_output → deterministic failure message,
        not the last free-form content (which would look successful).
        """
        llm = FakeLLM([
            LLMResponse(content="free form 1", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
            LLMResponse(content="free form 2", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
            LLMResponse(content="free form 3", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
            LLMResponse(content="free form 4", tool_calls=None, usage=Usage(1, 1), latency_ms=1, model="fake"),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, max_parse_retries=1),
            "go",
        )

        from jig import AgentSchemaNotCalledError

        assert result.parsed is None
        # Structured termination reason — the API surface callers should
        # branch on rather than parsing the output string.
        assert isinstance(result.error, AgentSchemaNotCalledError)
        assert result.error.retries == 2
        # Marker is kept for backward-compat debugging; the model's last
        # free-form content must not leak through.
        assert "agent terminated" in result.output
        assert "free form" not in result.output


def _content_response(payload: dict[str, Any] | str) -> LLMResponse:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return LLMResponse(
        content=content, tool_calls=None, usage=Usage(10, 10), latency_ms=10, model="fake",
    )


class TestNativeModeHappyPath:
    async def test_terminal_content_parsed_no_submit_output_tool(self):
        """Native mode omits submit_output and parses schema-constrained
        assistant content directly as the terminal result."""
        llm = FakeNativeLLM([
            _content_response({"strategy_types": ["mean_reversion"], "best_sharpe": 1.2}),
        ])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, structured_output_mode="native"),
            "go",
        )

        assert isinstance(result.parsed, StrategyOutput)
        assert result.parsed.strategy_types == ["mean_reversion"]
        assert result.error is None

        params = llm.calls[0]
        assert params.tools is None
        assert params.response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "StrategyOutput",
                "schema": _normalize_strict_schema(StrategyOutput.model_json_schema()),
                "strict": True,
            },
        }

    async def test_ordinary_tools_remain_available(self):
        """Native mode still runs a normal working-tool turn before the
        schema-constrained terminal content."""
        class Echo:
            @property
            def definition(self):
                return ToolDefinition(
                    name="echo",
                    description="echoes",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, args):
                return "echoed"

        llm = FakeNativeLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={})],
                usage=Usage(1, 1), latency_ms=1, model="fake",
            ),
            _content_response({"strategy_types": ["x"]}),
        ])
        result = await run_agent(
            _config(
                llm, output_schema=StrategyOutput, structured_output_mode="native",
                tools=[Echo()],
            ),
            "go",
        )

        assert result.parsed is not None
        assert result.parsed.strategy_types == ["x"]
        # First turn's tool list carries the user tool, but never submit_output.
        first_tools = [t.name for t in (llm.calls[0].tools or [])]
        assert first_tools == ["echo"]


class TestNativeModeTerminalErrors:
    """Native mode never retries a schema violation — it's decode-time
    enforcement, so a mismatch means a provider bug or schema drift, not a
    correctable model mistake."""

    async def test_invalid_json_terminates_without_retry(self):
        llm = FakeNativeLLM([_content_response("not valid json")])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, structured_output_mode="native"),
            "go",
        )

        assert result.parsed is None
        assert isinstance(result.error, AgentNativeOutputError)
        assert len(llm.calls) == 1  # no retry burned a second round-trip

    async def test_schema_violation_terminates_without_retry(self):
        # best_sharpe must be float | None — a string violates the schema.
        llm = FakeNativeLLM([_content_response({"strategy_types": ["x"], "best_sharpe": "nope"})])
        result = await run_agent(
            _config(llm, output_schema=StrategyOutput, structured_output_mode="native"),
            "go",
        )

        assert result.parsed is None
        assert isinstance(result.error, AgentNativeOutputError)
        assert len(llm.calls) == 1


class TestNativeModeCapabilityPreflight:
    async def test_incapable_client_raises_before_first_completion_call(self):
        """FakeLLM (base class) doesn't declare response_format support —
        selecting native mode on it must fail before complete() runs at all."""
        llm = FakeLLM([_content_response({"strategy_types": ["x"]})])
        config = _config(llm, output_schema=StrategyOutput, structured_output_mode="native")

        with pytest.raises(UnsupportedResponseFormatError):
            await run_agent(config, "go")
        assert llm.calls == []  # complete() was never invoked


class TestStructuredOutputModeValidation:
    async def test_native_requires_output_schema(self):
        with pytest.raises(ValueError, match="requires output_schema"):
            _config(FakeNativeLLM([]), structured_output_mode="native")

    async def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="structured_output_mode"):
            _config(
                FakeNativeLLM([]), output_schema=StrategyOutput,
                structured_output_mode="auto",
            )

    async def test_legacy_is_the_default(self):
        """Backward compatibility: omitting structured_output_mode keeps the
        long-standing submit_output behavior."""
        llm = FakeLLM([_submit_response({"strategy_types": ["x"]})])
        result = await run_agent(_config(llm, output_schema=StrategyOutput), "go")

        assert result.parsed is not None
        assert SUBMIT_OUTPUT_TOOL in [t.name for t in llm.calls[0].tools]


def _canon(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class TestFlushedSQLiteExtractionProof:
    """Hermetic proof for the brief's 'next_action': is the durable,
    flushed ``submit_output`` TOOL_CALL span byte-identical evidence for
    the validated result, or does runner.py need to persist a separate
    complete value on the AGENT_RUN root?

    Runs against a real :class:`SQLiteTracer`, with the store closed and
    reopened before assertions, so the answer reflects durable persisted
    evidence rather than in-memory spans. Exercises retry (invalid then
    valid submission) and a valid payload with unicode, nested objects,
    ordered arrays, and a JSON encoding over 200 characters.

    Answer: no. Pydantic fills in the default for a field the model never
    supplied (``note``), so the raw ``submit_output`` arguments diverge
    from ``parsed.model_dump(mode="json")``. This is why runner.py
    persists the canonical complete value on the AGENT_RUN root (see
    ``ROOT_OUTPUT_COMPLETE_KEY`` in ``jig.core.runner``) instead of
    trusting the accepted tool-call span as the extraction source.
    """

    async def test_flushed_accepted_tool_call_diverges_from_validated_result(
        self, tmp_path: Any,
    ) -> None:
        class NestedTag(BaseModel):
            label: str
            priority: int

        class ProofOutput(BaseModel):
            title: str
            tags: list[NestedTag]
            order: list[str]
            note: str = "unset"  # model omits this — Pydantic fills the default

        long_unicode_title = "Katamari café ☃ report — " + ("türk " * 60)
        assert len(long_unicode_title) > 200

        valid_args = {
            "title": long_unicode_title,
            "tags": [
                {"label": "north", "priority": 2},
                {"label": "south", "priority": 1},
            ],
            "order": ["first", "second", "third"],
        }
        invalid_args = {"title": 123, "tags": "not-a-list", "order": []}

        llm = FakeLLM([
            _submit_response(invalid_args, call_id="attempt-1"),
            _submit_response(valid_args, call_id="attempt-2"),
        ])

        db_path = str(tmp_path / "proof.db")
        tracer = SQLiteTracer(db_path=db_path)
        config = _config(
            llm, output_schema=ProofOutput, max_parse_retries=2, tracer=tracer,
        )

        result = await run_agent(config, "go")
        assert result.parsed is not None
        assert result.parsed.title == long_unicode_title
        assert result.parsed.note == "unset"

        # Durable, flushed read: close the tracer's connection and reopen a
        # fresh SQLiteTracer against the same file.
        await tracer.close()
        reopened = SQLiteTracer(db_path=db_path)
        spans = await reopened.get_trace(result.trace_id)

        submit_spans = [s for s in spans if s.name == SUBMIT_OUTPUT_TOOL]
        assert len(submit_spans) == 2  # one invalid attempt, one accepted

        failed = [s for s in submit_spans if s.error is not None]
        accepted = [s for s in submit_spans if s.error is None]
        assert len(failed) == 1
        assert len(accepted) == 1
        # The first, invalid submission must not be selected as final output.
        assert failed[0].input == invalid_args
        assert accepted[0].output == valid_args

        validated_value = result.parsed.model_dump(mode="json")

        # The proof: the accepted tool-call's raw arguments are NOT
        # reliable complete evidence on their own — Pydantic added
        # `note`'s default, which the model never sent.
        assert "note" not in accepted[0].output
        assert validated_value["note"] == "unset"
        assert _canon(accepted[0].output) != _canon(validated_value)

        # The runner's root-persisted complete value IS what diff.py
        # actually extracts, and it is byte-identical to the validated
        # result (unicode preserved, array order preserved, nested
        # objects intact).
        root = next(
            s for s in spans
            if s.kind == SpanKind.AGENT_RUN and s.parent_id is None
        )
        assert root.output["output_kind"] == "structured"
        assert root.output["output_complete"] == validated_value
        assert _canon(root.output["output_complete"]) == _canon(validated_value)
        expected_hash = hashlib.sha256(_canon(validated_value)).hexdigest()
        assert root.output["output_sha256"] == expected_hash
        assert root.output["output_byte_length"] == len(_canon(validated_value))


class TestCanonicalOutputHash:
    """Direct unit coverage of ``jig.core.runner._canonical_output_hash``,
    the definition of complete-output equality that
    ``jig.replay.diff.trace_diff`` relies on for ``identical``."""

    def test_key_order_does_not_affect_hash(self):
        from jig.core.runner import _canonical_output_hash

        a = {"b": 1, "a": {"y": 2, "x": 1}}
        b = {"a": {"x": 1, "y": 2}, "b": 1}
        assert _canonical_output_hash(a) == _canonical_output_hash(b)

    def test_array_order_does_affect_hash(self):
        from jig.core.runner import _canonical_output_hash

        assert _canonical_output_hash({"items": [1, 2, 3]}) != _canonical_output_hash(
            {"items": [3, 2, 1]}
        )

    def test_rejects_nan(self):
        from jig.core.runner import _canonical_output_hash

        with pytest.raises(ValueError, match="non-finite"):
            _canonical_output_hash({"score": float("nan")})

    def test_rejects_infinity(self):
        from jig.core.runner import _canonical_output_hash

        with pytest.raises(ValueError, match="non-finite"):
            _canonical_output_hash({"score": float("inf")})

    def test_rejects_nested_non_finite(self):
        from jig.core.runner import _canonical_output_hash

        with pytest.raises(ValueError, match="non-finite"):
            _canonical_output_hash({"outer": [{"inner": float("-inf")}]})

    def test_rejects_unsupported_value(self):
        from jig.core.runner import _canonical_output_hash

        with pytest.raises(ValueError, match="unsupported"):
            _canonical_output_hash({"when": object()})

    def test_rejects_non_string_object_key(self):
        """dict keys are always strings after JSON round-tripping in
        practice, but a caller that hands a non-string-keyed dict must get
        a clear rejection rather than a silently wrong hash."""
        from jig.core.runner import _canonical_output_hash

        with pytest.raises(ValueError, match="non-string object key"):
            _canonical_output_hash({1: "x"})


class TestSubmitOutputStrictness:
    async def test_submit_output_definition_opts_into_strict(self):
        """The synthetic submit_output tool must carry the same decode-time
        strictness response_format gets: strict=True on the definition and
        a closed (additionalProperties: false) parameter schema."""
        llm = FakeLLM([_submit_response({"strategy_types": ["x"]})])
        await run_agent(_config(llm, output_schema=StrategyOutput), "go")

        submit_def = next(
            t for t in llm.calls[0].tools if t.name == SUBMIT_OUTPUT_TOOL
        )
        assert submit_def.strict is True
        assert submit_def.parameters.get("additionalProperties") is False


class _Leg(BaseModel):
    pair: str
    size: float = 1.0


class _NestedOutput(BaseModel):
    legs: list[_Leg]
    label: str = ""


class _OpenOutput(BaseModel):
    model_config = {"extra": "allow"}

    value: int


class TestNormalizeStrictSchema:
    """Strict decoders reject open or partially-required object nodes;
    _normalize_strict_schema must close every level of the schema, not
    just the root (PR 82 review)."""

    def test_nested_defs_are_closed_and_fully_required(self):
        schema = _normalize_strict_schema(_NestedOutput.model_json_schema())

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["legs", "label"]
        leg = schema["$defs"]["_Leg"]
        assert leg["additionalProperties"] is False
        assert leg["required"] == ["pair", "size"]

    def test_explicit_additional_properties_true_is_overridden(self):
        raw = _OpenOutput.model_json_schema()
        assert raw.get("additionalProperties") is True

        schema = _normalize_strict_schema(raw)
        assert schema["additionalProperties"] is False

    def test_input_schema_is_not_mutated(self):
        raw = _NestedOutput.model_json_schema()
        _normalize_strict_schema(raw)
        assert raw == _NestedOutput.model_json_schema()

    async def test_submit_output_tool_uses_normalized_schema(self):
        llm = FakeLLM([_submit_response({"legs": [{"pair": "BTC"}]})])
        await run_agent(_config(llm, output_schema=_NestedOutput), "go")

        submit_def = next(
            t for t in llm.calls[0].tools if t.name == SUBMIT_OUTPUT_TOOL
        )
        leg = submit_def.parameters["$defs"]["_Leg"]
        assert leg["additionalProperties"] is False
        assert leg["required"] == ["pair", "size"]

    async def test_response_format_uses_normalized_schema(self):
        llm = FakeNativeLLM([
            _content_response({"legs": [{"pair": "BTC", "size": 2.0}]}),
        ])
        await run_agent(
            _config(llm, output_schema=_NestedOutput, structured_output_mode="native"),
            "go",
        )

        sent = llm.calls[0].response_format["json_schema"]["schema"]
        assert sent["required"] == ["legs", "label"]
        assert sent["$defs"]["_Leg"]["additionalProperties"] is False


class TestTwoPhaseMode:
    """native_two_phase: working turns run unconstrained with tools; the
    first no-tool-call turn triggers one schema-constrained, tool-free
    finalize call whose content is the terminal result."""

    _ENVELOPE = {
        "type": "json_schema",
        "json_schema": {
            "name": "StrategyOutput",
            "schema": _normalize_strict_schema(StrategyOutput.model_json_schema()),
            "strict": True,
        },
    }

    async def test_first_no_tool_call_turn_triggers_finalize_call(self):
        llm = FakeNativeLLM([
            _content_response("work is done"),
            _content_response(
                {"strategy_types": ["mean_reversion"], "best_sharpe": 1.2}
            ),
        ])
        result = await run_agent(
            _config(
                llm,
                output_schema=StrategyOutput,
                structured_output_mode="native_two_phase",
            ),
            "go",
        )

        assert isinstance(result.parsed, StrategyOutput)
        assert result.parsed.strategy_types == ["mean_reversion"]
        assert result.error is None
        # Working turn: unconstrained.
        assert llm.calls[0].response_format is None
        # Finalize turn: tool-free and schema-constrained.
        assert llm.calls[1].tools is None
        assert llm.calls[1].response_format == self._ENVELOPE

    async def test_tools_offered_on_working_turns_never_on_finalize(self):
        class Echo:
            @property
            def definition(self):
                return ToolDefinition(
                    name="echo",
                    description="echoes",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, args):
                return "echoed"

        llm = FakeNativeLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={})],
                usage=Usage(1, 1), latency_ms=1, model="fake",
            ),
            _content_response("finished the work"),
            _content_response({"strategy_types": ["x"]}),
        ])
        result = await run_agent(
            _config(
                llm, output_schema=StrategyOutput,
                structured_output_mode="native_two_phase", tools=[Echo()],
            ),
            "go",
        )

        assert result.parsed is not None
        # Working turns: tools offered (never submit_output), unconstrained.
        assert [t.name for t in llm.calls[0].tools] == ["echo"]
        assert llm.calls[0].response_format is None
        assert llm.calls[1].response_format is None
        # Finalize call: no tools, constrained.
        assert llm.calls[2].tools is None
        assert llm.calls[2].response_format == self._ENVELOPE

    async def test_finalize_validation_failure_fails_closed(self):
        llm = FakeNativeLLM([
            _content_response("done"),
            _content_response("not even json"),
        ])
        result = await run_agent(
            _config(
                llm, output_schema=StrategyOutput,
                structured_output_mode="native_two_phase",
            ),
            "go",
        )

        assert result.parsed is None
        assert isinstance(result.error, AgentNativeOutputError)
        assert "agent terminated" in result.output

    async def test_max_tool_calls_exhaustion_routes_through_finalize(self):
        class Echo:
            @property
            def definition(self):
                return ToolDefinition(
                    name="echo",
                    description="echoes",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, args):
                return "echoed"

        def _tool_turn(call_id):
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=call_id, name="echo", arguments={})],
                usage=Usage(1, 1), latency_ms=1, model="fake",
            )

        llm = FakeNativeLLM([
            _tool_turn("t1"),
            _tool_turn("t2"),           # over budget → cap notice injected
            _content_response("done"),  # model wraps up
            _content_response({"strategy_types": ["x"]}),
        ])
        config = _config(
            llm, output_schema=StrategyOutput,
            structured_output_mode="native_two_phase", tools=[Echo()],
        ).with_(max_tool_calls=1)
        result = await run_agent(config, "go")

        assert result.parsed is not None
        assert result.error is None
        # The run still ends through the constrained finalize call.
        assert llm.calls[3].tools is None
        assert llm.calls[3].response_format == self._ENVELOPE

    async def test_tool_call_on_finalize_turn_nudged_then_fails_closed(self):
        from jig import AgentAmbiguousTurnError

        def _anomalous_turn(call_id):
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=call_id, name="echo", arguments={})],
                usage=Usage(1, 1), latency_ms=1, model="fake",
            )

        llm = FakeNativeLLM([
            _content_response("done"),  # triggers the finalize request
            _anomalous_turn("a1"),      # tool call on a tools=None turn
            _anomalous_turn("a2"),      # second anomaly exhausts the budget
        ])
        result = await run_agent(
            _config(
                llm, output_schema=StrategyOutput,
                structured_output_mode="native_two_phase", max_parse_retries=1,
            ),
            "go",
        )

        assert result.parsed is None
        assert isinstance(result.error, AgentAmbiguousTurnError)

    async def test_requires_response_format_support(self):
        llm = FakeLLM([_content_response("x")])
        with pytest.raises(UnsupportedResponseFormatError):
            await run_agent(
                _config(
                    llm, output_schema=StrategyOutput,
                    structured_output_mode="native_two_phase",
                ),
                "go",
            )

    async def test_requires_output_schema(self):
        llm = FakeNativeLLM([])
        with pytest.raises(ValueError, match="requires output_schema"):
            _config(llm, structured_output_mode="native_two_phase")
