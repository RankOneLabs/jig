"""Tests for typed agent outputs (AgentConfig.output_schema, submit_output tool)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel, Field

from jig import (
    AgentConfig,
    CompletionParams,
    LLMResponse,
    MemoryEntry,
    Message,
    Role,
    Score,
    ScoreSource,
    ToolCall,
    ToolDefinition,
    Usage,
    run_agent,
)
from jig.core.runner import SUBMIT_OUTPUT_TOOL
from jig.core.types import (
    AgentMemory,
    FeedbackLoop,
    Grader,
    LLMClient,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


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


class FakeMemory(AgentMemory):
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        return "mem-1"

    async def query(self, query: str, limit: int = 5, filter=None, session_id=None) -> list[MemoryEntry]:
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

    async def score(self, result_id: str, scores: list[Score]) -> None:
        self.scored.append((result_id, scores))

    async def get_signals(self, query, limit=3, min_score=None, source=None):
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

    def start_span(self, parent_id, kind, name, input=None):
        s = Span(id=f"span-{len(self.spans)}", trace_id="t-0", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, input=input)
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
) -> AgentConfig:
    return AgentConfig(
        name="test",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        memory=FakeMemory(),
        feedback=FakeFeedback(),
        tracer=FakeTracer(),
        tools=ToolRegistry(tools or []),
        output_schema=output_schema,
        max_parse_retries=max_parse_retries,
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
