"""Tests for AgentConfig variant derivation, immutability, and validation."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from jig import AgentConfig, LLMResponse, Message, Score, ScoreSource
from jig.core.types import (
    AgentMemory,
    CompletionParams,
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryEntry,
    ScoredResult,
    ScoreSource as _ScoreSource,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


class _FakeLLM(LLMClient):
    async def complete(self, params: CompletionParams) -> LLMResponse:
        raise NotImplementedError


class _FakeMemory(AgentMemory):
    async def add(self, content, metadata=None): return "m"
    async def query(self, query, limit=5, filter=None, session_id=None): return []
    async def get_session(self, session_id): return []
    async def add_to_session(self, session_id, message): pass
    async def clear(self, session_id=None, before=None): pass


class _FakeFeedback(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None): return "r"
    async def score(self, result_id, scores): pass
    async def get_signals(self, query, limit=3, min_score=None, source=None): return []
    async def query(self, q): return []
    async def export_eval_set(self, since=None, min_score=None, max_score=None, limit=None): return []


class _FakeTracer(TracingLogger):
    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):
        return Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now())
    def start_span(self, parent_id, kind, name, input=None):
        return Span(id="s", trace_id="t", kind=kind, name=name, started_at=datetime.now())
    def end_span(self, span_id, output=None, error=None, usage=None): pass
    async def get_trace(self, trace_id): return []
    async def list_traces(self, since=None, limit=50, name=None): return []
    async def flush(self): pass


def _base(**overrides: Any) -> AgentConfig:
    defaults: dict[str, Any] = dict(
        name="base",
        description="base agent",
        system_prompt="be helpful",
        llm=_FakeLLM(),
        memory=_FakeMemory(),
        feedback=_FakeFeedback(),
        tracer=_FakeTracer(),
        tools=ToolRegistry(),
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


class TestWith:
    def test_derives_new_instance(self):
        base = _base()
        variant = base.with_(max_tool_calls=25)
        assert variant is not base
        assert variant.max_tool_calls == 25
        assert base.max_tool_calls == 10  # unchanged

    def test_preserves_unspecified_fields(self):
        base = _base(system_prompt="original prompt", max_tool_calls=15)
        variant = base.with_(max_tool_calls=30)
        assert variant.system_prompt == "original prompt"
        assert variant.llm is base.llm
        assert variant.memory is base.memory

    def test_overrides_multiple_fields(self):
        base = _base()
        variant = base.with_(
            name="variant",
            system_prompt="new prompt",
            max_tool_calls=5,
            session_id="sess-42",
        )
        assert variant.name == "variant"
        assert variant.system_prompt == "new prompt"
        assert variant.max_tool_calls == 5
        assert variant.session_id == "sess-42"

    def test_unknown_field_raises(self):
        base = _base()
        with pytest.raises(TypeError):
            base.with_(not_a_field=True)

    def test_validation_runs_on_derived(self):
        """.with_() re-runs __post_init__ so invalid overrides are caught."""
        base = _base()
        with pytest.raises(ValueError, match="max_tool_calls"):
            base.with_(max_tool_calls=0)

    def test_preserves_generic_parameter(self):
        """AgentConfig[T].with_() returns AgentConfig[T]."""
        class MyOutput(BaseModel):
            value: int

        base: AgentConfig[MyOutput] = _base(output_schema=MyOutput)  # type: ignore[assignment]
        variant = base.with_(max_tool_calls=7)
        # Runtime: parsed would be MyOutput when the agent runs
        assert variant.output_schema is MyOutput


class TestFrozen:
    def test_mutation_raises(self):
        config = _base()
        with pytest.raises(FrozenInstanceError):
            config.name = "mutated"  # type: ignore[misc]

    def test_grader_mutation_raises(self):
        config = _base()
        with pytest.raises(FrozenInstanceError):
            config.grader = None  # type: ignore[misc]


class TestValidation:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            _base(name="")

    def test_rejects_non_positive_max_tool_calls(self):
        with pytest.raises(ValueError, match="max_tool_calls"):
            _base(max_tool_calls=0)
        with pytest.raises(ValueError, match="max_tool_calls"):
            _base(max_tool_calls=-5)

    def test_rejects_non_positive_max_llm_retries(self):
        with pytest.raises(ValueError, match="max_llm_retries"):
            _base(max_llm_retries=0)

    def test_rejects_non_positive_max_llm_calls(self):
        with pytest.raises(ValueError, match="max_llm_calls"):
            _base(max_llm_calls=0)
        with pytest.raises(ValueError, match="max_llm_calls"):
            _base(max_llm_calls=-1)

    def test_rejects_negative_max_parse_retries(self):
        """0 is valid (no retry budget); negative isn't."""
        _base(max_parse_retries=0)  # doesn't raise
        with pytest.raises(ValueError, match="max_parse_retries"):
            _base(max_parse_retries=-1)


class TestKeywordOnly:
    def test_positional_args_rejected(self):
        """kw_only=True means every field must be passed by name."""
        with pytest.raises(TypeError):
            AgentConfig(
                "name",  # type: ignore[misc]
                "desc",
                "prompt",
                _FakeLLM(),
                _FakeMemory(),
                _FakeFeedback(),
                _FakeTracer(),
                ToolRegistry(),
            )


class TestTypedGraderVariant:
    """Real-world pattern: one base config, N model/grader variants for sweeps."""

    async def test_typed_grader_survives_variant(self):
        class Out(BaseModel):
            kind: str

        seen: list[Any] = []

        class MyGrader(Grader[Out]):
            async def grade(self, input, output, context=None):
                seen.append(output)
                return [Score(dimension="q", value=1.0, source=_ScoreSource.HEURISTIC)]

        base = _base(output_schema=Out, grader=MyGrader())
        variant = base.with_(max_tool_calls=5)
        assert isinstance(variant.grader, MyGrader)
        assert variant.output_schema is Out
