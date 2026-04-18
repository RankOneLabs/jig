"""Tests for jig.compare and jig.sweep experimentation primitives."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from jig import (
    AgentConfig,
    CompareResult,
    EvalCase,
    LLMResponse,
    Message,
    Role,
    Score,
    ScoreSource,
    SweepResult,
    Usage,
    compare,
    sweep,
)
from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryEntry,
    MemoryStore,
    Retriever,
    Span,
    SpanKind,
    TracingLogger,
)
from jig.tools import ToolRegistry


# --- Fakes (kept minimal, scripted responses per-instance) ---


class _FakeLLM(LLMClient):
    def __init__(self, content: str, cost: float = 0.001, usage_tokens: int = 10):
        self._content = content
        self._cost = cost
        self._tokens = usage_tokens

    async def complete(self, params):
        return LLMResponse(
            content=self._content,
            tool_calls=None,
            usage=Usage(self._tokens, self._tokens, cost=self._cost),
            latency_ms=1.0,
            model="fake",
        )


class _FakeMem(MemoryStore, Retriever):
    async def add(self, content, metadata=None): return "m"
    async def get(self, id): return None
    async def all(self): return []
    async def delete(self, id): pass
    async def retrieve(self, query, k=5, context=None): return []
    async def get_session(self, sid): return []
    async def add_to_session(self, sid, m): pass
    async def clear(self, session_id=None, before=None): pass


class _FakeFB(FeedbackLoop):
    async def store_result(self, content, input_text, metadata=None): return "r"
    async def score(self, rid, scores): pass
    async def get_signals(self, q, limit=3, min_score=None, source=None): return []
    async def query(self, q): return []
    async def export_eval_set(self, since=None, min_score=None, max_score=None, limit=None): return []


class _FakeTracer(TracingLogger):
    def start_trace(self, name, metadata=None, kind=SpanKind.AGENT_RUN):
        return Span(id="t", trace_id="t", kind=kind, name=name, started_at=datetime.now())
    def start_span(self, parent_id, kind, name, input=None):
        return Span(id="s", trace_id="t", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id)
    def end_span(self, span_id, output=None, error=None, usage=None): pass
    async def get_trace(self, tid): return []
    async def list_traces(self, since=None, limit=50, name=None): return []


class _ConstantGrader(Grader):
    def __init__(self, value: float):
        self._value = value

    async def grade(self, input, output, context=None):
        return [Score(dimension="q", value=self._value, source=ScoreSource.HEURISTIC)]


def _config(
    name: str,
    *,
    content: str = "ok",
    cost: float = 0.001,
    grader: Grader | None = None,
) -> AgentConfig:
    return AgentConfig(
        name=name,
        description=f"{name} agent",
        system_prompt="be brief",
        llm=_FakeLLM(content, cost=cost),
        store=_FakeMem(), retriever=None,
        feedback=_FakeFB(),
        tracer=_FakeTracer(),
        tools=ToolRegistry(),
        grader=grader,
    )


# --- compare ---


@pytest.mark.asyncio
class TestCompare:
    async def test_runs_all_configs_on_one_input(self):
        configs = [
            _config("sonnet", content="S-out"),
            _config("haiku", content="H-out"),
            _config("gemini", content="G-out"),
        ]
        result = await compare("go", configs)

        assert isinstance(result, CompareResult)
        assert result.input == "go"
        assert len(result.runs) == 3
        assert [r.config_name for r in result.runs] == ["sonnet", "haiku", "gemini"]
        assert [r.config_index for r in result.runs] == [0, 1, 2]

    async def test_rollup_per_config(self):
        configs = [
            _config("a", cost=0.01, grader=_ConstantGrader(0.8)),
            _config("b", cost=0.02, grader=_ConstantGrader(0.4)),
        ]
        result = await compare("input", configs)
        roll = result.rollup()

        assert set(roll) == {"a", "b"}
        assert roll["a"]["avg_scores"]["q"] == pytest.approx(0.8)
        assert roll["b"]["avg_scores"]["q"] == pytest.approx(0.4)
        assert roll["a"]["cost_usd"] == pytest.approx(0.01)
        assert roll["b"]["cost_usd"] == pytest.approx(0.02)
        assert roll["a"]["error_category"] is None
        assert roll["b"]["error_category"] is None

    async def test_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="concurrency"):
            await compare("x", [_config("a")], concurrency=0)

    async def test_duplicate_config_names_rejected(self):
        configs = [_config("same"), _config("same"), _config("unique")]
        with pytest.raises(ValueError, match="duplicates.*same"):
            await compare("x", configs)

    async def test_concurrency_parallel(self):
        """concurrency=N runs configs in parallel (sanity — no timeout here)."""
        configs = [_config(f"c{i}") for i in range(4)]
        # Just verify it doesn't error
        result = await compare("x", configs, concurrency=4)
        assert len(result.runs) == 4


# --- sweep ---


@pytest.mark.asyncio
class TestSweep:
    async def test_cases_x_configs(self):
        cases = ["case1", "case2", "case3"]
        configs = [_config("a"), _config("b")]

        result = await sweep(cases, configs)

        assert isinstance(result, SweepResult)
        assert result.sweep_id  # UUID generated
        assert len(result.runs) == 6  # 3 * 2
        # Every (case_index, config_index) pair present
        pairs = {(r.case_index, r.config_index) for r in result.runs}
        assert pairs == {(ci, gi) for ci in range(3) for gi in range(2)}

    async def test_accepts_evalcase(self):
        cases = [
            EvalCase(input="eval1", metadata={"tag": "a"}),
            EvalCase(input="eval2"),
        ]
        configs = [_config("only")]

        result = await sweep(cases, configs)
        inputs = sorted(r.input for r in result.runs)
        assert inputs == ["eval1", "eval2"]

    async def test_custom_sweep_id(self):
        result = await sweep(["x"], [_config("a")], sweep_id="my-sweep-42")
        assert result.sweep_id == "my-sweep-42"

    async def test_rollup_aggregates(self):
        """Same config run over multiple cases → averaged rollup."""
        configs = [
            _config("a", cost=0.01, grader=_ConstantGrader(0.5)),
            _config("b", cost=0.04, grader=_ConstantGrader(0.9)),
        ]
        result = await sweep(["c1", "c2", "c3"], configs)
        roll = result.rollup()

        assert set(roll) == {"a", "b"}
        assert roll["a"]["n"] == 3
        assert roll["b"]["n"] == 3
        assert roll["a"]["avg_scores"]["q"] == pytest.approx(0.5)
        assert roll["b"]["avg_scores"]["q"] == pytest.approx(0.9)
        assert roll["a"]["avg_cost_usd"] == pytest.approx(0.01)
        assert roll["b"]["avg_cost_usd"] == pytest.approx(0.04)
        assert roll["a"]["success_rate"] == 1.0
        assert roll["b"]["success_rate"] == 1.0
        assert roll["a"]["error_categories"] == {}

    async def test_empty_cases_empty_result(self):
        result = await sweep([], [_config("a")])
        assert result.runs == []
        assert result.rollup() == {}

    async def test_concurrency_validated(self):
        with pytest.raises(ValueError, match="concurrency"):
            await sweep(["x"], [_config("a")], concurrency=-1)

    async def test_duplicate_config_names_rejected(self):
        configs = [_config("a"), _config("b"), _config("a")]
        with pytest.raises(ValueError, match="duplicates"):
            await sweep(["case1"], configs)


@pytest.mark.asyncio
class TestSweepErrorCounting:
    """Verify rollup captures error_category counts from terminated runs."""

    async def test_rollup_counts_error_categories(self):
        # Construct a config whose agent will terminate with an error.
        # Use AgentLLMPermanentError via 401 status.
        from jig.core.errors import AgentLLMPermanentError
        from unittest.mock import patch

        from jig.core.errors import JigLLMError

        class _ErrLLM(LLMClient):
            async def complete(self, params):
                raise JigLLMError("auth fail", "fake", status_code=401, retryable=False)

        cfg = AgentConfig(
            name="bad",
            description="fails auth",
            system_prompt="x",
            llm=_ErrLLM(),
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        result = await sweep(["c1", "c2", "c3"], [cfg])
        roll = result.rollup()

        # All three runs terminated with llm_permanent_error
        assert roll["bad"]["success_rate"] == 0.0
        assert roll["bad"]["error_categories"] == {"llm_permanent_error": 3}
