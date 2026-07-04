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
    def start_span(self, parent_id, kind, name, input=None, metadata=None):
        return Span(id="s", trace_id="t", kind=kind, name=name, started_at=datetime.now(), parent_id=parent_id, metadata=metadata)
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

    async def test_seeds_default_one_preserves_existing_behavior(self):
        """Without ``seeds=``, every run has seed_index 0."""
        result = await sweep(["c1", "c2"], [_config("a"), _config("b")])
        assert len(result.runs) == 4
        assert all(r.seed_index == 0 for r in result.runs)

    async def test_seeds_repeats_each_pair(self):
        """``seeds=N`` runs every (case, config) pair N times."""
        cases = ["c1", "c2"]
        configs = [_config("a"), _config("b")]
        result = await sweep(cases, configs, seeds=3)

        # 2 cases * 2 configs * 3 seeds = 12 runs
        assert len(result.runs) == 12
        # Each (case, config) pair should appear exactly 3 times with
        # distinct seed_index values 0..2
        from collections import defaultdict

        by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for r in result.runs:
            by_pair[(r.case_index, r.config_index)].append(r.seed_index)
        for pair, seeds in by_pair.items():
            assert sorted(seeds) == [0, 1, 2]

    async def test_seeds_must_be_positive(self):
        with pytest.raises(ValueError, match="seeds"):
            await sweep(["x"], [_config("a")], seeds=0)
        with pytest.raises(ValueError, match="seeds"):
            await sweep(["x"], [_config("a")], seeds=-1)

    async def test_worker_pool_preserves_full_grid_under_low_concurrency(self):
        """The bounded-queue worker pool must produce exactly one
        run per (case, config, seed) tuple even when concurrency is
        much smaller than the workload.
        """
        cases = [f"c{i}" for i in range(10)]
        configs = [_config("a"), _config("b")]
        result = await sweep(cases, configs, concurrency=2, seeds=3)
        # 10 cases * 2 configs * 3 seeds = 60 runs
        assert len(result.runs) == 60
        # Every (case, config) pair should appear exactly seeds=3 times
        from collections import Counter

        pairs = Counter((r.case_index, r.config_index) for r in result.runs)
        assert all(count == 3 for count in pairs.values())
        assert len(pairs) == 20  # 10 cases * 2 configs

    async def test_worker_pool_no_dropped_runs_when_total_below_concurrency(self):
        """Worker count is capped at the workload size — small sweeps
        don't spin up unused workers and still complete every run.
        """
        result = await sweep(["only-case"], [_config("only")], concurrency=8)
        assert len(result.runs) == 1
        assert result.runs[0].config_name == "only"

    async def test_empty_sweep_completes_cleanly(self):
        """No cases or no configs → empty result; no workers, no hang."""
        empty_a = await sweep([], [_config("a")], concurrency=4)
        assert empty_a.runs == []
        empty_b = await sweep(["x"], [], concurrency=4)
        assert empty_b.runs == []

    async def test_worker_pool_emits_debug_breadcrumbs(self, caplog):
        """The sweep worker pool must log DEBUG breadcrumbs at dispatch
        and completion. This is the regression guard for the silent-
        wedge scenario: when a worker hangs mid-run-agent, the operator
        needs to see *which* slot/config is in flight without attaching
        a debugger to the container.
        """
        import logging as _logging

        with caplog.at_level(_logging.DEBUG, logger="jig.sweep"):
            await sweep(["c0", "c1"], [_config("a")], concurrency=2)

        messages = [r.getMessage() for r in caplog.records if r.name == "jig.sweep"]
        assert any("sweep starting" in m for m in messages), messages
        assert any("dispatching" in m for m in messages), messages
        assert any("completed" in m for m in messages), messages


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


@pytest.mark.asyncio
class TestSweepWorkerErrorIsolation:
    """Budget and infrastructure failures must not kill workers or wedge queues."""

    async def test_budget_exhaustion_becomes_per_case_error(self):
        """JigBudgetError from run_agent → structured SweepRun error, not worker death."""
        from jig import BudgetedLLMClient, BudgetTracker

        # Budget so tight it's exhausted after the first call
        budget = BudgetTracker(limit_usd=0.001)
        llm = _FakeLLM("ok", cost=0.001)
        budgeted_llm = BudgetedLLMClient(inner=llm, budget=budget, estimate_cost_usd=0.001)

        cfg = AgentConfig(
            name="budget-cfg",
            description="budget-constrained",
            system_prompt="x",
            llm=budgeted_llm,
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        # 3 cases but budget only covers ~1 call; remaining should be budget_exhausted errors
        result = await sweep(["c1", "c2", "c3"], [cfg], concurrency=1)
        assert len(result.runs) == 3
        error_categories = [r.result.error.category for r in result.runs if r.result.error]
        assert "budget_exhausted" in error_categories
        for run in result.runs:
            if run.result.error is not None:
                assert run.result.usage == {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0,
                    "llm_calls": 0,
                    "tool_calls": 0,
                }

    async def test_infrastructure_error_becomes_per_case_error(self):
        """Unexpected Exception from run_agent → structured SweepRun error, worker continues."""

        call_count = 0

        class _FlakyLLM(LLMClient):
            async def complete(self, params):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("transient infra failure")
                return LLMResponse(
                    content="ok", tool_calls=None,
                    usage=Usage(10, 10, cost=0.001),
                    latency_ms=1.0, model="fake",
                )

        cfg = AgentConfig(
            name="flaky",
            description="flaky infra",
            system_prompt="x",
            llm=_FlakyLLM(),
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        result = await sweep(["c1", "c2", "c3"], [cfg], concurrency=1)
        # All 3 slots filled — worker did not die on case 2's error
        assert len(result.runs) == 3
        successes = [r for r in result.runs if r.result.error is None]
        errors = [r for r in result.runs if r.result.error is not None]
        assert len(successes) == 2
        assert len(errors) == 1
        assert str(errors[0].result.error) == "RuntimeError: transient infra failure"
        assert errors[0].result.usage == {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "tool_calls": 0,
        }

    async def test_infrastructure_error_with_empty_message_keeps_type(self):
        class _EmptyError(Exception):
            pass

        class _BrokenLLM(LLMClient):
            async def complete(self, params):
                raise _EmptyError()

        cfg = AgentConfig(
            name="empty-error",
            description="empty infra error",
            system_prompt="x",
            llm=_BrokenLLM(),
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        result = await sweep(["c1"], [cfg], concurrency=1)

        assert len(result.runs) == 1
        assert result.runs[0].result.error is not None
        assert str(result.runs[0].result.error) == "_EmptyError"

    async def test_successes_preserved_alongside_failures(self):
        """Successful results are not discarded when some cases fail."""
        call_count = 0

        class _PartialLLM(LLMClient):
            async def complete(self, params):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise RuntimeError("every other call fails")
                return LLMResponse(
                    content="ok", tool_calls=None,
                    usage=Usage(10, 10, cost=0.001),
                    latency_ms=1.0, model="fake",
                )

        cfg = AgentConfig(
            name="partial",
            description="partial failures",
            system_prompt="x",
            llm=_PartialLLM(),
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        result = await sweep(["c1", "c2", "c3", "c4"], [cfg], concurrency=2)
        assert len(result.runs) == 4
        successes = [r for r in result.runs if r.result.error is None]
        assert len(successes) == 2


@pytest.mark.asyncio
class TestCompareErrorIsolation:
    """Budget and infrastructure failures in compare() → structured CompareRun results."""

    async def test_budget_error_becomes_structured_result(self):
        """JigBudgetError in a config → CompareRun with error, not exception propagation."""
        from jig import BudgetedLLMClient, BudgetTracker

        budget = BudgetTracker(limit_usd=0.0001)
        llm = _FakeLLM("ok", cost=0.001)
        budgeted_llm = BudgetedLLMClient(inner=llm, budget=budget, estimate_cost_usd=0.001)

        good_cfg = _config("good")
        bad_cfg = AgentConfig(
            name="budget-limited",
            description="over budget",
            system_prompt="x",
            llm=budgeted_llm,
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )

        result = await compare("input", [good_cfg, bad_cfg])
        assert len(result.runs) == 2
        bad_run = next(r for r in result.runs if r.config_name == "budget-limited")
        assert bad_run.result.error is not None
        assert bad_run.result.error.category == "budget_exhausted"
        # Good config still succeeded
        good_run = next(r for r in result.runs if r.config_name == "good")
        assert good_run.result.error is None

    async def test_infrastructure_error_becomes_structured_result(self):
        """Exception in a config → CompareRun with error, not exception propagation."""

        class _CrashLLM(LLMClient):
            async def complete(self, params):
                raise RuntimeError("disk full")

        crash_cfg = AgentConfig(
            name="crash",
            description="crashes",
            system_prompt="x",
            llm=_CrashLLM(),
            store=_FakeMem(), retriever=None,
            feedback=_FakeFB(),
            tracer=_FakeTracer(),
            tools=ToolRegistry(),
        )
        good_cfg = _config("good")

        result = await compare("input", [good_cfg, crash_cfg])
        assert len(result.runs) == 2
        crash_run = next(r for r in result.runs if r.config_name == "crash")
        assert crash_run.result.error is not None
        good_run = next(r for r in result.runs if r.config_name == "good")
        assert good_run.result.error is None
