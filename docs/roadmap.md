# Jig roadmap — 14-phase redesign toward an experimentation framework

**Mission:** jig makes it easy to swap models, track what agents did, and
handle failures consistently — so probabilistic behavior is diffable,
comparable, and debuggable.

The plan is 14 phases (0–13). Phase 0 shipped the ergonomic LLM layer;
phases 1–3 tightened the core abstractions and error model; phases 4–6
build the experimentation primitives; phases 7–11 wire smithers and
replay; phases 12–13 migrate consumers.

## Status at a glance

| Phase | Name                                      | Status        | PR   | Breaking |
| ----- | ----------------------------------------- | ------------- | ---- | -------- |
| 0     | LLM layer ergonomics                      | **merged**    | #12        | no       |
| 1     | Typed agent outputs                       | **merged**    | #13        | yes      |
| 2     | `AgentConfig` variant derivation          | **merged**    | #14        | yes      |
| 3     | Error taxonomy                            | **merged**    | #15        | minor    |
| 4     | Feedback query API + `PastResults`        | **merged**    | #16        | minor    |
| 5     | `jig.compare` + `jig.sweep`               | **merged**    | #16        | no       |
| 6     | Memory split (Store / Retriever)          | **merged**    | #17        | yes      |
| 7     | Smithers dispatch — LLM                   | **merged**    | jig #19, smithers #18 | no |
| 8     | Smithers dispatch — tools + steps         | **merged**    | jig #19, smithers #18 | no |
| 9     | Trace propagation across dispatch         | **planned**   | see `phase-9-trace-propagation.md` | no |
| 10    | Callback-based sweep fan-out              | pending       | —          | no       |
| 11    | Agent replay from trace                   | **open**      | #20        | minor    |
| 12    | Ta capstone migration                     | pending       | —          | —        |
| 13    | Scout + algerknown lift                   | pending       | —          | —        |

Phases 4+5 are bundled in one PR because their surfaces pair naturally
(feedback query ↔ sweep auto-persist). Phase 6 is independent and runs
in a parallel agent session. Phases 7+8 were bundled in one pair of
PRs (jig #19 + smithers #18) because they tell the same dispatch
story.

**Next critical-path work: phase 9.** Detailed plan at
[`phase-9-trace-propagation.md`](phase-9-trace-propagation.md) on
branch `feat/phase-9-trace-propagation`. Phase 11 (PR #20) is parallel
and does not gate 9/10/12/13.

## Locked design decisions

These were decided early and inform every phase:

1. **Python 3.12 floor.** PEP 695 generics (`class AgentConfig[T]:`)
   are used everywhere; no `TypeVar` boilerplate.
2. **Typed outputs: provider-native + JSON fallback.** Adapters with
   native structured-output support (OpenAI `response_format`, Gemini
   `response_schema`, Anthropic forced tool) use it; others fall back
   to forcing a `submit_output` tool and pydantic-validating its args.
3. **Sweep results auto-persist to `FeedbackLoop`** as
   `ScoreSource.EVAL`. Every sweep run is an eval; tomorrow's
   `PastResults` lookup sees yesterday's sweep.
4. **Dispatch function registry via Python entry points.** Homelab
   trust boundary — the auto-discovery convenience beats the
   yaml-allowlist explicitness. Workers log every discovered fn on
   startup for visibility.
5. **Trace aggregation is federated.** Each machine writes to its
   local `jig_traces.db`; a rollup service on willie queries across
   machines at read time. No cross-machine SQLite writes.

## Phase details

### Phase 0 — LLM layer ergonomics *(merged #12)*

`jig.llm.from_model(name)`, `jig.complete(model, messages)`, per-model
pricing table with auto cost-stamping in all adapters,
`BudgetTracker`/`BudgetedLLMClient`, `JigBudgetError`.

Rationale: make `jig.llm` the path of least resistance so consumers
stop reaching for `anthropic`/`openai` SDKs directly.

### Phase 1 — Typed agent outputs *(merged #13)*

`AgentConfig[T].output_schema: type[T]` where `T: BaseModel`.
`AgentResult[T].parsed: T | None`. `Grader[T]` generic. Runner injects
a reserved `submit_output` tool whose args are pydantic-validated;
`max_parse_retries` budget for correction. Reserved tool name rejected
when user registers their own `submit_output`.

Unblocks: agent-to-agent handoff via field access instead of regex;
graders consume parsed instances.

### Phase 2 — `AgentConfig` variant derivation *(merged #14)*

Frozen, kw_only dataclass. `config.with_(model=..., max_tool_calls=15)`
returns a new instance with overrides applied; preserves the generic
`T`. `__post_init__` validates numeric bounds. New `max_llm_calls`
cap (default 50) prevents unbounded loops when model keeps emitting
tool calls past `max_tool_calls`.

Ambiguous-turn guard: if the model emits `submit_output` alongside
other tool calls, runner nudges to retry (would otherwise silently
drop other tool executions).

Unblocks: explorer/specialist/refiner in ta become one base + three
variants. Sweeps over config combinations are trivial list comprehensions.

### Phase 3 — Error taxonomy *(merged #15)*

Filled out empty error classes:
`JigMemoryError(source, operation, retryable)`,
`JigToolError(tool_name, phase, retryable, underlying)` where
`phase: Literal["schema", "execute", "serialize"]`.

New `AgentError` hierarchy for terminal conditions:
`AgentMaxLLMCallsError`, `AgentMaxLLMRetriesError`,
`AgentSchemaValidationError`, `AgentSchemaNotCalledError`,
`AgentAmbiguousTurnError`, `AgentLLMPermanentError`. Each has a
stable `category` tag used for span metadata + rollup queries.

`AgentResult.error: AgentError | None` replaces string-matching on
`output` markers. Root trace span stamped with `error_category`.

Runner hardened:
- `run_agent` body wrapped in `try/finally`; tracer flush always runs
- LLM call count increments at attempt time (not success), so cap
  applies to failures
- `TracingLogger.flush()` no-op default so `StdoutTracer` works
- Fast-fail on known-permanent HTTP status codes `{400, 401, 403,
  404, 422}` instead of naive `not retryable` (would have collapsed
  transient 5xx into immediate termination)

### Phase 4 — Feedback query API + `PastResults` *(open #16)*

`FeedbackQuery(similar_to, agent_name, model, tags, min_score,
max_age, limit)`. `FeedbackLoop.query(q) -> list[ScoredResult]` and
`FeedbackLoop.store_result(content, input, metadata) -> str` promoted
to the protocol. `SQLiteFeedbackLoop.query` combines embedding
similarity with metadata-JSON filters.

`jig.tools.PastResults(feedback, default_k=5, agent_name=None)`
wraps the query API as a tool. Agents call it with
`{hypothesis, min_score?, k?}`.

### Phase 5 — `jig.compare` + `jig.sweep` *(open #16)*

`compare(input, configs, concurrency=1) -> CompareResult` for
single-probe A/B/C on configs. `sweep(cases, configs,
concurrency=1, sweep_id=None) -> SweepResult` for the full grid.
`SweepResult.rollup()` aggregates per-config: n, avg scores per
dimension, avg cost/latency, success rate, `error_categories`
counter (via phase 3's taxonomy).

Deferred to follow-ups: auto-persist to `FeedbackLoop`, dispatch
backend, pandas integration.

### Phase 6 — Memory split *(in flight, parallel agent)*

**Parallel development.** Another agent is implementing this in its
own worktree. Coordination happens via rebase at merge time — shared
surfaces are `core/types.py` (separate protocol additions) and the
two `__init__.py` exports (adjacent lines).

Splits `AgentMemory` into `MemoryStore` (storage) + `Retriever`
(retrieval strategy). Reference retrievers: `DenseRetriever` at
minimum, `BM25Retriever`/`HybridRetriever`/`RerankingRetriever`
optionally. `LocalMemory` refactored as composition.

`AgentConfig.memory` replaced with `store` + `retriever` fields.
Every retrieval produces a `SpanKind.MEMORY_QUERY` span with
retrieved-id/score metadata for traceability.

Unblocks: retrieval strategy becomes a sweep axis; algerknown's
ChromaDB + vector pattern fits naturally.

### Phase 7 — Smithers dispatch, LLM track

Fix `DispatchClient` tool-use support (currently refused).
`from_model("dispatch/<model>")` routes via smithers. Smithers
worker executors (Ollama, vLLM) carry OpenAI-compatible tool-call
schema end-to-end.

### Phase 8 — Smithers dispatch, tools + steps

New `jig.dispatch.run(fn_ref, payload, trace_context)` entry point.
`Tool(dispatch=True)` routes tool execution through smithers. Worker
function registry via Python entry points (`jig.smithers_fn` group).
Data locality via NFS (willie exposes `/ta-data` read-only to workers).

Payoff: ta's specialist sweep drops from ~83h to ~8h (nested parallelism
across ~3 workers × distributed Optuna trials).

### Phase 9 — Trace propagation across dispatch

`TraceContext(trace_id, parent_span_id)` serialized into every dispatch
payload. Worker starts its spans with caller's `parent_span_id`.
Federated read: each machine's local traces DB + rollup service on
willie for cross-machine queries.

### Phase 10 — Callback-based sweep fan-out

`jig.dispatch.listen(port)` spins up an HTTP receiver that resolves
per-job futures on smithers callback. `jig.sweep(..., dispatch=smithers)`
uses it automatically — 500 sweep runs no longer need 500 polling
coroutines. Smithers callback delivery gets retry-with-backoff.

### Phase 11 — Agent replay from trace

`jig.replay(trace_id, config_override)` reads recorded tool-call
outputs as canned responses, reruns the agent loop with an overridden
config. `jig.trace_diff(trace_a, trace_b)` produces a structured diff
(which tools called differently, score deltas).

The probabilistic-debugging superpower: isolate the model variable by
holding tool results constant across two runs.

### Phase 12 — Ta capstone migration

Delete `CostTrackingLLM`, `create_llm()`, `_parse_strategy_types()`,
three near-duplicate config builders, the regex-based grader extraction.
Wire `RunBacktestTool(dispatch=True)`. Benchmark specialist sweep
before/after.

### Phase 13 — Scout + algerknown lift

Scout's `evaluate/generate/critique/revise` pipeline → one `run_agent`
call with typed `ReplyCandidate` output + revision tool. Algerknown's
proposer → typed-output agent; ChromaDB → `MemoryStore` +
`DenseRetriever`. Scout and algerknown stop calling the Anthropic SDK
directly, use `jig.complete()`.

## Parallel development policy

Phases 4+5 and 6 are semantically independent (feedback/sweep vs memory).
When both branches are open simultaneously, the shared surfaces are:

- `src/jig/core/types.py` — each adds distinct dataclasses/protocols;
  merge conflicts, if any, are mechanical insertions in different
  regions
- `src/jig/__init__.py` and `src/jig/core/__init__.py` — both add
  export lines; adjacent but separate

Merge order: whichever lands first defines the "new main"; the other
rebases. Expected conflict cost: one-line resolutions in export lists.

Phase 7–11 are gated on phase 6 (memory) because some of them
restructure pieces phase 6 refactors. Phase 12 is gated on 7+8+9.
Phase 13 is gated on most of the stack.
