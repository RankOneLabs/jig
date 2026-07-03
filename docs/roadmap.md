# Jig roadmap

jig makes it easy to swap models, track what agents did, and handle
failures consistently — so probabilistic behavior is diffable,
comparable, and debuggable.

## What's shipped

**Swap models and compare runs**

- `jig.llm.from_model()` routes to any adapter — Anthropic, OpenAI,
  OpenRouter, Google, Ollama, or a remote dispatch fleet.
- `await jig.compare(input, configs)` runs one input against N configs
  (concurrently when `concurrency > 1`). `await jig.sweep(cases,
  configs)` covers the full case × config grid.
  `SweepResult.rollup()` aggregates per-config scores, latency, cost,
  and error breakdown.
- Per-call pricing is automatic across adapters; budgets can be
  enforced via `BudgetTracker` / `BudgetedLLMClient`.

**Typed agent outputs**

- `AgentConfig[T].output_schema` enforces a pydantic model on every
  run — provider-native structured output where available, JSON
  fallback otherwise.
- Graders read `result.parsed` fields directly; no regex extraction.
- A stable error taxonomy (`AgentError` hierarchy with `category`
  tags) makes error counts first-class sweep metrics.

**Feedback loop** *(SQLite path integration-tested end-to-end)*

- Quality scores from past runs feed back into the prompt
  automatically via `FeedbackLoop` (`SQLiteFeedbackLoop`).
- `await feedback.export_eval_set()` exports promptfoo-compatible test
  cases for batch evaluation.
- The full lifecycle — `run_agent` / `run_pipeline` → `store_result` →
  `score` → `query` / `get_signals` / `export_eval_set` — is proven by
  integration tests against a real SQLite database.
- Honcho and Zep adapters exist but are not integration-tested.

**Trace and replay agent runs**

- Every run produces a structured trace: LLM calls, tool executions,
  memory queries, grading steps — all as queryable spans.
- `jig.replay` reruns a recorded agent trace with a different config
  while holding the recorded tool outputs constant — requires a
  `tracer`, `llm`, and `feedback` instance. Isolates the model
  variable without re-running expensive tool calls.
- `jig.trace_diff` produces a structured diff between two recorded
  traces (requires a `tracer` instance): which tools were called
  differently, score deltas, cost deltas.

**Optional distributed dispatch**

- `jig.dispatch.run()` and `jig.llm.DispatchClient` route function
  calls and inference to remote GPU workers.
- Callback-based sweep fan-out: N parallel dispatched runs share one
  HTTP receiver — no polling coroutine per job.
- Trace parent links propagate across dispatch boundaries; worker
  spans appear as children in the caller's trace.

**Memory**

- `MemoryStore` (persistence) and `Retriever` (retrieval strategy) as
  separate, composable interfaces.
- Built-in: local SQLite + embeddings. Optional: Honcho, Zep (not
  integration-tested in this repo).

## What's next

**Richer eval tooling**

Version eval datasets, compare results across model iterations, and
tighten the promptfoo integration so export → run → diff is a single
command.

**Sweep auto-persistence**

Persist every sweep run to the feedback loop automatically so
longitudinal analysis across model versions requires no extra wiring.

**Regression baseline**

Capture a sweep baseline, diff future runs against it, and gate on
quality bounds — so a model upgrade that regresses a metric is
visible before it ships.

**Observability helpers**

Structured logging patterns, query helpers, and dashboard-ready
aggregations for agent runs in production.
