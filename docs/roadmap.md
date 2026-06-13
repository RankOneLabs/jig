# Jig roadmap

jig makes it easy to swap models, track what agents did, and handle
failures consistently — so probabilistic behavior is diffable,
comparable, and debuggable.

## What's shipped

**Swap models and compare runs**

- `jig.llm.from_model()` routes to any adapter — Anthropic, OpenAI,
  OpenRouter, Google, Ollama, or a remote dispatch fleet.
- `jig.compare(input, configs)` runs one input against N configs in
  parallel. `jig.sweep(cases, configs)` covers the full case × config
  grid. `SweepResult.rollup()` aggregates per-config scores, latency,
  cost, and error breakdown.
- Per-call pricing and budget tracking are automatic across all
  adapters.

**Typed agent outputs**

- `AgentConfig[T].output_schema` enforces a pydantic model on every
  run — provider-native structured output where available, JSON
  fallback otherwise.
- Graders read `result.parsed` fields directly; no regex extraction.
- A stable error taxonomy (`AgentError` hierarchy with `category`
  tags) makes error counts first-class sweep metrics.

**Feedback loop**

- Quality scores from past runs feed back into the prompt
  automatically via `FeedbackLoop`.
- `feedback.export_eval_set()` exports promptfoo-compatible test cases
  for batch evaluation.

**Trace and replay agent runs**

- Every run produces a structured trace: LLM calls, tool executions,
  memory queries, grading steps — all as queryable spans.
- `jig.replay(trace_id, config_override)` reruns an agent with a
  different config while holding recorded tool outputs constant.
  Isolates the model variable without re-running expensive tool calls.
- `jig.trace_diff(trace_a, trace_b)` produces a structured diff:
  which tools were called differently, score deltas, cost deltas.

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
- Built-in: local SQLite + embeddings. Optional: Honcho, Zep.

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
