# Phase 9 plan — trace propagation across dispatch

**Goal.** A jig agent that calls `jig.dispatch.run("pkg.mod:fn", args)` or uses
a `Tool(dispatch=True)` produces a trace whose worker-side spans reparent under
the caller's span. Same for `DispatchClient.complete()` — the LLM call on a
smithers worker appears as a child of the caller's agent span.

Federated read: each worker machine writes to its own local `jig_traces.db`;
a rollup service on willie queries across machines at read time. No cross-
machine SQLite writes.

## Scope

**In:**

- Smithers accepts `trace_context` on `JobSubmission`, carries through `Job`
  → `TaskRequest` to the worker, parses on the worker side, uses as span
  parent.
- Worker-local `jig_traces.db` populated during task execution. Schema matches
  jig's span table so the rollup can union results.
- Worker emits at minimum one root task span plus one child span per LLM
  inference / function execution. The goal is that after rollup,
  `SQLiteTracer.get_trace(trace_id)` returns caller spans plus worker spans
  with parent links intact.
- Rollup service on willie — new FastAPI container that exposes
  `GET /traces/{trace_id}` and fans out to each worker.
- Jig-side `FederatedTracer` wrapping `SQLiteTracer` + `RollupClient` so
  `trace_diff` / `replay` / general trace reads span worker boundaries.

**Out (deferred):**

- OTLP / OpenTelemetry interop.
- Full-detail span capture inside executors (task-level + one child per
  inference/function is enough for phase 9).
- Real-time streaming of worker spans (rollup is eventually consistent; a
  few seconds of lag is fine for the debugging use case).
- Authentication between rollup and worker DBs (single Tailscale network).

## Prerequisite audit

Verify these before writing code:

- `smithers/shared/models.py:227-238` — `JobSubmission`: no `model_config`,
  no `trace_context`. Pydantic default `extra="ignore"` silently drops the
  field jig sends.
- `smithers/shared/models.py:240-265` — `Job`: same absence.
- `smithers/shared/models.py:299-305` — `TaskRequest`: same absence.
- `smithers/dispatch/dispatcher.py:92-104` — where `Job` is constructed from
  `JobSubmission`. Must copy `trace_context`.
- `smithers/dispatch/dispatcher.py:181-187` — where `TaskRequest` is
  constructed from `Job`. Must copy `trace_context`.
- `smithers/worker/task_manager.py:136-180` — `_run_task`. Span start at
  ~line 150 after PENDING→RUNNING; span end at ~167/~180 (success / error /
  finally). One and only root-task-span site.
- `smithers/worker/executors/ollama.py:30` and `vllm.py:42` — where a child
  span wraps `_make_inference_request`.
- `smithers/worker/executors/function.py:157-163` — sync/async branch; wrap
  with a `TOOL_CALL` span.
- `jig/src/jig/tracing/sqlite.py:32-53` — `_SCHEMA`. Worker's DB must match.
- `jig/src/jig/core/types.py:149-204` — `SpanKind`, `Span`, `TraceContext`.
- `springfield/machines/willie/docker-compose.yml` — where the
  `trace-rollup` service is added. Image build context points at the
  smithers repo.
- `smithers/dispatch/job_store.py:1-80` — reference pattern for the
  aiosqlite init / schema module.

## Key design calls

### Schema: duplicate in smithers (don't import jig)

A small new `smithers/shared/trace_schema.py` exports the `spans` schema
string verbatim from jig, plus one trailing column `source_machine TEXT`
(NULL in jig's DB, always populated worker-side; useful for rollup
disambiguation).

Why duplicate instead of importing: jig is the *consumer* of worker DBs via
the rollup. Smithers importing jig would reverse the dep exactly wrong;
jig importing smithers at read time would drag smithers deps into every
replay call.

Both sides export `JIG_TRACE_SCHEMA_VERSION = 1`. Rollup warns on mismatch.

### Rollup: Tailscale-proxy query, not rsync-pull

Stateless FastAPI service on willie. `GET /traces/{trace_id}` fans out to
each worker's read-only `GET /traces/{trace_id}` endpoint with bounded
concurrency. Sleeping machines surface as `"status": "unreachable"` per
source in the response; jig's federated reader renders partial traces
with a banner.

Why not rsync-pull:

- Rollup always stale by the pull interval.
- Storage on willie sized for retention across N workers.
- Key management / auth story is non-trivial.
- Full-DB transfer cost even when nobody queries.

Why proxy-query wins here:

- Stateless. Restart-friendly.
- As fresh as the worker DB (which flushes per run).
- Sleeping workers are expected to be unreachable for the use case
  (you only inspect traces for recent runs where the worker was reachable).
- Tailscale ACL is already the auth boundary.

If historical queries for sleeping machines ever matter, a batch rsync
layer can be added *on top* without changing jig's interface. Retrofit in
the other direction is harder.

### Contextvar for span parent on workers

Instead of changing the `Executor` ABC signature (5 executors downstream),
use a module-level `ContextVar[SpanContext | None]` in
`smithers/worker/tracer.py`. `_run_task` sets it before calling
`executor.run(request)`; executors consult `WorkerTracer.current_parent()`.
Keeps the interface stable.

### WAL mode on worker SQLite

Workers write while the rollup reads. Without WAL, readers block writers.
Set `PRAGMA journal_mode=WAL` on first connect plus `busy_timeout=2000`.
Rollup reads are `query_only=ON`.

## File-by-file plan, in implementation order

### Step 1 — wire format accepts `trace_context`

Modify `smithers/shared/models.py`:

- Add `trace_context: dict[str, str] | None = None` to `JobSubmission`,
  `Job`, `TaskRequest`.
- Add explicit `model_config = ConfigDict(extra="ignore")` to `JobSubmission`
  to document the choice.

Modify `smithers/dispatch/dispatcher.py`:

- Line ~92: pass `trace_context=submission.trace_context` to `Job(...)`.
- Line ~181: pass `trace_context=job.trace_context` to `TaskRequest(...)`.

**Tests:** `JobSubmission.model_validate({..., "trace_context": {...}})`
round-trips; `submit_job` preserves the field onto the resulting `Job`;
`_dispatch_to_worker` posts a body containing `trace_context`; omitting
the field still works (backwards compat).

This step is purely additive and ships independently.

### Step 2 — duplicated schema module

New file `smithers/shared/trace_schema.py`:

- `JIG_TRACE_SCHEMA_VERSION = 1`.
- `SPANS_SCHEMA` string matching jig's, plus trailing `source_machine TEXT`.
- `SpanKind` string constants (duplicate, don't import).
- `SpanRow` TypedDict/dataclass so worker code has a typed surface without
  pulling pydantic.

**Tests:** snapshot test pinning the schema string; drift from jig's trips
the test immediately.

### Step 3 — worker-local tracer

New file `smithers/worker/tracer.py`:

- `WorkerTracer(db_path, source_machine)`.
- `start_span(parent_id, trace_id, kind, name, input=None, metadata=None) -> span_id`.
- `end_span(span_id, output=None, error=None, usage=None)`.
- `flush()`, `close()`.
- In-memory buffer (dict keyed by span_id) flushed via aiosqlite:
  one `INSERT OR REPLACE` per span, one commit per flush.
- First connect: `executescript(SPANS_SCHEMA)`, `PRAGMA journal_mode=WAL`,
  `PRAGMA busy_timeout=2000`.
- `source_machine` written into every row.
- `asyncio.Lock` guards the connection.

**Tests:** start → end → flush → raw aiosqlite read; assert all 16 fields
persisted. Parent `trace_id` set from an id the tracer never issued (came
from jig). Two concurrent spans don't corrupt. `source_machine` always
populated.

### Step 4 — parse `TraceContext` and wire tracer into `task_manager`

Modify `smithers/worker/task_manager.py`:

- Constructor gains `tracer: WorkerTracer | None = None`.
- `_run_task`:
  - After PENDING→RUNNING (line ~150): parse `request.trace_context` into
    `(trace_id, parent_span_id)` (inline helper).
  - If both tracer and trace_context present: `start_span(...)` with
    `kind="task_run"`, `name=f"task:{task_type}"`. Stash span_id.
  - Set contextvar before `executor.run(request)`.
  - Try/finally: `end_span(...)` with output / error / usage;
    `await tracer.flush()` **before** the terminal state transition.
- If trace_context absent or tracer None: run exactly as today. Phase 9
  must never change behavior for non-traced callers.

**Ordering invariant:** `executor.run` → `tracer.end_span` →
`tracer.flush` → acquire state lock → set terminal state. This guarantees
jig's poll loop sees `status=complete` only *after* spans are on disk
and the rollup can return them.

**Tests:** `FakeTracer` injected into `TaskManager`, task with
`trace_context` on the request produces `start_span` / `end_span` calls
with the expected parent linkage.

### Step 5 — child spans in inference + function executors

Modify `smithers/worker/executors/ollama.py`, `vllm.py`:

- Before `_make_inference_request`: if `WorkerTracer.current()` is set,
  `start_span(kind="llm_call", name=f"ollama:{task.model}", input={"messages_count": ..., "tools_count": ...})`.
- Try/finally end_span with `output={"content_len": ..., "tool_calls_count": ...}`,
  `usage=...`, `error=str(e) if raised`.
- Metadata only — no full message content. Jig's caller-side `LLM_CALL`
  span already has the content; we're just marking worker-side duration.

Modify `smithers/worker/executors/function.py`:

- Wrap the sync/async call with `start_span(kind="tool_call", name=f"fn:{typed.fn_ref}", input={"args_keys": sorted(typed.args)})`.
- `output={"value_type": type(value).__name__}` on success.

Modify `smithers/worker/server.py`:

- Construct `WorkerTracer` per worker, pass to `TaskManager(tracer=tracer)`.
- DB path: `os.environ.get("JIG_TRACE_DB_PATH", "jig_traces.db")`.
- `source_machine=machine_name`.
- Lifespan shutdown: `await tracer.close()`.

**Tests:** run an executor with tracer on the contextvar; assert one child
span produced with correct kind + parent linkage. No tracer → zero spans,
no errors. Callable raises → `end_span` called with error populated.

### Step 6 — worker `/traces/{trace_id}` endpoint

Modify `smithers/worker/server.py`:

- `@app.get("/traces/{trace_id}")` queries the tracer's DB via a new
  `get_trace_rows` method (returns `list[dict]`, not parsed `Span`).
- Response: `{"source_machine": "mcbain", "schema_version": 1, "spans": [{...}, ...]}`.
- Empty spans → return `[]`, not 404 (a given worker may have zero spans
  for a trace that exists elsewhere — expected in fan-out).

**Tests:** TestClient against `create_app()`, seed a span, GET
`/traces/{tid}`, assert JSON shape + schema version.

### Step 7 — rollup service

New package `smithers/trace_rollup/`:

- `server.py` — FastAPI app.
- `config.py` — reads worker URLs from `routing_config.yaml`'s `machines`
  section; env overrides.
- `client.py` — async httpx fan-out, bounded concurrency via
  `asyncio.Semaphore`, ~3s per-request timeout.
- `aggregator.py` — union spans across responses, dedup by span id (last
  write wins with a warning log), return combined list ordered by
  `started_at`.

Endpoints:

- `GET /traces/{trace_id}` → `{"trace_id": ..., "spans": [...], "sources": [{"machine": "mcbain", "status": "ok", "count": 3}, {"machine": "burns", "status": "unreachable", "count": 0}]}`.
- `GET /health` → `{"status": "ok", "workers_reachable": N}`.
- `GET /traces?since=...&limit=...&name=...` — deferred unless
  scope-budget allows; requires a `list_traces` endpoint on workers too.

Deploy: add `trace-rollup` service to willie's compose, port 8901 (adjacent
to dispatch at 8900), network `springfield`. Dockerfile runs
`uvicorn smithers.trace_rollup.server:app --port 8901`.

**Tests:** mock two worker endpoints (httpx respx). One returns spans, one
raises ConnectError. Assert aggregated response includes both sources
with correct statuses and the reachable spans. Schema version mismatch
logged.

### Step 8 — jig-side `FederatedTracer`

New file `jig/src/jig/tracing/federated.py`:

- `RollupClient` — thin httpx wrapper around
  `GET {rollup_url}/traces/{trace_id}`. Returns `list[Span]` parsed from
  the JSON. Factor the row-to-Span conversion into a shared helper.
- `FederatedTracer(TracingLogger)` wrapping a local `SQLiteTracer` +
  optional `RollupClient`.
  - Write path: 100% local. Worker spans never come through jig's writer.
  - Read path: `get_trace(trace_id) = local UNION rollup`, dedup by id,
    sort by `started_at`. Rollup unreachable → local-only + warning log.
  - `list_traces` delegates to local (root AGENT_RUN spans are always
    caller-side).

Default `rollup_url = "http://willie:8901"` matching the dispatch URL
pattern. Export from `jig.tracing.__init__`.

**Tests:** seed local with caller spans, stub RollupClient with worker
spans (same trace_id, parents pointing into local spans); `get_trace`
returns unified list with parent links resolving. Rollup unreachable →
local-only returned, warning logged. Duplicate id → one span + warning.

### Step 9 — end-to-end integration

`smithers/tests/test_trace_e2e.py`:

- In-process worker (existing `create_app` + TestClient pattern).
- Submit a function task with `trace_context={"trace_id": "t1", "parent_span_id": "p1"}`.
- Poll `/tasks/{id}` to completion.
- `GET /traces/t1` → at least 2 spans, task span's parent_id is "p1",
  fn span's parent_id is the task span.

`jig/tests/test_federated_integration.py`:

- respx-mocked rollup server returning canned worker spans.
- Dispatched `DispatchClient.complete` with a `TraceContext`.
- Local tracer captures caller span; `FederatedTracer.get_trace(trace_id)`
  returns both caller and worker spans with intact parenting.

## Risks and edge cases

- **Worker asleep at query time** — rollup marks source `"unreachable"`;
  jig renders partial trace with a banner. Acceptable.
- **Clock skew between machines** — spans ordered by `started_at`; NTP
  drift can swap adjacent siblings. Parent-child structure is always
  preserved via `parent_id`, never timestamps. Document wall-clock
  fidelity as `+/- NTP drift`.
- **Worker SQLite locks** — WAL + `busy_timeout=2000` handles concurrent
  reader/writer. Rollup reads are `query_only=ON`.
- **Task completes before rollup sees spans** — the ordering invariant
  (`flush` before terminal state) means by the time jig sees
  `status=complete`, spans are on disk.
- **Span id collision across machines** — uuid4 statistically impossible.
  `source_machine` column gives forensic disambiguation.
- **Schema drift between jig and smithers** — version constants + snapshot
  tests in both repos. Rollup logs warning and still serves (pass rows
  through).
- **Dispatcher doesn't emit spans** — correct; dispatcher is
  infrastructure, not an agent. `trace_context` passthrough is enough.
- **Retries** — rejected+retried jobs produce two task spans under the
  same parent. Useful signal; document it.
- **trace_context present but tracer=None** — log-and-continue; don't
  fail the task.
- **Large payloads in spans** — keep worker span input/output small
  (counts, types, model name). Storage on workers is precious; jig has
  full content on the caller side.
- **Rollup is a SPOF for debugging** — stateless, `FederatedTracer` falls
  back to local-only if rollup is down. Acceptable.

## Short implementation order

1. Wire format (models + dispatcher) + tests. Additive, ships alone.
2. `trace_schema.py` + `worker/tracer.py` + tests. No external surface
   change.
3. `task_manager.py` tracer integration + executor child spans + tests.
4. Worker `/traces/{trace_id}` endpoint + tests.
5. Rollup service + compose entry + tests.
6. Jig `FederatedTracer` + `RollupClient` + tests.
7. End-to-end integration tests in both repos.
