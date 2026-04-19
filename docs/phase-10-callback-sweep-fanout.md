# Phase 10 plan — Callback-based sweep fan-out

**Goal.** Replace per-job polling with a single callback listener so
sweeps can dispatch hundreds of runs without hundreds of polling
coroutines. `jig.dispatch.listen(port)` starts an HTTP receiver that
resolves per-job futures on smithers callback; `jig.sweep(...,
dispatch="smithers")` uses it automatically. Polling stays as a
fallback for single-shot callers and for environments where the
listener can't bind a reachable port.

## Scope

**In:**

- New `jig.dispatch.listen(port=0, host="0.0.0.0", base_url=None)` —
  starts an `aiohttp.web` HTTP receiver that handles `POST
  /callbacks/{nonce}` and resolves a per-job `asyncio.Future` keyed by
  that nonce (registered before submit to sidestep the post-submit /
  pre-register race).
  `aiohttp` lands as an optional extra `jig[callback]`; the same
  `try:` / `ImportError → install jig[callback]` pattern jig already
  uses for `[anthropic]`, `[openai]`.
- `jig.dispatch.run(...)` gains a `callback_url` auto-populated when a
  listener is active. Result delivery switches from polling to await on
  the listener-resolved future. Polling remains the fallback when no
  listener is registered or the listener fails health check.
- `jig.sweep(...)` gains `dispatch: str | None = None`. When set to
  `"smithers"`, each run goes through `run_agent`'s dispatch path with
  the shared listener — N parallel runs share one HTTP receiver and
  one httpx client.
- Retry-with-backoff on the jig side covers listener bring-up and
  listener-health probes. Callback *delivery* retry is already handled
  by smithers' callback worker (10 attempts, 60s–3600s backoff with
  ±25% jitter); we rely on it, don't duplicate it.
- Tracing: each dispatched job emits a single `dispatch_job` span
  covering submit → future resolution. Poll ticks (fallback path) stay
  as child spans of that parent.
- `jig.dispatch.__init__.py` exports `listen`, `stop`, and the
  listener-lifecycle helpers alongside the existing `run`/`aclose`.

**Out (explicit non-goals):**

- Smithers-side changes. Callback delivery infrastructure already
  exists at `smithers/dispatch/callbacks.py:56–80` +
  `callback_worker.py` (at-least-once, durable, retry-with-backoff,
  idempotent). Jig only needs to populate `callback_url` in submissions.
- `DispatchClient` (LLM) API surface — no caller-visible change. The
  underlying submit/await path swaps from polling to callback when a
  listener is up.
- Consumer repo changes. `ta`, `algerknown`, `scout` do not call
  `sweep(dispatch=...)` today; no migration needed. This phase
  establishes the API.
- Replay/snapshot, budget tracker, memory store, feedback loop —
  orthogonal.
- Phase 11 (agent replay from trace) — independent; can land before
  or after.

## Prerequisite audit

Concrete file:line references against jig post-phase-12 main.

### Current `jig.dispatch.run` polling loop

`src/jig/dispatch/client.py`:
- `:269–316` `run(fn_ref, payload, *, dispatch_url, requester,
  timeout_seconds=300, poll_interval=0.5, poll_max_interval=5.0,
  trace_context=None)` → user-visible entry point.
- `:93–108` `_build_submission(...)` — the dict POSTed to
  `/jobs/submit`. **Does not set `callback_url` today**; phase 10
  adds the field.
- `:146–217` `_submit_and_poll(...)` — the polling loop.
  Exponential backoff (0.5s doubles to 5s cap), honors `timeout_seconds`,
  treats 404/5xx as retryable, `ConnectError` as retryable, malformed
  JSON as retryable. This stays as the fallback path.
- `:220–267` `_get_shared_http()` / `aclose()` — lazy shared
  `httpx.AsyncClient` keyed by event loop. Listener will reuse this
  client for submissions.

### Current `jig.sweep`

`src/jig/sweep.py:172–216` `sweep(cases, configs, *, concurrency=1,
sweep_id=None)` — fans out `(case × config)` pairs via
`asyncio.gather` under a `Semaphore(concurrency)`. All runs are local
via `run_agent`; **no dispatch parameter exists today**. `compare()`
at `:141–169` is the single-input variant; same fan-out pattern.

### Smithers side (no change needed)

`smithers/dispatch/callbacks.py:56–80` — `deliver_callback(job)` POSTs
`{job_id, status, result, error}` to `job.callback_url` when set.

`smithers/dispatch/callback_worker.py:43–46` — background sweeper
with durable state, 10 attempts, 60s→3600s exponential backoff,
±25% jitter, idempotent.

**Conclusion:** jig just has to (a) set `callback_url` on submission
and (b) handle the inbound POST. Smithers already delivers reliably.

### Retry primitives already present

`src/jig/core/retry.py:9–25` `with_retry(fn, *, max_attempts=3,
base_delay=1.0, retryable=...)` — exponential backoff helper, takes
a retryability predicate. Phase 10 uses it for listener health
probes.

### Usage-field population on dispatched calls

`src/jig/llm/dispatch.py:265–275` builds `LLMResponse` from
`result.get("usage")`. The callback path resolves to the **same
result dict shape** because smithers hands it to `deliver_callback`
unchanged. No usage-field regression; a single assertion in the
receiver verifies shape parity.

### Tracing

Dispatched calls today accept `trace_context` (src/jig/dispatch/client.py:79)
forwarded to smithers for worker reparenting (phase 9). No per-tick
spans — the caller wraps the whole `run()` call in one span. Phase 10
preserves this: the `dispatch_job` span covers submit → future
resolution (callback path) or submit → poll-loop-exit (fallback),
with the same boundary either way.

### Test infrastructure

`tests/test_dispatch.py`, `tests/test_dispatch_module.py`,
`tests/test_sweep.py` — all use `unittest.mock.AsyncMock`. **No real
HTTP listener test infrastructure today.** Phase 10 adds a dev
dependency (`httpx[test]` brings a `MockTransport` helper, already
transitive; or `aiohttp.test_utils`) and exercises the receiver end
to end by POSTing against the real handler.

### Public-surface delta

`src/jig/dispatch/__init__.py`: current `__all__ = ["DispatchError",
"JobTimeoutError", "aclose", "run"]`. Add `listen`, `stop`,
`CallbackListener`, `ListenerError`. `src/jig/__init__.py:47` already
re-exports `dispatch_run`; add `dispatch_listen`.

## Key design calls

### 1. Listener is a per-process singleton, lazy start

`jig.dispatch.listen(port=0)` starts one HTTP receiver per process.
Callers don't manage its lifecycle directly; `listen()` is idempotent
(returns the existing listener's URL if already running).
`dispatch.stop()` shuts it down. `aclose()` calls `stop()` first.

Port `0` means "OS picks" — `listen()` returns the bound URL so the
caller can see what was chosen.

**`base_url` resolution order:**
1. Explicit `base_url=` kwarg to `listen()`.
2. `$JIG_CALLBACK_HOST` env var → `http://$JIG_CALLBACK_HOST:<port>`.
3. `socket.gethostname()` → `http://<hostname>:<port>`. Works out of
   the box on the homelab's Tailscale network because MagicDNS
   resolves short hostnames (e.g. `otto` → `otto.ts.net`).

The returned URL embeds in every `callback_url` jig sends to
smithers. For CI or loopback-only environments, callers pass
`base_url="http://127.0.0.1"` explicitly — the default falls through
to the host's Tailscale name which CI won't have.

Singleton rationale: a sweep with 500 parallel runs wants one HTTP
server, not 500. Per-call listeners would also fail to work in test
environments where multiple sweeps overlap.

### 2. `dispatch.run` auto-populates `callback_url` when a listener is up

```python
async def run(fn_ref, payload, *, dispatch_url, requester,
              timeout_seconds=300, trace_context=None):
    listener = _active_listener()  # may be None
    if listener is not None:
        callback_url = listener.url_for(job_id_to_be_assigned)
        future = listener.register(job_id_to_be_assigned)
        # submit with callback_url
        # await future with timeout_seconds
    else:
        # existing poll loop
```

Registration is keyed by a UUID4 nonce the caller generates before
submit — the listener holds an `asyncio.Future` per pending nonce.
The submission carries `callback_url=<base>/callbacks/<nonce>?token=...`,
and when that POST arrives the listener looks up the nonce and
resolves the future with the body dict. Generating the nonce locally
(rather than waiting for the `job_id` smithers hands back) closes
the small race window between submit and the first callback attempt.

### 3. Fallback: polling stays

Cases where the listener isn't reachable:
- No listener registered (e.g. single-shot CLI use).
- Listener bound to a port smithers can't reach (NAT, firewall).
- Listener failed its pre-submit health probe.

In those cases, `dispatch.run` uses the existing poll loop verbatim.
The fallback is intentional: it keeps callers who don't opt in from
paying the listener startup cost, and it keeps the fan-out path
resilient when the homelab network misbehaves.

Pre-submit health probe: `await with_retry(listener.health_check,
max_attempts=2, base_delay=0.2, retryable=lambda _: True)`. Two
tries, 0.2s backoff — if the listener's own HTTP endpoint doesn't
respond in under a second, we fall back.

### 4. `sweep(..., dispatch="smithers")` wires it together

```python
async def sweep(cases, configs, *, concurrency=1, sweep_id=None,
                dispatch: str | None = None) -> SweepResult[T]:
    if dispatch == "smithers":
        listener = await dispatch_module.listen()
        try:
            return await _run_sweep(cases, configs, concurrency, sweep_id)
        finally:
            # Only stop if we started it; if caller pre-started a
            # listener, leave it running.
            await dispatch_module.maybe_stop(listener_we_owned)
    return await _run_sweep(cases, configs, concurrency, sweep_id)
```

The sweep body doesn't change — `run_agent` is still per-run, still
under the semaphore. What changes: each agent's `LLMClient` is a
`DispatchClient` (chosen by `from_model("dispatch/<name>")`), and
those dispatch calls now go through the callback path. The
semaphore bounds *concurrent LLM calls*, but waiting is now an
`await future` rather than an `await poll_loop`, so the overhead per
waiting run drops to zero.

`dispatch="smithers"` rather than a bool because future work may add
other dispatch backends (mcclure workers, remote openai via a
gateway); the string leaves room.

### 5. Retry curve

Jig-side retries are limited to listener operations:
- Listener startup: `with_retry(start, max_attempts=3,
  base_delay=1.0)` — port collisions or transient bind errors.
- Listener health probe: as above, `max_attempts=2, base_delay=0.2`.

We do **not** retry callback delivery on the jig side. Smithers'
callback worker owns that path end-to-end. If smithers exhausts its
10 attempts, the job is marked callback-failed and jig's future
times out at `timeout_seconds`. That timeout is already in the
public API.

### 6. Tracing: one span per job

The `dispatch_job` span opens at `run()` entry, closes at future
resolution (or poll exit, or timeout). Under the span:
- `submit` event — included in span attributes, not a child span.
- `callback_received` OR `poll_tick_N` events — attributes on the
  span.

No extra span per poll tick — this matches today's behavior and
keeps the trace DB from bloating when sweeps run.

### 7. Callback payload shape

```json
POST /callbacks/{nonce}?token={secret}
{
  "job_id": "...",
  "status": "complete" | "failed" | "cancelled",
  "result": {...},      // for status=complete
  "error": "..."        // for status=failed
}
```

This matches `smithers/dispatch/callbacks.py` today. The receiver
validates the shape with pydantic (reuse `jig.core.types`) and
resolves the future with either `result` or an exception wrapping
`error`.

### 8. Authentication / secret

Simplest possible: a shared secret in the callback URL path, e.g.
`/callbacks/<job_id>?token=<32-char-random>`. The listener generates
one secret per process start; smithers forwards the URL verbatim.
Not TLS-grade, but keeps accidental crossed-wire callbacks from
different jig processes out of each other's futures.

Leaving an explicit env override `JIG_CALLBACK_SECRET` for operators
who want a fixed value (e.g. for shared-process deployments).

### 9. Multi-process / test isolation

When tests run in parallel (`pytest-xdist`), each worker needs its
own listener port. `listen(port=0)` handles this — OS gives each
worker a distinct port. Tests tear down via `await
dispatch.aclose()` in a fixture.

## Step-by-step plan

Single PR against jig main: `feat/phase-10-callback-sweep-fanout`.

### P1 — Listener module

New `src/jig/dispatch/listener.py`. Uses `aiohttp.web` — ships its own
HTTP server (no separate ASGI runner), has `aiohttp.test_utils.TestServer`
built in for tests, one dep instead of two. Lands as the optional
extra `jig[callback]`:

```toml
# pyproject.toml
[project.optional-dependencies]
callback = ["aiohttp>=3.9"]
```

`listener.py` imports `aiohttp` at module scope and re-raises the
import error with an actionable message:

```python
try:
    from aiohttp import web
except ImportError as exc:
    raise ImportError(
        "jig.dispatch.listen requires aiohttp. Install with `pip install jig[callback]`."
    ) from exc
```

Same guard pattern as `src/jig/llm/anthropic.py:28–29`.

```python
class CallbackListener:
    async def start(self, port: int, host: str, base_url: str | None) -> None: ...
    async def stop(self) -> None: ...
    def url_for(self, job_id: str) -> str: ...
    def register(self, job_id: str) -> asyncio.Future: ...
    def resolve(self, job_id: str, result: dict) -> None: ...
    def fail(self, job_id: str, error: str) -> None: ...
    async def health_check(self) -> None: ...  # HTTP GET to /health
```

Module-level `_active: CallbackListener | None`. `listen()` and
`stop()` wrap the singleton. POST route validates token + body,
resolves the future.

**Tests:** `tests/test_dispatch_listener.py` — start, bind, health,
register, resolve via real HTTP POST, stop, idempotent start/stop,
token mismatch rejection, unknown job_id rejection.

### P2 — `dispatch.run` listener-aware

Rewrite `src/jig/dispatch/client.py:269–316` to branch on
`_active_listener()`:

- Listener path: health-probe → submit with `callback_url` → await
  future with `asyncio.wait_for(timeout_seconds)` → return result.
- Fallback: existing `_submit_and_poll`.

Thread the trace span through both branches so the `dispatch_job`
span covers whichever path runs.

**Tests:** extend `tests/test_dispatch_module.py` with listener-on
and listener-off cases. Use a real listener in both (it's cheap to
start one per test), plus a `monkeypatch` that asserts
`_submit_and_poll` wasn't called when the callback path succeeded.

### P3 — Sweep `dispatch` kwarg

`src/jig/sweep.py:172–216` — add `dispatch: str | None = None`. When
`"smithers"`, ensure a listener is running for the sweep's duration.
Document that the caller can pre-start a listener with
`jig.dispatch.listen()` to share it across sweeps.

**Tests:** `tests/test_sweep.py` — new test class
`TestSweepDispatch` that fakes a smithers server + a jig listener
end-to-end. Run a 3-case × 2-config sweep; assert every run resolved
via the callback path, not the poll fallback.

### P4 — Public exports + docs

- `src/jig/dispatch/__init__.py`: export `listen`, `stop`,
  `CallbackListener`, `ListenerError`.
- `src/jig/__init__.py`: re-export `dispatch_listen`, `dispatch_stop`.
- `docs/roadmap.md` status table: phase 10 → merged.
- New `docs/dispatch.md` (or extend an existing doc) — one-page
  explainer of when to use listener vs polling, how to configure
  `base_url` / port, the callback token env var.

### P5 — Integration smoke against live smithers

Runbook, not code. On the target host:

1. Bring up smithers workers (`docker compose up` in smithers
   repo).
2. In a jig venv, run a scripted sweep: 20 cases, dispatch=smithers.
3. Watch smithers logs for callback POSTs; assert each job's
   callback delivered on the first attempt (retry-count on the
   callback worker should stay at 1 for the sweep).
4. Kill jig mid-sweep; confirm smithers' callback_worker tries up
   to 10 times, then marks callback-failed. Restart jig, rerun; new
   listener picks up new jobs cleanly.

## Risks and edge cases

- **Port reachability.** Smithers must reach jig's listener. In the
  homelab that's Tailscale-trivial; in CI/test environments, the
  listener binds loopback. The fallback polling path covers both.
  Risk: operator sets `base_url` wrong and callbacks silently fail
  → every job times out at `timeout_seconds`. Mitigation: listener
  does a startup self-POST to `/health` and logs the reachable URL
  loud and clear. Also: if we see zero callbacks land within the
  first 3 submissions, log a warning suggesting polling fallback.

- **Missed callback delivery.** Smithers retries 10× over ~10
  hours. Jig's `timeout_seconds` defaults to 300s. For sweeps this
  is usually fine; for long-running single jobs, operators pass a
  larger `timeout_seconds`. No change needed unless we see this
  bite in practice.

- **Process crash mid-sweep.** Futures are in-memory; a crashed jig
  loses them. On restart, the new jig process's listener has no
  record of the old job ids and rejects inbound callbacks. Smithers
  records the 404, retries 10×, gives up. The operator reruns
  affected cases. Acceptable — recording jobs durably is out of
  scope; that's a follow-up.

- **Sweep with mixed-dispatch configs.** A config whose LLM
  isn't routed through smithers (e.g. `claude-*` direct) runs
  locally regardless of `dispatch="smithers"`. That's correct:
  `dispatch="smithers"` says "use the listener when the path would
  otherwise poll," not "force everything remote."

- **Test flakiness from port binding.** xdist workers rarely
  collide on `port=0`, but pytest tearing down before `stop()`
  leaves ports in TIME_WAIT. Conftest fixture `await aclose()` on
  session teardown catches this.

- **Listener dep surface.** Locked to `aiohttp` as the single
  callback-server dep, behind `jig[callback]`. Users who don't call
  `jig.dispatch.listen()` never import it. The `[anthropic]` /
  `[openai]` precedent means reviewers already know the pattern.

- **Security posture of the token.** Shared-secret token in the
  URL query string. A motivated attacker on the same Tailscale
  network could enumerate, but the blast radius is resolving a
  future with a garbage result — the agent consumes it and
  proceeds. Accepted as the phase-10 posture; HMAC is explicitly
  not planned (see Follow-ups).

- **Backoff curve on listener startup.** Three attempts at 1s/2s/4s
  = 7s before fallback. For sweeps this is negligible. For
  single-shot `dispatch.run` calls it adds latency only on the
  error path.

## Testing strategy

**Coverage target:** listener starts/stops cleanly, registers
futures, resolves via real HTTP POST, rejects mismatched tokens,
rejects unknown job_ids, times out cleanly, falls back to polling
when unreachable.

**Integration (one test):** `tests/test_sweep.py` — spin up a fake
smithers server (httpx MockTransport or aiohttp.test_server), a
real jig listener, run a 6-run sweep (`2 cases × 3 configs`),
assert:
- Each submission carries a `callback_url`.
- The fake smithers invokes the callback with a synthetic result.
- Every future resolves without hitting the poll path.
- The `dispatch_job` span count equals 6.
- `_submit_and_poll` was never called (guard via monkeypatch).

**No benchmark gate.** Phase 10's value is "fewer coroutines per
sweep" — measurable in asyncio stats if the reviewer wants it, but
not a merge gate.

## Short implementation order

1. P1 — listener module + receiver + tests (largest chunk).
2. P2 — dispatch.run branch + tests.
3. P3 — sweep kwarg + end-to-end test.
4. P4 — exports + docs.
5. P5 — live-smithers smoke runbook.

All in one PR against jig main. Target: ~800–1000 LoC (listener is
~300, tests ~400, dispatch/sweep changes ~200).

## Follow-ups (explicitly out of scope)

- Durable job-id → future map across restarts.
- HMAC-signed callback payloads. Shared-secret token is the
  accepted posture; revisit only if a concrete threat emerges.
- Webhook-style subscription across multiple smithers clusters.
- Extending `dispatch=` to other backends beyond `"smithers"`.
- Phase 14 sweep driver that coordinates across cross-repo sweeps.
