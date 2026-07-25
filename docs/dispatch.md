# Dispatch

> **Advanced / optional.** Core jig usage (polling via `DispatchClient` or `jig.dispatch_run`) works without any extra setup. The callback listener is an optional performance optimisation — it requires the `callback` extra (`pip install 'jig[callback]'`) and a process that can receive inbound HTTP from your worker network.

jig submits deterministic functions (`jig.dispatch.run`) and LLM
inference (`jig.llm.DispatchClient`) to [smithers] workers. Two wait
strategies are available:

- **Polling** (default) — submit, then poll `GET /jobs/{id}` with
  exponential backoff. Fine for one-shot calls; expensive when a
  sweep dispatches dozens of jobs concurrently because each job
  holds its own polling coroutine.
- **Callback** — submit with a `callback_url`, then `await` an
  `asyncio.Future` that a small HTTP receiver resolves when smithers
  posts the result. One HTTP receiver per process, N futures cheap.

## Quick start

```python
import jig

# Start the receiver (once per process).
await jig.dispatch_listen()

# Any subsequent dispatch.run() / DispatchClient.complete() uses the
# callback path automatically while the listener is up.
result = await jig.dispatch_run("demo.fn:echo", {"x": 1})

# Sweeps opt in with dispatch="smithers"; the listener is managed for
# the sweep's lifetime if one isn't already running.
sweep_result = await jig.sweep(
    cases, configs, concurrency=32, dispatch="smithers",
)

await jig.dispatch_stop()  # or jig.dispatch.aclose() at process exit
```

## Install

The listener needs `aiohttp`, carried behind an optional extra:

```bash
pip install 'jig[callback]'
```

Callers that never invoke `dispatch_listen()` don't import `aiohttp`.

## Configuration

| Env var                  | Default                  | Description                                               |
|--------------------------|--------------------------|-----------------------------------------------------------|
| `JIG_CALLBACK_HOST`      | `socket.gethostname()`   | Hostname smithers posts back to. Override for CI / NAT.   |
| `JIG_CALLBACK_SECRET`    | fresh random per process | Shared secret embedded in the callback URL's `?token=`.   |

`listen()` kwargs:

- `port=0` — OS picks; `listener.port` returns the bound value.
- `host="0.0.0.0"` — bind interface.
- `base_url=None` — override the URL embedded in `callback_url`.
  Falls back to `JIG_CALLBACK_HOST`, then `socket.gethostname()`.

## Fallback behavior

When the listener isn't reachable (stopped, unhealthy, or never
started), dispatch calls silently fall back to polling:

- `_current_listener()` returns `None` → polling.
- Listener is registered but its `GET /health` probe fails →
  `_submit_and_poll` logs the reason and polls.

Callers never see a "listener broken" error; they just pay the
polling cost on that call.

## Timeout and cancellation ownership

Smithers owns the execution deadline and the worker lifecycle. Jig sends
`timeout_seconds` with the job, then waits an additional cleanup grace
(10 seconds by default) for smithers to cancel the worker and publish a
terminal status. The grace does not extend the worker's execution budget; it
prevents a caller retry from racing the previous job's cleanup.

Jig owns the request/response caller's intent:

- Cancelling `dispatch.run()` or `DispatchClient.complete()` sends
  `DELETE /jobs/{id}` to smithers and awaits the response before propagating
  `CancelledError`.
- If no terminal status arrives by the execution deadline plus cleanup grace,
  Jig sends the same cancellation request before raising `JobTimeoutError`.
- `dispatch.run(..., cancel_on_timeout=False)` is the explicit opt-out for a
  function job that should remain durable after this caller stops waiting.
  Coroutine cancellation still cancels the remote job; detached submission is
  a separate lifecycle and should not use this request/response helper.

`cleanup_grace_seconds` is configurable on both `dispatch.run()` and
`DispatchClient`; it defaults to 10 seconds and is clamped to zero when a
negative value is supplied. LLM dispatch always cancels on client timeout.
Jig only talks to the smithers coordinator—smithers is responsible for
propagating cancellation to Frink, McClure, or another selected worker.

## Security posture

The shared-secret token in the URL query string is not HMAC-grade.
The threat model is accidental callback crossing between jig
processes on the same trusted network, not an active attacker.
Smithers delivers callbacks over plain HTTP to the configured callback
host (set via `JIG_CALLBACK_HOST` or the `base_url` kwarg).

## Tracing

Every dispatched call emits a single trace span covering submit →
resolution. The span's attributes include the chosen path
(`callback` or `poll`), the job id, and (on failure) the terminal
status. Poll ticks do not produce child spans.

When an agent executes a `Tool(dispatch=True)`, `ToolRegistry` forwards
the active tool execution context to the dispatch client as
`trace_context`, with the local tool-call span as the worker parent.
Tools can also add payload fields by overriding
`dispatch_payload_extra(context, arguments)`.

## Tool lifecycle hooks

A dispatched tool can observe and intervene at four points. All are
duck-typed — define the method to opt in, omit it and nothing changes.

| hook | when | raising |
|---|---|---|
| `pre_dispatch(arguments, context)` | before `dispatch_payload_extra`, before anything ships | rejects the call; message becomes `ToolResult.error` prefixed `gate:` |
| `on_dispatch_submitted(job_id)` | after the job is accepted, before the terminal wait | logged and swallowed |
| `on_dispatch_result(result, context)` | after the job returns successfully | becomes `ToolResult.error` — **not** swallowed |
| `on_dispatch_error(error, context)` | after the job fails | logged and swallowed |

`pre_dispatch`, `on_dispatch_result`, and `on_dispatch_error` may be sync
or async; the registry awaits an awaitable return. `on_dispatch_submitted`
is called synchronously by the dispatch client and its return value is
never awaited, so defining it `async` silently does nothing.

`pre_dispatch` shares `dispatch_payload_extra`'s flexible calling
convention: its parameters may be named `context`/`tool_context`/`ctx` and
`arguments`/`args`, in either order, or omitted entirely. The other three
take fixed positional arguments as shown.

`on_dispatch_result` receives the **raw** worker return value, not the
string `ToolResult.output` is coerced from — the hook exists precisely
because that coercion loses structure. Return `None` to pass the result
through, or a value to replace it. It runs *outside* the registry's
`execute_timeout`: that budget belongs to the remote job, and local
reconciliation must not be clipped by it. Its exceptions are deliberately
not swallowed — if reconciliation failed, downstream state is unrecorded
and the model must not proceed as though the result were usable.

`on_dispatch_error` fires for every way a dispatched call can fail after
shipping: `DispatchError`, `JobTimeoutError`, `asyncio.TimeoutError`, any
other exception, and business failures (below). It does **not** fire for a
`pre_dispatch` rejection — nothing was dispatched — nor when
`on_dispatch_result` itself raises, which is a reconciliation failure
rather than a dispatch failure.

### Business failures

A dispatched function that runs to completion but whose *outcome* failed
has no way to say so by returning: any return is a success, and raising
loses the partial payload. `jig.dispatch.tool_error` reserves a key for it:

```python
from jig.dispatch import tool_error

def run_study(**kwargs):
    ...
    return tool_error("no trades generated", trials=200, best_sharpe=None)
```

The registry converts this into `ToolResult.error` (the message) while the
sibling fields survive into `ToolResult.output`, and synthesizes a
`DispatchBusinessError` through `on_dispatch_error` so one reconciliation
implementation handles every failure mode. `DispatchBusinessError` is
deliberately **not** a `DispatchError` subclass: `DispatchError` means the
job never produced a result, this means it did and the result is a
failure. Branch on `isinstance` when that distinction matters.

`on_dispatch_result` never sees a business failure — it is a success hook,
and a business failure is not a success.

[smithers]: https://github.com/rankonelabs/smithers
