# Dispatch

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

## Security posture

The shared-secret token in the URL query string is not HMAC-grade.
The threat model is accidental callback crossing between jig
processes on the same network, not an active attacker on Tailscale.
Smithers delivers callbacks over plain HTTP to Tailscale hostnames —
the same trust assumptions as the rest of the dispatch protocol.

## Tracing

Every dispatched call emits a single trace span covering submit →
resolution. The span's attributes include the chosen path
(`callback` or `poll`), the job id, and (on failure) the terminal
status. Poll ticks do not produce child spans.

[smithers]: https://github.com/rankonelabs/smithers
