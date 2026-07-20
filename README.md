# jig

Minimal agent framework. Six interfaces, two execution models, everything swappable.

A jig holds the pieces while you work on them. It doesn't have opinions about your pieces — just the shapes of the plugs between them.

## Install

```bash
uv add 'jig @ git+https://github.com/rankonelabs/jig.git'                          # core only
uv add 'jig[anthropic] @ git+https://github.com/rankonelabs/jig.git'               # + Anthropic adapter
uv add 'jig[ollama] @ git+https://github.com/rankonelabs/jig.git'                  # + Ollama adapter
uv add 'jig[all] @ git+https://github.com/rankonelabs/jig.git'                     # everything
```

## Quick start

These examples require `jig[anthropic,ollama]` (or `jig[all]`) and a running Ollama server.

Smallest runnable agent — no memory:

```python
import asyncio
from jig import AgentConfig, run_agent
from jig.llm import AnthropicClient
from jig.feedback import SQLiteFeedbackLoop
from jig.tracing import StdoutTracer
from jig.tools import ToolRegistry

config = AgentConfig(
    name="my-agent",
    description="A simple agent",
    system_prompt="You are a helpful assistant.",
    llm=AnthropicClient(model="claude-sonnet-4-20250514"),
    feedback=SQLiteFeedbackLoop(),
    tracer=StdoutTracer(),
    tools=ToolRegistry(),
)

result = asyncio.run(run_agent(config, "What's the weather like?"))
print(result.output)
```

With local memory (SQLite + embeddings):

```python
import asyncio
from jig import AgentConfig, run_agent
from jig.llm import AnthropicClient
from jig.memory import LocalMemory
from jig.feedback import SQLiteFeedbackLoop
from jig.tracing import StdoutTracer
from jig.tools import ToolRegistry

store, retriever = LocalMemory()

config = AgentConfig(
    name="my-agent",
    description="A simple agent",
    system_prompt="You are a helpful assistant.",
    llm=AnthropicClient(model="claude-sonnet-4-20250514"),
    store=store,
    retriever=retriever,
    feedback=SQLiteFeedbackLoop(),
    tracer=StdoutTracer(),
    tools=ToolRegistry(),
)

result = asyncio.run(run_agent(config, "What's the weather like?"))
print(result.output)
```

## Interfaces

| Interface | Purpose | Adapters |
|---|---|---|
| `LLMClient` | LLM completions | Anthropic, OpenAI, OpenRouter, Gemini, Ollama, Dispatch (optional Smithers backend) |
| `MemoryStore` | Persistence + session history | Local (SQLite + embeddings), Honcho\*, Zep\* |
| `Retriever` | Prompt-context strategy | DenseRetriever (embeddings), HonchoMemory\*, ZepMemory\* |
| `FeedbackLoop` | Score tracking + eval export | SQLite (integration-tested end-to-end) |
| `Grader` | Auto-score outputs | LLM Judge, Heuristic, Ground Truth, Composite |
| `TracingLogger` | Structured spans | SQLite, Stdout |

Plus `ToolRegistry` (concrete) and `Tool` (abstract) for tool use.

\* Honcho and Zep adapters exist but are not covered by integration tests in this repo.
The SQLite path (`LocalMemory`, `SQLiteFeedbackLoop`) is the verified baseline.
For older feedback databases, see [SQLite feedback maintenance](docs/sqlite-feedback-maintenance.md).

## Two execution models

### `run_agent()` — LLM-in-the-loop

The LLM decides what to do next. You provide tools and a system prompt; jig runs the completion + tool loop.

1. Start trace
2. Resolve system prompt (string or async callable)
3. Query memory for relevant context
4. Query feedback for quality signals from past runs
5. Assemble `CompletionParams` (system separate from messages)
6. LLM call + tool execution loop
7. Store output in memory
8. Auto-grade if grader configured
9. Close trace, return `AgentResult`

### `run_pipeline()` — orchestrator-controlled

You define the step sequence; jig wraps it with tracing, grading, and feedback. The LLM (if any) is a transform *within* a step, not the decision-maker.

```python
from jig import PipelineConfig, Step, run_pipeline
from jig.tracing import StdoutTracer

async def fetch(ctx):
    return await get_document(ctx["input"])

async def summarize(ctx):
    return await llm_summarize(ctx["fetch"])

async def score(ctx):
    return evaluate_quality(ctx["summarize"])

result = await run_pipeline(
    PipelineConfig(
        name="summarizer",
        steps=[
            Step(name="fetch", fn=fetch),
            Step(name="summarize", fn=summarize),
            Step(name="score", fn=score),
        ],
        tracer=StdoutTracer(),
    ),
    input="https://example.com/article",
)
print(result.output)            # score result
print(result.step_outputs)      # {"fetch": ..., "summarize": ..., "score": ...}
```

Each step receives a context dict (`ctx`) and returns anything. The framework:
- Stores the return value in `ctx[step.name]` for downstream steps
- Traces every step as a `PIPELINE_STEP` span
- Short-circuits on error if `is_err` / `extract_err` are configured
- Skips steps conditionally via `skip_when`
- Grades per-step (via `Step.grader`) or pipeline-wide (via `PipelineConfig.grader`)
- Stores graded results via `FeedbackLoop` if configured

#### `map_pipeline()`

Runs `run_pipeline` per item with a shared parent trace. Optionally grades the batch.

```python
from jig import map_pipeline

result = await map_pipeline(config, items=[doc1, doc2, doc3])
# result.results — list of PipelineResult, one per item
```

#### Nested pipelines

A step can call `run_pipeline` internally. Pass `ctx["_tracer"]` and `ctx["_span_id"]` to nest spans under the parent trace.

```python
async def inner_step(ctx):
    return await run_pipeline(
        inner_config,
        input=ctx["previous_step"],
        _parent_span_id=ctx["_span_id"],
    )
```

## CompletionParams

LLM adapters receive a single `CompletionParams` object — typed core fields plus a provider escape hatch:

```python
CompletionParams(
    messages=[...],              # universal
    system="You are...",         # universal, handled per-provider
    tools=[...],                 # universal
    temperature=0.7,             # universal
    max_tokens=4096,             # universal
    provider_params={"top_k": 40},  # forwarded to SDK, provider-specific
    response_format={               # optional, portable structured output
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object", "...": "..."}},
    },
)
```

### Structured output (`response_format`)

`response_format` carries the OpenAI-compatible envelope
(`{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`).
Adapters never normalize or mutate it — each adapter's own capability
determines what happens with a non-null value:

| Adapter | Behavior |
| --- | --- |
| Dispatch (→ vLLM) | Forwarded unchanged — an opaque transport; the smithers worker's executor owns validation. |
| OpenAI | Forwarded unchanged as the `response_format` request field. |
| OpenRouter | Forwarded unchanged (shares `OpenAIClient.complete()`). |
| Ollama (direct + dispatched) | Validated, then translated: the inner `json_schema.schema` object becomes the top-level `format` field/kwarg. Wrapper metadata (`name`, `description`, `strict`) is not sent upstream. |
| Anthropic, Gemini, and every other adapter | Rejected — `response_format` is not implemented there yet. |

Omitting `response_format` leaves every existing request byte-identical to today. A malformed or unsupported value raises `UnsupportedResponseFormatError` (a `ValueError` subclass, exported from `jig`) before any request is made, so an unsupported constraint fails loudly instead of silently running unconstrained. On most adapters that error propagates directly; on `AnthropicClient` it is currently wrapped in `JigLLMError` instead (its request-preparation code predates this contract and wasn't updated to let the typed error through) — the call still fails before any request, just without the precise error type:

```python
from jig import CompletionParams, UnsupportedResponseFormatError

try:
    await client.complete(CompletionParams(messages=[...], response_format={"type": "text"}))
except UnsupportedResponseFormatError as e:
    ...  # bad shape or unsupported adapter — caller's contract violation, not a provider failure
```

## Project layout

```
src/jig/
├── core/           # types, runner, pipeline, errors, retry, prompt builder
├── llm/            # anthropic, openai, openrouter, google, ollama, dispatch adapters
├── memory/         # local (sqlite+embeddings), honcho, zep
├── feedback/       # feedback loop, llm judge, heuristic, ground truth, composite
├── tracing/        # sqlite tracer, stdout tracer, federated tracer
├── tools/          # registry + common tools
├── dispatch/       # dispatch client and callback listener
├── observability/  # structured logging helpers
├── eval/           # eval datasets and calibration
├── replay/         # replay runner, diff, snapshot
├── budget.py       # budget tracking
├── sweep.py        # compare and sweep
├── sweep_stats.py  # sweep result aggregation
└── regression.py   # regression testing
```

## Contributing

```bash
uv run --extra dev --extra callback pytest
```
